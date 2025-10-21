import numpy as np
import torch
import mujoco
from mujoco import viewer
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
import csv

sys.path.append("./models/biLSTM")
from biLSTM_module import biLSTM
sys.path.append("./models/MLP")
from MLP import MLPModel
sys.path.append("./utils")
from getDataset import SimulationDataset


dataset = SimulationDataset('./simulationData/target.npz')
filepath = "./res/vari_cable_mpc.xml"
model = mujoco.MjModel.from_xml_path(filepath)
data = mujoco.MjData(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# biLSTM model
nn_model = biLSTM()
nn_model.to(device)
nn_model.load_state_dict(torch.load("./models/biLSTM/models/BEST_bi_LSTM_2_rel_vel_bs128_hs256_lr00001_nl3_epochs50_10k.pth", weights_only=True, map_location=device))
nn_model.eval()


horizon = 5  # Horizon
dt = 0.02
control_dim = 4
NUM_SAMPLES = 100 # Number of samples
TEMPERATURE = 0.002 # Temperature
STD_DEV = torch.tensor([0.2, 0.6, 0.05, 3], device=device)
Q = torch.eye(18, device=device) * 150
R = torch.eye(4, device=device) * 5

def get_state_from_mujoco(model, data, dataset, device, prev_cable_pos=None):
    #Get body ids
    body_index = model.body('Box_first').id
    cabel_first_index = model.body('CB0').id
    cabel_last_index = model.body('cube_body_1').id  # "b_Last"

    #Get EE State
    ee_euler_np = data.qpos[body_index + 2:body_index + 5]  # (3,)
    ee_pos_np = data.xpos[body_index]  # (3,)
    ee_pos_combined_np = np.hstack([ee_pos_np[:2], ee_euler_np[:2]])  # (4,)
    ee_vel_np = data.qvel[body_index - 1: body_index + 3]  # (4,)


    if prev_cable_pos is not None:
        prev_cable_pos = prev_cable_pos.to(device)

    #Get DLO State
    xpos_np = data.xpos[cabel_first_index:cabel_last_index]  # (N, 3), N number of markers
    coords_np = xpos_np[:, :2]  # (N, 2)
    coords_filtered_np = coords_np[6:-3]  # (N-9, 2)
    coords_reduced_np = coords_filtered_np[::5]    # (M, 2), z.B. (8, 2)

    ee_pos = torch.from_numpy(ee_pos_combined_np).float().to(device)
    ee_vel = torch.from_numpy(ee_vel_np).float().to(device)
    cable_pos = torch.from_numpy(coords_reduced_np).float().to(device)

    #Get Cable velocity
    if prev_cable_pos is not None:
        cable_vel = cable_pos - prev_cable_pos  # (9, 2)
    else:
        cable_vel = torch.zeros_like(cable_pos)  # (9, 2)

    #Create Robot State
    current_robot_state_pos = ee_pos.view(-1) # Shape: (1, 4)
    current_robot_state_vel = ee_vel.view(-1)  # Shape: (1, 4)
    current_robot_state = torch.cat((current_robot_state_pos, current_robot_state_vel), dim=0)

    #Create Dlo State
    cable_pos_flat = cable_pos.view(-1)
    cable_vel_flat = cable_vel.view(-1)
    current_dlo_state = torch.cat((cable_pos_flat, cable_vel_flat), dim=0)

    #Create State
    state = torch.cat((current_robot_state, current_dlo_state), dim=0)
    return state, cable_pos


def forward_model_batch(x_batch, u_batch, model, pred_mocap_ids, data, dt=0.02):
    """
    Batch version of the forward_model function.
    Processes a batch of states and control signals simultaneously.
    Args:
        x_batch: Tensor of shape (batch_size, state_dim)
        u_batch: Tensor of shape (batch_size, control_dim)
    Returns:
        x_next_batch: Tensor of shape (batch_size, state_dim)
    """

    batch_size = x_batch.shape[0]

    # EE_Pos and EE_Vel States
    ee_pos = x_batch[:, 0:4]          # (batch_size, 4)
    ee_vel = x_batch[:, 4:8]          # (batch_size, 4)

    # DLO States
    feature_pos = x_batch[:, 8:26].view(batch_size, 9, 2)   # (batch_size, 9, 2)
    feature_vel = x_batch[:, 26:44].view(batch_size, 9, 2)  # (batch_size, 9, 2)

    # 2. Calculate new Velocity and Position
    ee_vel_new = ee_vel + u_batch  # (batch_size, 4)
    v_max = 0.1
    ee_vel_new = torch.clamp(ee_vel_new, -v_max, v_max)
    ee_pos_new = ee_pos + ee_vel_new * dt  # (batch_size, 4)

    # 3. DLO Norm
    eps = 1e-9
    l2_norm_cable_pos = torch.norm(feature_pos, p=2, dim=2, keepdim=True)  # (batch_size, 9, 1)

    # EE Normalization
    ee_pos_normalized = (ee_pos - dataset.EE_pos_mean.to(device)) / (dataset.EE_pos_std.to(device) + eps)
    ee_vel_normalized = (ee_vel - dataset.EE_vel_mean.to(device)) / (dataset.EE_vel_std.to(device) + eps)

    # Kabel Normalization
    cable_pos_normalized = feature_pos / (l2_norm_cable_pos + eps)  # (batch_size, 9, 2)
    cable_vel_normalized = (feature_vel - dataset.cable_vel_mean_2D.to(device)) / (dataset.cable_vel_std_2D.to(device) + eps)

    # 4. Preproccess NN Inputs
    robot_state_pos = ee_pos_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
    robot_state_vel = ee_vel_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
    robot_state = torch.cat((robot_state_pos, robot_state_vel), dim=2)  # (batch_size, 9, 8)

    # Combine states
    dlo_state = torch.cat((cable_pos_normalized, cable_vel_normalized), dim=2)  # (batch_size, 9, 4)
    inputs = torch.cat((robot_state, dlo_state), dim=2)  # (batch_size, 9, 12)

    # Reshape: (batch_size * 9, 12)
    inputs = inputs.view(-1, 12)

    # 5. Batch-Inference
    with torch.no_grad():
        norm_predicted_velocity = nn_model(inputs)  # (batch_size * 9, 2)

    # Reshape: (batch_size, 9, 2)
    norm_predicted_velocity = norm_predicted_velocity.view(batch_size, 9, 2)

    # 6. Denormalization of the predicted velocity
    cable_vel_std_2D = dataset.cable_vel_std_2D.to(device)
    cable_vel_mean_2D = dataset.cable_vel_mean_2D.to(device)
    predicted_velocity = norm_predicted_velocity * cable_vel_std_2D + cable_vel_mean_2D

    # 7. Predicted Feature Position
    predicted_feature_pos = feature_pos + predicted_velocity  # (batch_size, 9, 2)
    # 8. Create next state
    x_next_batch = torch.cat([
        ee_pos_new,
        ee_vel_new,
        predicted_feature_pos.view(batch_size, -1),
        predicted_velocity.view(batch_size, -1)
    ], dim=1)  # (batch_size, 44)

    return x_next_batch, predicted_feature_pos


def cost_function_batch(x_batch, u_batch, p_target, Q, R, u_prev_batch=None, S=None):
    # x_batch: (batch_size, state_dim)
    # u_batch: (batch_size, control_dim)
    # u_prev_batch: (batch_size, control_dim) or None

    batch_size = x_batch.shape[0]
    p = x_batch[:, 8:26].view(batch_size, -1)  # (batch_size, 18)
    diff = p - p_target  # (batch_size, 18)

    # Shape cost: 0.5 * diff^T Q diff
    shape_cost = 0.5 * (diff @ Q * diff).sum(dim=1)  # (batch_size,)

    # Control cost: 0.5 * u^T R u
    control_cost = 0.5 * (u_batch @ R * u_batch).sum(dim=1)  # (batch_size,)


    return shape_cost + control_cost



def mppi_optimize(x0, p_target, u_init=None, u_prev=None):
    global Q, R, S

    if u_init is None:
        u_init = torch.zeros((horizon, control_dim), device=device)
    else:
        u_init = torch.roll(u_init, -1, dims=0)
        u_init[-1] = 0

    all_costs = []
    all_predicted_positions = []

    p_target_tensor = p_target.to(device).flatten()

    for mppi_iter in range(1):
        # 1. Generate Samples
        noise = torch.randn((NUM_SAMPLES, horizon, control_dim), device=device) * STD_DEV  # (NUM_SAMPLES, horizon, control_dim)
        samples = u_init.unsqueeze(0) + noise  # (NUM_SAMPLES, horizon, control_dim)

        # 2. Batch-Simulation
        x_batch = x0.clone().repeat(NUM_SAMPLES, 1)  # (NUM_SAMPLES, state_dim)
        total_costs = torch.zeros(NUM_SAMPLES, device=device)
        u_prev_batch = u_prev.clone().repeat(NUM_SAMPLES, 1) if u_prev is not None else None

        for t in range(horizon):
            # Get control signal for current timestep
            u_t = samples[:, t]  # (NUM_SAMPLES, control_dim)

            # Batch Prediction
            x_batch, predicted_feature_pos = forward_model_batch(x_batch, u_t, model, pred_mocap_ids, data, dt=0.02)

            # Cost Calculation
            total_costs += cost_function_batch(x_batch, u_t, p_target_tensor, Q, R, u_prev_batch)

            # Update u_prev_batch
            u_prev_batch = u_t.clone()

            all_predicted_positions.append(predicted_feature_pos[0].cpu().numpy())

        # 3. Weight calculation
        min_cost = torch.min(total_costs)
        weights = torch.exp(-(total_costs - min_cost) / TEMPERATURE)
        weights /= torch.sum(weights)  # Normalisiere

        # 4. Update of control (Tensor-Operationen)
        weights = weights.view(-1, 1, 1)  # (NUM_SAMPLES, 1, 1)
        u_init = torch.sum(weights * samples, dim=0)  # (horizon, control_dim)

        all_costs.append(torch.mean(total_costs).item())

    return u_init.cpu().numpy(), all_costs, all_predicted_positions


x_current, cable_pos_current = get_state_from_mujoco(model, data, dataset, device)
random_idx = random.randint(0, len(dataset) - 1)
inputs_dumm, p_target = dataset[random_idx]
p_target = dataset.get_absolute_coords(random_idx)
p_target = torch.tensor(p_target).view(9, 2)
u_init = None
error_list = []

NUM_TRIALS = 100
MAX_ITERATIONS = 1500  # 30 seconds

# Create lists to store the results
trial_results = []
real_positions_list = []
predicted_positions_list = []

# Mujoco
with mujoco.viewer.launch_passive(model, data) as viewer:

    target_body_names = [f"target_body_{i}" for i in range(1, 10)]
    pred_body_names = [f"pred_body_{i}" for i in range(1, 10)]
    target_mocap_ids = {}
    pred_mocap_ids = {}

    for name in target_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mocap_id = model.body_mocapid[body_id]
        target_mocap_ids[name] = mocap_id

    for name in pred_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mocap_id = model.body_mocapid[body_id]
        pred_mocap_ids[name] = mocap_id

    # Trials Loop
    for trial in range(NUM_TRIALS):
        print(f"Start Experiment {trial+1}/{NUM_TRIALS}")

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        cable_pos_current = None

        # Create Sphere for Targets positions
        for i, pt in enumerate(p_target, start=1):
            x, y = pt
            z = 0.15
            mocap_id = target_mocap_ids[f"target_body_{i}"]
            data.mocap_pos[mocap_id] = np.array([x, y, z])

        converged = False
        steps_to_convergence = MAX_ITERATIONS
        start_time = time.time()

        error_list = []
        ee_positions = []
        all_costs_per_step = []

        for step in range(MAX_ITERATIONS):
            # get State
            x_current, cable_pos_current = get_state_from_mujoco(model, data, dataset, device, cable_pos_current)
            ee_pos_current = x_current[:2].cpu().numpy()  # z.B. (x,y)
            ee_positions.append(ee_pos_current)

            # MPPI-Optimization
            U_opt, costs, predicted_positions = mppi_optimize(x_current, p_target)
            all_costs_per_step.append(costs)

            # Set Control signals
            data.ctrl[0] = np.clip(U_opt[0,0], -0.1, 0.1)
            data.ctrl[1] = np.clip(U_opt[0,1], 0.0, 3)
            data.ctrl[5] = np.clip(U_opt[0,3], -3, 3)

            # Calculate error
            p_current_now = x_current[8:26].reshape(-1, 2).cpu().numpy()
            p_target_cpu = p_target.cpu().numpy()
            error = np.linalg.norm(p_current_now - p_target_cpu, axis=1)
            error_list.append(error)

            real_positions_list.append(p_current_now)      # p_current_now: acutal Marker Positions, shape (N,2)
            predicted_positions_list.append(predicted_positions)  # predicted_positions: Predictions, shape (N,2)
            # Simualte 20 timestep (0.02s)
            for _ in range(20):
                mujoco.mj_step(model, data)
                viewer.sync()

            # Check convergence
            if np.all(np.abs(error) < 0.02):
                elapsed_time = time.time() - start_time
                converged = True
                steps_to_convergence = step + 1
                break

        # Save result
        trial_results.append((converged, steps_to_convergence))
    viewer.close()


num_converged = sum(1 for (conv, _) in trial_results if conv)


with open("trial_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Trial", "Converged", "Iterations_to_Convergence"])
    for i, (conv, steps) in enumerate(trial_results, start=1):
        writer.writerow([i, conv, steps])

all_costs_per_step_array = np.array(all_costs_per_step)

mean_costs = np.mean(all_costs_per_step_array, axis=1)
std_costs = np.std(all_costs_per_step_array, axis=1)

steps = np.arange(len(mean_costs))

euklidian_error = np.array(error_list)
total_error = np.mean(euklidian_error, axis=1)


plt.figure(figsize=(8, 5))
plt.plot(steps, total_error, label="Fehler", color="b")
plt.xlabel("Optimization Step")
plt.ylabel("Euklidian Error")
plt.title("Evolution of the euklidian error per Optimization Step")
plt.legend()
plt.grid()
plt.savefig("error_plot.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(steps, mean_costs, label="Mean Costs", color="b")
plt.fill_between(steps, mean_costs - std_costs, mean_costs + std_costs, color="b", alpha=0.2)
plt.xlabel("Optimization Step")
plt.ylabel("Costs")
plt.title("Evolution of Costs per Optimization Step")
plt.legend()
plt.grid()
plt.savefig("kosten_plot.png", dpi=300, bbox_inches="tight")
plt.close()


real_positions = np.array(real_positions_list)
predicted_positions = np.array(predicted_positions_list)
predicted_first = predicted_positions[:, 0, :, :]

T = real_positions.shape[0]

for t in range(T):
    plt.figure(figsize=(8, 6))

    # Actual positions (blue)
    plt.scatter(real_positions[t, :, 0], real_positions[t, :, 1],
                color="blue", label="Real", s=50)

    # Predictions (red)
    plt.scatter(predicted_first[t, :, 0], predicted_first[t, :, 1],
                color="red", label="Pred", s=50)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Comparison of positions at time step {t}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"positions_timestep_{t}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Plots saved.")