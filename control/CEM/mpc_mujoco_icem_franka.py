import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
import sys
import matplotlib.pyplot as plt

sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\02_iCEM\iCEM\utils') 
from IK_NS import InverseKinematicsController
from getDataset import SimulationDataset
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\02_iCEM\iCEM\models\biLSTM') 
from biLSTM_module import biLSTM

# Load dataset
dataset = SimulationDataset(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\02_iCEM\iCEM\simulationData\target.npz')

# Load Mujoco model
filepath = r"C:\Users\cleme\Desktop\HiWi-Stellen\ISW\02_iCEM\iCEM\res\scene.xml"
model = mujoco.MjModel.from_xml_path(filepath)
data = mujoco.MjData(model)
model.opt.timestep = 0.001

# Controller setup
site_name = "attachment_site"
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
mocap_body_name = "target_EE"
q0 = np.array([0.108, 0.0716, 0.132, -2.71, -0.000547, 2.79, -0.545])

# Load biLSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = biLSTM()
nn_model.to(device)
model_path = r'models/biLSTM/models/BEST_bi_LSTM_2_rel_vel_bs128_hs256_lr00001_nl3_epochs50_10k.pth'
nn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
nn_model.eval()


# Parameters
NUM_SAMPLES = 50
horizon = 20
NUM_ELITES = 10
elites_frac = 0.3
colored_beta = 1.0
icem_optimizations = 5
reduction_factor = 1.25
momentum =0.1

dt = 0.2
control_dim = 4

# Define standard deviation for each control dimension
STD_DEV = torch.tensor([0.2, 0.2, 0.2, 0.5], device=device)
Q = torch.eye(18, device=device) * 300
S = torch.eye(control_dim, device=device) * 10  # Example value, can be adjusted
R = torch.eye(4, device=device) * 100 # 0.01

NUM_TRIALS = 2  # Number of trials
MAX_ITERATIONS = 5  # corresponds to approx. 30 seconds (with dt ~ 0.02)
print('Everything loaded successfully.')

#############################################################
######################### FUNCTIONS #########################
#############################################################

def get_state_from_mujoco(model, data, dataset, device, prev_cable_pos=None):
    # Get body IDs
    body_index = model.body('Box_first').id
    cabel_first_index = model.body('CB0').id
    cabel_last_index = model.body('cube_body_1').id  # "b_Last"

    # Get end-effector state
    ee_euler_np = data.qpos[body_index + 2:body_index + 5]  # (3,)
    ee_pos_np = data.xpos[body_index]  # (3,)
    ee_pos_combined_np = np.hstack([ee_pos_np[:2], ee_euler_np[:2]])  # (4,)
    ee_vel_np = data.qvel[body_index - 1: body_index + 3]  # (4,)

    if prev_cable_pos is not None:
        prev_cable_pos = prev_cable_pos.to(device)

    # Get DLO (cable) state
    xpos_np = data.xpos[cabel_first_index:cabel_last_index]  # (N, 3), where N is the number of cable markers
    coords_np = xpos_np[:, :2]  # (N, 2)
    coords_filtered_np = coords_np[6:-3]  # (N-9, 2)
    coords_reduced_np = coords_filtered_np[::5]    # (M, 2), e.g., (8, 2)

    # Convert all tensors to the appropriate device
    ee_pos = torch.from_numpy(ee_pos_combined_np).float().to(device)
    ee_vel = torch.from_numpy(ee_vel_np).float().to(device)
    cable_pos = torch.from_numpy(coords_reduced_np).float().to(device)
    
    # Get DLO (cable) state in 3D 
    xpos_np_3d = data.xpos[cabel_first_index:cabel_last_index]  # (N, 3), where N is the number of cable markers
    coords_np_3d = xpos_np_3d[:, :3]  # (N, 3)
    coords_filtered_np_3d = coords_np_3d[6:-3]  # (N-9, 3)
    coords_reduced_np_3d = coords_filtered_np_3d[::5]    # (M, 3), e.g., (8, 3)

    # Convert all tensors to the appropriate device

    cable_pos_3d = torch.from_numpy(coords_reduced_np_3d).float().to(device)

    # Get cable velocity
    if prev_cable_pos is not None:
        cable_vel = cable_pos - prev_cable_pos  # (9, 2)
    else:
        cable_vel = torch.zeros_like(cable_pos)  # (9, 2)

    # Create robot state
    current_robot_state_pos = ee_pos.view(-1)  # Shape: (1, 4)
    current_robot_state_vel = ee_vel.view(-1)  # Shape: (1, 4)
    current_robot_state = torch.cat((current_robot_state_pos, current_robot_state_vel), dim=0)

    # Create DLO state
    cable_pos_flat = cable_pos.view(-1)
    cable_vel_flat = cable_vel.view(-1)
    current_dlo_state = torch.cat((cable_pos_flat, cable_vel_flat), dim=0)

    # Create overall state
    state = torch.cat((current_robot_state, current_dlo_state), dim=0)
    return state, cable_pos, cable_pos_3d


def forward_model_batch(x0, u_seq, model, dataset, dt=0.2, device=device):
    """
    Propagate a batch of states through a full control horizon.
    
    Args:
        x0: Tensor of shape (num_samples, state_dim), initial states
        u_seq: Tensor of shape (num_samples, horizon, control_dim), control sequences
        nn_model: trained neural network model
        dataset: dataset object containing normalization stats
        dt: simulation timestep
        device: torch device
    
    Returns:
        x_traj: Tensor of shape (num_samples, horizon+1, state_dim)
                containing the state trajectory for each sample
    """
    num_samples, horizon, control_dim = u_seq.shape
    state_dim = x0.shape[1]

    # Initialize trajectory tensor
    x_traj = torch.zeros(num_samples, horizon+1, state_dim, device=device)
    x_traj[:, 0, :] = x0

    x_current = x0

    for t in range(horizon):
        u_current = u_seq[:, t, :]  # (num_samples, control_dim)
        
        # --- One-step forward ---
        batch_size = x_current.shape[0]

        ee_pos = x_current[:, 0:4]
        ee_vel = x_current[:, 4:8]
        feature_pos = x_current[:, 8:26].view(batch_size, 9, 2)
        feature_vel = x_current[:, 26:44].view(batch_size, 9, 2)

        # Update robot states
        ee_vel_new = torch.clamp(ee_vel + u_current, -0.2, 0.2)
        ee_pos_new = ee_pos + ee_vel_new * dt

        # Normalize states
        eps = 1e-9
        ee_pos_norm = (ee_pos_new - dataset.EE_pos_mean.to(device)) / (dataset.EE_pos_std.to(device) + eps)
        ee_vel_norm = (ee_vel_new - dataset.EE_vel_mean.to(device)) / (dataset.EE_vel_std.to(device) + eps)
        l2_norm = torch.norm(feature_pos, p=2, dim=2, keepdim=True)
        cable_pos_norm = feature_pos / (l2_norm + eps)
        cable_vel_norm = (feature_vel - dataset.cable_vel_mean_2D.to(device)) / (dataset.cable_vel_std_2D.to(device) + eps)

        # Prepare network inputs
        robot_state_pos = ee_pos_norm.unsqueeze(1).repeat(1, 9, 1)
        robot_state_vel = ee_vel_norm.unsqueeze(1).repeat(1, 9, 1)
        robot_state = torch.cat([robot_state_pos, robot_state_vel], dim=2)
        dlo_state = torch.cat([cable_pos_norm, cable_vel_norm], dim=2)
        inputs = torch.cat([robot_state, dlo_state], dim=2).view(-1, 12)

        # Predict cable velocities
        with torch.no_grad():
            norm_pred_vel = nn_model(inputs).view(batch_size, 9, 2)

        predicted_velocity = norm_pred_vel * dataset.cable_vel_std_2D.to(device) + dataset.cable_vel_mean_2D.to(device)
        predicted_feature_pos = feature_pos + predicted_velocity

        # Assemble next state
        x_next = torch.cat([
            ee_pos_new,
            ee_vel_new,
            predicted_feature_pos.view(batch_size, -1),
            predicted_velocity.view(batch_size, -1)
        ], dim=1)

        # Save and propagate
        x_traj[:, t+1, :] = x_next
        x_current = x_next

    return x_traj

def rollout_trajectory(x0, u_traj, model, dataset, dt=0.02):
    """
    Simulate full trajectory given a sequence of controls.
    
    Args:
        x0: Tensor (batch_size, state_dim)
        u_traj: Tensor (batch_size, horizon, control_dim)
    Returns:
        x_traj: Tensor (batch_size, horizon+1, state_dim)
    """
    return forward_model_batch(x0, u_traj, model, dataset, dt=dt)


def cost_function_batch(x_batch, p_target, Q):
    # x_batch: (batch_size, state_dim)
    # u_batch: (batch_size, control_dim)
    # u_prev_batch: (batch_size, control_dim) or None
    # S: Weighting matrix for control changes or None

    batch_size = x_batch.shape[0]
    p = x_batch[:, 8:26].view(batch_size, -1)  # (batch_size, 18)
    diff = p - p_target  # (batch_size, 18)

    # Shape cost: 0.5 * diff^T Q diff
    shape_cost = 0.5 * (diff @ Q * diff).sum(dim=1)  # (batch_size,)

    # Control cost: 0.5 * u^T R u
    #control_cost = 0.5 * (u_batch @ R * u_batch).sum(dim=1)  # (batch_size,)

    # New: Change cost if S is not None and u_prev_batch is provided
    # change_cost = 0
    # if S is not None and u_prev_batch is not None:
    #     delta_u = u_batch - u_prev_batch  # (batch_size, control_dim)
    #     change_cost = 0.5 * (delta_u @ S * delta_u).sum(dim=1)  # (batch_size,)

    #return shape_cost + control_cost + change_cost
    return shape_cost

def two_costs(x_batch, u_batch, p_target, Q, R):
    # x_batch: (batch_size, state_dim)
    # u_batch: (batch_size, control_dim)
    # u_prev_batch: (batch_size, control_dim) or None
    # S: Weighting matrix for control changes or None

    control_cost = torch.zeros(x_batch.shape[0], device=x_batch.device)
    batch_size = x_batch.shape[0]
    p = x_batch[:, 8:26].view(batch_size, -1)  # (batch_size, 18)
    diff = p - p_target  # (batch_size, 18)

    for t in range(x_batch.shape[1]):
        u_t = u_batch[:, t, :]
        
        # Control cost: 0.5 * u^T R u
        control_cost = 0.5 * (u_t @ R * u_t).sum(dim=1)  # (batch_size,)
        final_control_cost += control_cost
    
    # Shape cost: 0.5 * diff^T Q diff
    shape_cost = 0.5 * (diff @ Q * diff).sum(dim=1)  # (batch_size,)

    # New: Change cost if S is not None and u_prev_batch is provided
    # change_cost = 0
    # if S is not None and u_prev_batch is not None:
    #     delta_u = u_batch - u_prev_batch  # (batch_size, control_dim)
    #     change_cost = 0.5 * (delta_u @ S * delta_u).sum(dim=1)  # (batch_size,)

    #return shape_cost + control_cost + change_cost
    return shape_cost + control_cost    


def colored_samples(beta, num_samples, horizon, control_dim=4, sigma=None, mu=None, device="cpu"):
    """
    Generate colored (1/f^β) noise samples for control sequences using PyTorch.

    Args:
        beta (float): Power-law exponent (0=white, 1=pink, 2=brown)
        num_samples (int): Number of trajectories to sample
        horizon (int): Number of timesteps per trajectory
        control_dim (int): Control dimensions (e.g., x, y, z, rot)
        sigma (torch.Tensor or float, optional): Std dev for each timestep and control dim (H, C) or scalar
        mu (torch.Tensor or float, optional): Mean for each timestep and control dim (H, C) or scalar
        device (str or torch.device): Target device

    Returns:
        torch.Tensor: (num_samples, horizon, control_dim)
    """
    # White Gaussian noise in frequency domain
    white = torch.randn(num_samples, control_dim, horizon, device=device)

    # Frequency scaling for power-law
    freqs = torch.fft.rfftfreq(horizon, d=1.0).to(device)
    freqs[0] = 1e-6  # avoid divide by zero
    amplitude = 1.0 / (freqs ** (beta / 2.0))
    amplitude = amplitude.view(1, 1, -1)  # broadcast

    # FFT → filter → IFFT
    white_fft = torch.fft.rfft(white, dim=2)
    colored_fft = white_fft * amplitude
    colored = torch.fft.irfft(colored_fft, n=horizon, dim=2)
    colored = colored / colored.std(dim=(1,2), keepdim=True)  # normalize


    # Permute to (num_samples, horizon, control_dim)
    colored = colored.permute(0, 2, 1)

    # Apply sigma (time-dependent)
    if sigma is not None:
        sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        if sigma.ndim == 0:  # scalar
            sigma = sigma.view(1, 1, 1)
        elif sigma.ndim == 1:  # per control dim
            sigma = sigma.view(1, 1, control_dim)
        elif sigma.ndim == 2:  # per timestep
            sigma = sigma.unsqueeze(0)  # (1, H, C)
        colored = colored * sigma

    # Apply mu (time-dependent)
    if mu is not None:
        mu = torch.tensor(mu, dtype=torch.float32, device=device)
        if mu.ndim == 0:
            mu = mu.view(1, 1, 1)
        elif mu.ndim == 1:
            mu = mu.view(1, 1, control_dim)
        elif mu.ndim == 2:
            mu = mu.unsqueeze(0)  # (1, H, C)
        colored = colored + mu
    
    return colored



def icem_optimize(x_current, p_target, Q, R, elites, elites_prev, horizon=20, control_dim=4, num_iterations=10,
                  N_init=200, K=20, y=1.25, sigma_init=1.0, min_sigma=0.05, 
                  beta_sigma=0.8, elites_frac=0.3):
    """
    Simplified iCEM mockup returning only the first action of the best control sequence.
    """
    # --- Initialization ---
    mu = torch.zeros(horizon, control_dim, device=device)
    sigma = torch.ones(horizon, control_dim, device=device) * sigma_init

    # --- Optimization iterations ---
    for it in range(num_iterations):

        print("icem-Iteration: ", it+1)
    
        # --- Adaptive number of samples ---
        if it == 0: # first iteration  
            if elites_prev is not None: 
                N_elites = int(K*elites_frac) 
            else:
                N_elites = 0
            N_i = max(2*K, N_init - N_elites)
        else:
            N_elites = int(K * elites_frac)
            N_i = int(max(N_init * y ** (-it), 2*K) - N_elites)

        # --- Sample controls ---
        u_samples = colored_samples(
            beta=1,
            num_samples=N_i,
            horizon = horizon,
            control_dim=control_dim,
            mu=mu,
            sigma=sigma,
        )

        # Add previous elites if available
        if it == 0 and elites_prev is not None:
            u_samples = torch.cat([u_samples, elites_prev[:N_elites]], dim=0)  # concat along sample dimension
            print("Using previous elites.")
        elif it == 0 and elites_prev is None:
            print("No previous elites available.")
            pass
        else:
            u_samples = torch.cat([u_samples, elites[:N_elites]], dim=0)  # concat along sample dimension
            print("Using current elites.")
            
        
            
        # Add mean action for stability at last step
        if it == num_iterations - 1:
            mean_sample = u_samples.mean(dim=0, keepdim=True)  # shape (1, horizon, control_dim)
            u_samples = torch.cat([u_samples, mean_sample], dim=0)  # concat along sample dim

            # Simulate trajectories for the mean sample
            x_batch = rollout_trajectory(x_current.repeat(u_samples.shape[0], 1), u_samples, model, dataset, dt=dt)
            costs = cost_function_batch(x_batch[:, -1, :], p_target.flatten(), Q)
            # costs = two_costs(x_batch, u_samples, p_target, Q, R)
            
            # Select best overall trajectory
            best_cost_val, u_best_id = torch.min(costs, dim=0)
            u_best = u_samples[u_best_id:u_best_id + 1]  # shape (1, horizon, control_dim)
            
            # Return only the first action
            u_first = u_best[:, 0, :]  # shape (1, control_dim)
            
        else:
            x_batch = rollout_trajectory(x_current.repeat(u_samples.shape[0], 1), u_samples, model, dataset, dt=dt)
            costs = cost_function_batch(x_batch[:, -1, :], p_target.flatten(), Q)
            # costs = two_costs(x_batch, u_samples, p_target, Q, R)

        print("x_batch:", x_batch.shape)
        print("u_batch:", u_samples.shape)
        print("p_target:", p_target.shape)
        # --- Select elites ---
        elite_ids = torch.topk(costs, K, largest=False).indices        
        elites = u_samples[elite_ids]
        
        # if elites is None:
        #     print("Elites is None!")
        # else:
        #     print("Elites shape:", elites.shape)

        # --- Update distribution ---
        mu = torch.mean(elites, dim=0)
        sigma = beta_sigma * sigma + (1 - beta_sigma) * torch.std(elites, dim=0)
        
    final_cost = costs.min().item() 
    final_mu = mu.mean().item()
    final_sigma = sigma.mean().item()
       
    return u_first, elites, final_cost, final_mu, final_sigma

controller = InverseKinematicsController(
    model=model,
    data=data,
    site_name=site_name,
    joint_names=joint_names,
    mocap_body_name=mocap_body_name,
    q0=q0,
    integration_dt=0.3,
    damping=1e-4,
    Kpos=0.95,
    Kori=0.95,
    dt=0.002,
    max_angvel=0.3
)
x_current, cable_pos_current, cable_pos_current_3d = get_state_from_mujoco(model, data, dataset, device)
#random_idx = random.randint(0, len(dataset) - 1)
random_idx = np.random.choice([1940])
inputs_dumm, p_target = dataset[random_idx]
p_target = dataset.get_absolute_coords(random_idx)
p_target = torch.tensor(p_target).view(9, 2)
error_list = []

# Lists for storing results
trial_results = []  # Each element: (converged (bool), iterations to convergence)

# Start the viewer in "passive" mode
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)
    data.qpos[controller.dof_ids] = q0
    data.mocap_pos[controller.mocap_id] = np.array([0.40, 0.002, 0.005])
    data.ctrl[8] = -10
    data.ctrl[7] = -10
    # Create dictionaries for target and prediction bodies (as in your original code)
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

    # Outer loop: trials
    for trial in range(NUM_TRIALS):
        print(f"Starting trial {trial+1}/{NUM_TRIALS}")

        # --- Reset Simulation ---
        # Reset MuJoCo internal state
        mujoco.mj_resetData(model, data)
        data.qpos[controller.dof_ids] = q0
        data.mocap_pos[controller.mocap_id] = np.array([0.4, 0.002, 0.004])
        data.ctrl[8] = -20
        data.ctrl[7] = -20
        mujoco.mj_forward(model, data)

        # Optionally reset the state of your cable here,
        # e.g. cable_pos_current = initial_cable_pos
        cable_pos_current = None  # or initial_cable_pos if defined
        
        # Reset iCEM variables
        elites = None
        elites_prev = None

        # Set target positions (if they need to be reset for each trial)
        for i, pt in enumerate(p_target, start=1):
            x, y = pt
            z = 0.01
            mocap_id = target_mocap_ids[f"target_body_{i}"]
            data.mocap_pos[mocap_id] = np.array([x, y, z])

        # Variables for monitoring the trial
        converged = False
        steps_to_convergence = MAX_ITERATIONS  # If not converged, remains at MAX_ITERATIONS
        start_time = time.time()

        # Optionally, lists to collect additional data during each trial (e.g., errors, costs, EE positions)
        error_list = []
        ee_positions = []
        all_costs_per_step = []
        all_mu = []
        all_sigma = []

        # Inner loop: up to MAX_ITERATIONS iterations
        for step in range(MAX_ITERATIONS):
            # Retrieve state (adjust according to your system)
            x_current, cable_pos_current = get_state_from_mujoco(model, data, dataset, device, cable_pos_current)
            ee_pos_current = x_current[:2].cpu().numpy()  # e.g., end-effector position (x,y)
            ee_positions.append(ee_pos_current)
            print(f"Step {step+1}/{MAX_ITERATIONS}")

            
            # Perform MPPI optimization
            U_opt, elites_prev, final_cost, final_mu, final_sigma = icem_optimize(x_current, 
                p_target, 
                Q,R,
                elites=elites,
                elites_prev=elites_prev,
                horizon=horizon, 
                control_dim=control_dim, 
                num_iterations=icem_optimizations,
                N_init=NUM_SAMPLES, 
                K=NUM_ELITES, 
                y=reduction_factor, 
                sigma_init=1.0, 
                min_sigma=0.05, 
                beta_sigma=0.8, 
                elites_frac=elites_frac)

            # Smooth the control sequence (optional)
            # U_opt_smoothed = smooth_control_sequence(U_opt, window_size=1)
            # u_apply = U_opt_smoothed[0]
            
            u_apply = U_opt[0]

            x = np.clip(u_apply[0], -0.2, 0.2)
            y = np.clip(u_apply[1], 0, 0.35)
            rot_z = np.clip(u_apply[3], -1, 1) * 0.5  # Ensure rot_z is a valid value

            data.mocap_pos[controller.mocap_id] = np.array([x + 0.4, y, 0.004])
            controller.get_ctrl()
            # data.ctrl[6] = rot_z
            p_current_now = x_current[8:26].reshape(-1, 2).cpu().numpy()
            p_target_cpu = p_target.cpu().numpy()
            error = np.linalg.norm(p_current_now - p_target_cpu, axis=1)
            error_list.append(error)
            all_costs_per_step.append(final_cost)
            all_mu.append(final_mu)
            all_sigma.append(final_sigma)
            
            # Simulate 20 timesteps with the current control command
            for _ in range(20):
                controller.get_ctrl()
                # data.ctrl[6] = rot_z
                mujoco.mj_step(model, data)
                viewer.sync()

            if np.all(np.abs(error) < 0.03):
                elapsed_time = time.time() - start_time
                print(f"Trial {trial+1}: Converged after {step+1} iterations (Time: {elapsed_time:.4f}s)")
                converged = True
                steps_to_convergence = step + 1  # or (step+1)*20 for actual number of timesteps
                break
            

        trial_results.append((converged, steps_to_convergence))

    viewer.close()
    
        
plt.figure()
plt.plot(all_costs_per_step, marker='o')
plt.xlabel("Control step")
plt.ylabel("Best cost")
plt.title("iCEM convergence over control steps")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(error_list, marker='o')
plt.xlabel("Control step")
plt.ylabel("Error")
plt.title("iCEM error over control steps")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(all_mu, marker='o')
plt.xlabel("Control step")
plt.ylabel("Error")
plt.title("iCEM mean over control steps")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(all_sigma, marker='o')
plt.xlabel("Control step")
plt.ylabel("Error")
plt.title("iCEM std over control steps")
plt.grid(True)
plt.show()

# Summary of results after all trials:
num_converged = sum(1 for result in trial_results if result[0])
print(f"Out of {NUM_TRIALS} trials, {num_converged} converged.")

# Optionally, save or further process the results
# For example, write to a CSV file:
# import csv
# with open("trial_results.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Trial", "Converged", "Iterations_to_Convergence"])
#     for i, (conv, steps) in enumerate(trial_results, start=1):
#         writer.writerow([i, conv, steps])

# If all_costs_per_step is a list of lists, convert it to a 2D array
all_costs_per_step_array = np.array(all_costs_per_step)

# Compute mean and standard deviation over the samples
mean_costs = np.mean(all_costs_per_step_array, axis=1)
std_costs = np.std(all_costs_per_step_array, axis=1)

# X-axis: steps
steps = np.arange(len(mean_costs))

# Euclidean error
euclidean_error = np.array(error_list)
total_error = np.mean(euclidean_error, axis=1)
