import numpy as np
import torch
import mujoco
from mujoco.glfw import glfw
import random
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt
from getDataset import SimulationDataset
import time
from scipy.stats import qmc, norm
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
import imageio
import matplotlib.animation as animation
import sys
sys.path.append("./control")
from IK_NS import InverseKinematicsController
sys.path.append("./biLSTM")
from biLSTM_module import MultiMarkerLSTM

dataset = SimulationDataset('data/simulation_data_rollout_norot.npz')

# Load model
filepath = "res/scene.xml"
model = mujoco.MjModel.from_xml_path(filepath)
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()                        # Visualization options

# Set additional model settings if needed, e.g. gravity compensation
model.opt.timestep = 0.001
simend = 0.1  # simulation time
print_camera_config = 1  # set to 1 to print camera config (useful for initializing the view)

# Controller configuration parameters
site_name = "attachment_site"           # Name of the end-effector site
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
mocap_body_name = "target_EE"           # Name of the mocap body
q0 = np.array([0.108, 0.0716, 0.132, -2.71, -0.000547, 2.79, -0.545])  # Initial joint position

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
nn_model = MultiMarkerLSTM(hidden_size=256, num_layers=3)
nn_model.to(device)
nn_model.load_state_dict(torch.load("models/BEST_bi_LSTM_2_rel_vel_bs128_hs256_lr00001_nl3_epochs50_10k.pth", weights_only=True, map_location=torch.device('mps')))
nn_model.eval()

horizon = 12  # MPPI planning horizon
dt = 0.02
control_dim = 4
NUM_SAMPLES = 100  # Number of samples
TEMPERATURE = 0.001
STD_DEV = torch.tensor([0.3, 0.75, 0.0, 3], device=device)  # Standard deviation for each control dimension
Q = torch.eye(18, device=device) * 300
S = torch.eye(control_dim, device=device) * 1
R = torch.eye(4, device=device) * 0.01

def smooth_control_sequence(U_opt, window_size=3):
    """
    Smooth the control sequences using a moving average.
    Args:
        U_opt: Array of control sequences (horizon, control_dim)
        window_size: Number of steps for the moving average
    Returns:
        Smoothed control sequences
    """
    smoothed_U = np.convolve(U_opt[:, 0], np.ones(window_size)/window_size, mode='valid')
    return np.array([np.convolve(U_opt[:, i], np.ones(window_size)/window_size, mode='valid')
                     for i in range(U_opt.shape[1])]).T


def get_state_from_mujoco(model, data, dataset, device, prev_cable_pos=None):
    # Get body IDs
    body_index = model.body('Box_first').id
    cabel_first_index = model.body('CB0').id
    cabel_last_index = model.body('cube_body_1').id  # "b_Last"

    # Get EE state
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
    coords_reduced_np = coords_filtered_np[::5]    # (M, 2), for example (8, 2)

    # Convert all tensors to the correct device
    ee_pos = torch.from_numpy(ee_pos_combined_np).float().to(device)
    ee_vel = torch.from_numpy(ee_vel_np).float().to(device)
    cable_pos = torch.from_numpy(coords_reduced_np).float().to(device)

    # Get cable velocity
    if prev_cable_pos is not None:
        cable_vel = cable_pos - prev_cable_pos  # (9, 2)
    else:
        cable_vel = torch.zeros_like(cable_pos)  # (9, 2)

    # Create robot state
    current_robot_state_pos = ee_pos.view(-1)  # Shape: (1, 4)
    current_robot_state_vel = ee_vel.view(-1)  # Shape: (1, 4)
    current_robot_state = torch.cat((current_robot_state_pos, current_robot_state_vel), dim=0)

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
        model: The MuJoCo model
        pred_mocap_ids: Dictionary of predicted mocap IDs (not used in this code, but kept for compatibility)
        data: The MuJoCo data structure
        dt: Time step for integration (default: 0.02)

    Returns:
        x_next_batch: Tensor of shape (batch_size, state_dim) representing the next state
    """
    # 1. Extract states for the entire batch
    batch_size = x_batch.shape[0]

    # Robot states
    ee_pos = x_batch[:, 0:4]          # (batch_size, 4)
    ee_vel = x_batch[:, 4:8]          # (batch_size, 4)

    # Cable states
    feature_pos = x_batch[:, 8:26].view(batch_size, 9, 2)   # (batch_size, 9, 2)
    feature_vel = x_batch[:, 26:44].view(batch_size, 9, 2)    # (batch_size, 9, 2)

    # 2. Compute new robot states for the entire batch
    ee_vel_new = ee_vel + u_batch  # (batch_size, 4)
    v_max = 0.2
    ee_vel_new = torch.clamp(ee_vel_new, -v_max, v_max)
    ee_pos_new = ee_pos + ee_vel_new * dt  # (batch_size, 4)

    # 3. Normalization for the entire batch
    eps = 1e-9
    l2_norm_cable_pos = torch.norm(feature_pos, p=2, dim=2, keepdim=True)  # (batch_size, 9, 1)

    # Normalize robot state
    ee_pos_normalized = (ee_pos_new - dataset.EE_pos_mean.to(device)) / (dataset.EE_pos_std.to(device) + eps)
    ee_vel_normalized = (ee_vel_new - dataset.EE_vel_mean.to(device)) / (dataset.EE_vel_std.to(device) + eps)

    # Normalize cable state
    cable_pos_normalized = feature_pos / (l2_norm_cable_pos + eps)  # (batch_size, 9, 2)
    cable_vel_normalized = (feature_vel - dataset.cable_vel_mean_2D.to(device)) / (dataset.cable_vel_std_2D.to(device) + eps)

    # 4. Prepare network inputs
    # Repeat robot states for each marker
    robot_state_pos = ee_pos_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
    robot_state_vel = ee_vel_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
    robot_state = torch.cat((robot_state_pos, robot_state_vel), dim=2)  # (batch_size, 9, 8)

    # Combine with cable states
    dlo_state = torch.cat((cable_pos_normalized, cable_vel_normalized), dim=2)  # (batch_size, 9, 4)
    inputs = torch.cat((robot_state, dlo_state), dim=2)  # (batch_size, 9, 12)

    # Reshape for the network: (batch_size * 9, 12)
    inputs = inputs.view(-1, 12)

    # 5. Batch inference with the neural network
    with torch.no_grad():
        norm_predicted_velocity = nn_model(inputs)  # (batch_size * 9, 2)

    # Reshape back: (batch_size, 9, 2)
    norm_predicted_velocity = norm_predicted_velocity.view(batch_size, 9, 2)

    # 6. Denormalize the prediction
    cable_vel_std_2D = dataset.cable_vel_std_2D.to(device)
    cable_vel_mean_2D = dataset.cable_vel_mean_2D.to(device)
    predicted_velocity = norm_predicted_velocity * cable_vel_std_2D + cable_vel_mean_2D

    # 7. Update state using the predicted velocity
    predicted_feature_pos = feature_pos + predicted_velocity  # (batch_size, 9, 2)

    # 8. Assemble the next state
    x_next_batch = torch.cat([
        ee_pos_new,
        ee_vel_new,
        predicted_feature_pos.view(batch_size, -1),
        predicted_velocity.view(batch_size, -1)
    ], dim=1)  # (batch_size, 44)

    return x_next_batch


def cost_function_batch(x_batch, u_batch, p_target, Q, R, u_prev_batch=None, S=None):
    # x_batch: (batch_size, state_dim)
    # u_batch: (batch_size, control_dim)
    # u_prev_batch: (batch_size, control_dim) oder None
    # S: Gewichtungsmatrix für Steuerungsänderungen oder None

    batch_size = x_batch.shape[0]
    p = x_batch[:, 8:26].view(batch_size, -1)  # (batch_size, 18)
    diff = p - p_target  # (batch_size, 18)

    # Shape cost: 0.5 * diff^T Q diff
    shape_cost = 0.5 * (diff @ Q * diff).sum(dim=1)  # (batch_size,)

    # Control cost: 0.5 * u^T R u
    control_cost = 0.5 * (u_batch @ R * u_batch).sum(dim=1)  # (batch_size,)

    # NEU: Änderungskosten, falls S nicht None ist und es einen u_prev_batch gibt
    change_cost = 0
    if S is not None and u_prev_batch is not None:
        delta_u = u_batch - u_prev_batch  # (batch_size, control_dim)
        # 0.5 * (delta_u^T S delta_u)
        change_cost = 0.5 * (delta_u @ S * delta_u).sum(dim=1)  # (batch_size,)

    return shape_cost + control_cost + change_cost

@staticmethod
def halton_noise(num_samples, horizon, control_dim, std_dev, device):

    d = horizon * control_dim

    sampler = qmc.Halton(d=d, scramble=True)
    samples = sampler.random(n=num_samples)
    samples = samples.reshape(num_samples, horizon, control_dim)
    samples_normal = norm.ppf(samples)
    samples_tensor = torch.tensor(samples_normal, dtype=torch.float32, device=device)
    samples_tensor = samples_tensor * std_dev

    return samples_tensor

def mppi_optimize(x0, p_target, u_init=None, u_prev=None):
    global Q, R, S

    if u_init is None:
        u_init = torch.zeros((horizon, control_dim), device=device)
    else:
        u_init = torch.roll(u_init, -1, dims=0)
        u_init[-1] = 0

    all_costs = []

    p_target_tensor = p_target.to(device).flatten()

    for mppi_iter in range(1):

        noise = halton_noise(NUM_SAMPLES, horizon, control_dim, STD_DEV, device)
        samples = u_init.unsqueeze(0) + noise  # (NUM_SAMPLES, horizon, control_dim)

        # 2. Batch-Simulation
        x_batch = x0.clone().repeat(NUM_SAMPLES, 1)  # (NUM_SAMPLES, state_dim)
        total_costs = torch.zeros(NUM_SAMPLES, device=device)
        u_prev_batch = u_prev.clone().repeat(NUM_SAMPLES, 1) if u_prev is not None else None

        for t in range(horizon):

            u_t = samples[:, t]  # (NUM_SAMPLES, control_dim)
            x_batch = forward_model_batch(x_batch, u_t, model, pred_mocap_ids, data, dt=0.02)
            total_costs += cost_function_batch(x_batch, u_t, p_target_tensor, Q, R, u_prev_batch, S)

            u_prev_batch = u_t.clone()

        min_cost = torch.min(total_costs)
        weights = torch.exp(-(total_costs - min_cost) / TEMPERATURE)
        weights /= torch.sum(weights)

        weights = weights.view(-1, 1, 1)  # (NUM_SAMPLES, 1, 1)
        u_init = torch.sum(weights * samples, dim=0)  # (horizon, control_dim)

        all_costs.append(torch.mean(total_costs).item())

    return u_init.cpu().numpy(), all_costs


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
        max_angvel=0.2
    )
x_current, cable_pos_current = get_state_from_mujoco(model, data, dataset, device)
random_idx = random.randint(0, len(dataset) - 1)
inputs_dumm, p_target = dataset[random_idx]
p_target = dataset.get_absolute_coords(2180)
p_target = torch.tensor(p_target).view(9, 2)
u_init = None
error_list = []

NUM_TRIALS = 1
MAX_ITERATIONS = 750

trial_results = []

glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

cam.azimuth = 180
cam.elevation = -90
cam.distance = 1
cam.lookat = np.array([0.4, 0.25, 0])

output_path='rendered_video.mp4'
writer = imageio.get_writer(output_path, fps=60)

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0/60.0):
        mujoco.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    mujoco.mj_resetData(model, data)
    data.qpos[controller.dof_ids] = q0
    data.mocap_pos[controller.mocap_id] = np.array([0.40, 0.002, 0.005])
    data.ctrl[8] = -10
    data.ctrl[7] = -10
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

    for trial in range(NUM_TRIALS):
        print(f"Starte Versuch {trial+1}/{NUM_TRIALS}")

        mujoco.mj_resetData(model, data)
        data.qpos[controller.dof_ids] = q0
        data.mocap_pos[controller.mocap_id] = np.array([0.4, 0.002, 0.004])
        data.ctrl[8] = -20
        data.ctrl[7] = -20
        mujoco.mj_forward(model, data)
        cable_pos_current = None

        for i, pt in enumerate(p_target, start=1):
            x, y = pt
            z = 0.01
            mocap_id = target_mocap_ids[f"target_body_{i}"]
            data.mocap_pos[mocap_id] = np.array([x, y, z])

        converged = False
        steps_to_convergence = MAX_ITERATIONS
        start_time = time.time()

        error_list = []
        ee_positions = []
        all_costs_per_step = []

        for step in range(MAX_ITERATIONS):
            x_current, cable_pos_current = get_state_from_mujoco(model, data, dataset, device, cable_pos_current)
            ee_pos_current = x_current[:2].cpu().numpy()
            ee_positions.append(ee_pos_current)

            U_opt, costs = mppi_optimize(x_current, p_target)
            all_costs_per_step.append(costs)

            U_opt_smoothed = smooth_control_sequence(U_opt, window_size=1)
            u_apply = U_opt_smoothed[0]

            x = u_apply[0]
            x = np.clip(u_apply[0], -0.2, 0.2)
            y = u_apply[1]
            y = np.clip(u_apply[1], 0, 0.35)
            x = u_apply[0]
            rot_z = u_apply[3]
            rot_z = np.clip(u_apply[3], -1, 1) * 0.5


            data.mocap_pos[controller.mocap_id] = np.array([x + 0.4, y, 0.004])
            controller.get_ctrl()
            data.ctrl[6] = rot_z
            p_current_now = x_current[8:26].reshape(-1, 2).cpu().numpy()
            p_target_cpu = p_target.cpu().numpy()
            error = np.linalg.norm(p_current_now - p_target_cpu, axis=1)
            error_list.append(error)
            for _ in range(20):
                controller.get_ctrl()
                data.ctrl[6] = rot_z
                mujoco.mj_step(model, data)
                # Update scene and render
                mujoco.mjv_updateScene(model, data, opt, None, cam,
                                mujoco.mjtCatBit.mjCAT_ALL.value, scene)
                mujoco.mjr_render(viewport, scene, context)
                height, width = 912, 1200
                # Read pixels from the framebuffer
                buffer = np.zeros((height, width, 3), dtype=np.uint8)
                glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer)
                # Flip the image vertically (OpenGL uses bottom-left origin)
                buffer = np.flip(buffer, axis=0)
                # Append the frame to the video writer
                writer.append_data(buffer)

                # swap OpenGL buffers (blocking call due to v-sync)
                glfw.swap_buffers(window)

                # process pending GUI events, call GLFW callbacks
                glfw.poll_events()


            if np.all(np.abs(error) < 0.03):
                elapsed_time = time.time() - start_time
                print(f"Experiment {trial+1}: converges after {step+1} Iterations (Time: {elapsed_time:.4f}s)")
                converged = True
                steps_to_convergence = step + 1
                break

    writer.close()
    glfw.terminate()

num_converged = sum(1 for (conv, _) in trial_results if conv)
print(f"From {NUM_TRIALS} Trials {num_converged} converged.")

import csv
with open("trial_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Trial", "Converged", "Iterations_to_Convergence"])
    for i, (conv, steps) in enumerate(trial_results, start=1):
        writer.writerow([i, conv, steps])


all_costs_per_step_array = np.array(all_costs_per_step)

mean_costs = np.mean(all_costs_per_step_array, axis=1)
std_costs = np.std(all_costs_per_step_array, axis=1)


# Euklidian Error
euklidian_error = np.array(error_list)
total_error = np.mean(euklidian_error, axis=1)
N = len(error_list)
Upper = N / 3
print("Länge Error List", N)
steps = np.linspace(0, Upper, N)

total_error = np.mean(euklidian_error, axis=1) * 100


fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], label="Fehler", color="b")
ax.set_xlim(0, steps[-1])
ax.set_ylim(0, 15)
ax.set_xlabel("Zeit in [s]")
ax.set_ylabel("Euklidischer Abstand in cm")
ax.set_title("Euklidischer Abstandsfehler über die Zeit")
ax.legend()
ax.grid()

def update(frame):
    total_error[frame] = total_error[frame]
    line.set_data(steps[:frame+1], total_error[:frame+1])

    return line,

# Animation
ani = animation.FuncAnimation(fig, update, frames=N, blit=True, interval=16.7)

# Save Animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save('error_animation.mp4', writer=writer)


