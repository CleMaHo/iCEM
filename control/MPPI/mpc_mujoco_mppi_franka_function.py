import numpy as np
import torch
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('Agg')
import time
from scipy.stats import qmc, norm
import sys
from scipy.signal import butter, filtfilt
import random




sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\utils') 
from IK_NS import InverseKinematicsController
from getDataset import SimulationDataset
from logging_utils import log_results_and_parameters
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\models\biLSTM') 
from biLSTM_module import biLSTM
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\control\MPPI')
# sys.path.append("./models/MLP")
# from MLP import MLPModel

# Load dataset
dataset = SimulationDataset(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\simulationData\target.npz')

# Load Mujoco model
filepath = r"C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\res\scene.xml"
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
model_path = r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\models\biLSTM\models\BEST_bi_LSTM_2_rel_vel_bs128_hs256_lr00001_nl3_epochs50_10k.pth'
nn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
nn_model.eval()

# ### define MPPI parameters
# # MPPI parameters
# horizon = 5 # Horizon for MPPI
# dt = 0.2 
# control_dim = 4
# NUM_SAMPLES = 20  # Number of samples
# TEMPERATURE = 0.002  # Dynamic temperature adjustment

# # Define standard deviation for each control dimension
# std_x = 0.5
# std_y = 0.5
# std_z = 0.05
# std_d = 3

# # Define the cost matrices
# Q1 = 18
# Q2 = 150
# S1 = 1
# R1 = 4
# R2 = 5


def mpc_franka_lp(horizon, dt, control_dim, NUM_SAMPLES, TEMPERATURE,
               std_x, std_y, std_z, std_d, Q1, Q2, S1, R1, R2,
               cutoff, fs, filter_order, device): 
           
    Q = torch.eye(Q1, device=device) * Q2
    S = torch.eye(control_dim, device=device) * S1  # Example value, can be adjusted
    R = torch.eye(R1, device=device) * R2 # 0.01
    STD_DEV = torch.tensor([std_x, std_y, std_z, std_d], device=device)

    # Define the number of trials and maximum iterations per trial
    NUM_TRIALS = 100  # Number of trials
    MAX_ITERATIONS = 500  # corresponds to approx. 30 seconds (with dt ~ 0.02)
    
    def smooth_control_sequence(U_opt, window_size=3):
        """
        Smooths the control sequences using a moving average.
        Args:
            U_opt: Array with control sequences (horizon, control_dim)
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
        return state, cable_pos

    def forward_model_batch(x_batch, u_batch, model, pred_mocap_ids, data, dt=0.02):
        """
        Batch version of the forward_model function.
        Processes a batch of states and control inputs simultaneously.
        Args:
            x_batch: Tensor of shape (batch_size, state_dim)
            u_batch: Tensor of shape (batch_size, control_dim)
        Returns:
            x_next_batch: Tensor of shape (batch_size, state_dim)
        """
        # 1. Extract states for the entire batch
        batch_size = x_batch.shape[0]

        # Extract robot states
        ee_pos = x_batch[:, 0:4]          # (batch_size, 4)
        ee_vel = x_batch[:, 4:8]          # (batch_size, 4)

        # Extract cable states
        feature_pos = x_batch[:, 8:26].view(batch_size, 9, 2)   # (batch_size, 9, 2)
        feature_vel = x_batch[:, 26:44].view(batch_size, 9, 2)  # (batch_size, 9, 2)

        # 2. Compute new robot states for the whole batch
        ee_vel_new = ee_vel + u_batch  # (batch_size, 4)
        v_max = 0.2
        ee_vel_new = torch.clamp(ee_vel_new, -v_max, v_max)
        ee_pos_new = ee_pos + ee_vel_new * dt  # (batch_size, 4)

        # 3. Normalization for the entire batch
        eps = 1e-9
        l2_norm_cable_pos = torch.norm(feature_pos, p=2, dim=2, keepdim=True)  # (batch_size, 9, 1)

        # Normalize robot states
        ee_pos_normalized = (ee_pos_new - dataset.EE_pos_mean.to(device)) / (dataset.EE_pos_std.to(device) + eps)
        ee_vel_normalized = (ee_vel_new - dataset.EE_vel_mean.to(device)) / (dataset.EE_vel_std.to(device) + eps)

        # Normalize cable states
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

        # 7. Update cable positions with predicted velocity
        predicted_feature_pos = feature_pos + predicted_velocity  # (batch_size, 9, 2)
        # 8. Assemble the next state
        x_next_batch = torch.cat([
            ee_pos_new,
            ee_vel_new,
            predicted_feature_pos.view(batch_size, -1),
            predicted_velocity.view(batch_size, -1)
        ], dim=1)  # (batch_size, 44)

        # Print elapsed time for the batched forward pass
        return x_next_batch

    def cost_function_batch(x_batch, u_batch, p_target, Q, R, u_prev_batch=None, S=None):
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
        control_cost = 0.5 * (u_batch @ R * u_batch).sum(dim=1)  # (batch_size,)

        # New: Change cost if S is not None and u_prev_batch is provided
        change_cost = 0
        if S is not None and u_prev_batch is not None:
            delta_u = u_batch - u_prev_batch  # (batch_size, control_dim)
            change_cost = 0.5 * (delta_u @ S * delta_u).sum(dim=1)  # (batch_size,)

        return shape_cost + control_cost + change_cost
    
    def butter_filter(data, cutoff, fs, filter_order):
        """
        Applies a Butterworth low-pass filter along the time axis.
        Args:
            data (np.ndarray): Input of shape (num_samples, horizon, control_dim)
            cutoff (float): Cutoff frequency [Hz]
            fs (float): Sampling rate [Hz]
            order (int): Filter order
        Returns:
            np.ndarray: Filtered data of same shape
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=1)

    @staticmethod
    def halton_noise(num_samples, horizon, control_dim, std_dev, device, cutoff, fs, filter_order):
        """
        Generates Halton-based samples and transforms them into normally distributed samples.
        Args:
            num_samples (int): Number of samples (batch size).
            horizon (int): Planning horizon (number of timesteps).
            control_dim (int): Dimension of the control.
            std_dev (torch.Tensor or numpy.array): Standard deviation per control dimension.
            device (torch.device): Device on which the tensor should reside.
        Returns:
            torch.Tensor: A tensor of shape (num_samples, horizon, control_dim) with normally distributed noise.
        """
        # Total dimensions per sample is horizon * control_dim
        d = horizon * control_dim

        # Create the Halton sampler for d dimensions; scramble=True yields better uniformity
        sampler = qmc.Halton(d=d, scramble=True)

        # Generate num_samples samples in the range [0, 1]^d
        samples = sampler.random(n=num_samples)  # Shape: (num_samples, d)

        # Reshape into (num_samples, horizon, control_dim)
        samples = samples.reshape(num_samples, horizon, control_dim)

        # Transform the uniform samples into normally distributed samples (mean 0, variance 1)
        # using the inverse CDF of the normal distribution.
        samples_normal = norm.ppf(samples)
        
        # Clip to avoid extreme values
        if cutoff is not None and fs is not None:
            samples_normal = butter_filter(samples_normal, cutoff, fs, filter_order)
            
        # Ensure the samples are contiguous in memory    
        samples_normal = np.ascontiguousarray(samples_normal)

        # Convert to a Torch tensor
        samples_tensor = torch.tensor(samples_normal, dtype=torch.float32, device=device)

        # If std_dev is a tensor of shape (control_dim,), it will be broadcast automatically.
        samples_tensor = samples_tensor * std_dev

        return samples_tensor

    def mppi_optimize(x0, p_target, Q, R, S, u_init=None, u_prev=None):
        # global Q, R, S  # Use the adjusted Q, R, and S

        # Initialize control as tensor
        if u_init is None:
            u_init = torch.zeros((horizon, control_dim), device=device)
        else:
            u_init = torch.roll(u_init, -1, dims=0)
            u_init[-1] = 0

        all_costs = []

        # Convert p_target to tensor and move it to device
        p_target_tensor = p_target.to(device).flatten()

        for mppi_iter in range(1):
            # 1. Generate samples as a batch tensor with separate STD_DEV for each control dimension
            noise = halton_noise(NUM_SAMPLES, horizon, control_dim, STD_DEV, device, cutoff, fs, filter_order)
            samples = u_init.unsqueeze(0) + noise  # (NUM_SAMPLES, horizon, control_dim)

            # 2. Batch simulation
            x_batch = x0.clone().repeat(NUM_SAMPLES, 1)  # (NUM_SAMPLES, state_dim)
            total_costs = torch.zeros(NUM_SAMPLES, device=device)
            u_prev_batch = u_prev.clone().repeat(NUM_SAMPLES, 1) if u_prev is not None else None

            for t in range(horizon):
                # Get control signals for this timestep
                u_t = samples[:, t]  # (NUM_SAMPLES, control_dim)

                start_time = time.time()
                # Perform batch update
                x_batch = forward_model_batch(x_batch, u_t, model, pred_mocap_ids, data, dt=0.02)
                end_time = time.time()
                elapsed_time = end_time - start_time
                # print(f"Batched forward pass took {elapsed_time} seconds.")
                # Accumulate costs
                total_costs += cost_function_batch(x_batch, u_t, p_target_tensor, Q, R, u_prev_batch, S)

                # Update previous control command
                u_prev_batch = u_t.clone()

            # 3. Compute weights
            min_cost = torch.min(total_costs)
            weights = torch.exp(-(total_costs - min_cost) / TEMPERATURE)
            weights /= torch.sum(weights)  # Normalize

            # 4. Update the control sequence using tensor operations
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
        max_angvel=0.3
    )
    x_current, cable_pos_current = get_state_from_mujoco(model, data, dataset, device)
    #random_idx = random.randint(0, len(dataset) - 1)
    random_idx = np.random.choice([1940])
    inputs_dumm, p_target = dataset[random_idx]
    p_target = dataset.get_absolute_coords(random_idx)
    p_target = torch.tensor(p_target).view(9, 2)
    u_init = None
    error_list = []

    # Lists for storing results
    trial_results = []  # Each element: (converged (bool), iterations to convergence)


    # Start the simulation loop without viewer
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
        mujoco.mj_resetData(model, data)
        data.qpos[controller.dof_ids] = q0
        data.mocap_pos[controller.mocap_id] = np.array([0.4, 0.002, 0.004])
        data.ctrl[8] = -20
        data.ctrl[7] = -20
        mujoco.mj_forward(model, data)

        cable_pos_current = None  # or initial_cable_pos if defined

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

            U_opt, costs = mppi_optimize(x_current, p_target, Q, R, S)
            all_costs_per_step.append(costs)

            U_opt_smoothed = smooth_control_sequence(U_opt, window_size=1)
            u_apply = U_opt_smoothed[0]

            x = np.clip(u_apply[0], -0.2, 0.2)
            y = np.clip(u_apply[1], 0, 0.35)
            rot_z = np.clip(u_apply[3], -1, 1) * 0.5

            data.mocap_pos[controller.mocap_id] = np.array([x + 0.4, y, 0.004])
            controller.get_ctrl()
            p_current_now = x_current[8:26].reshape(-1, 2).cpu().numpy()
            p_target_cpu = p_target.cpu().numpy()
            error = np.linalg.norm(p_current_now - p_target_cpu, axis=1)
            error_list.append(error)
            
            for _ in range(20):
                controller.get_ctrl()
                mujoco.mj_step(model, data)
                # viewer.sync()  # REMOVE THIS LINE

            if np.all(np.abs(error) < 0.03):
                elapsed_time = time.time() - start_time
                print(f"Trial {trial+1}: Converged after {step+1} iterations (Time: {elapsed_time:.4f}s)")
                converged = True
                steps_to_convergence = step + 1
                break

        trial_results.append((converged, steps_to_convergence))
        

    # Summary of results after all trials:
    num_converged = sum(1 for result in trial_results if result[0])
    print(f"Out of {NUM_TRIALS} trials, {num_converged} converged.")

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

    # Optionally, save or further process the results
    # For example, write to a CSV file:
    log_results_and_parameters(
        trial_results=trial_results,
        MAX_ITERATIONS=MAX_ITERATIONS,
        horizon = horizon,
        dt = dt,
        control_dim=control_dim,
        NUM_SAMPLES=NUM_SAMPLES,
        TEMPERATURE=TEMPERATURE,
        STD_DEV=[std_x, std_y, std_z, std_d],
        Q_val=[Q1, Q2],
        R_val=[R1, R2],
        S_val=S1,
        cutoff=cutoff,
        fs=fs,
        filter_order=filter_order
    )
                
    # # Create error plot
    # plt.figure(figsize=(8, 5))
    # plt.plot(steps, total_error, label="Error", color="b")
    # plt.xlabel("Optimization Step")
    # plt.ylabel("Euclidean Distance")
    # plt.title("Evolution of the shape error per optimization step")
    # plt.legend()
    # plt.grid()
    # plt.savefig("error_plot.png", dpi=300, bbox_inches="tight")  # Save high resolution image
    # plt.close()

    # # Create cost plot
    # plt.figure(figsize=(8, 5))
    # plt.plot(steps, mean_costs, label="Mean Costs", color="b")
    # plt.fill_between(steps, mean_costs - std_costs, mean_costs + std_costs, color="b", alpha=0.2)
    # plt.xlabel("Optimization Step")
    # plt.ylabel("Cost")
    # plt.title("Evolution of cost per optimization step")
    # plt.legend()
    # plt.grid()

    # # Save the diagram
    # plt.savefig("kosten_plot.png", dpi=300, bbox_inches="tight")  # Save high resolution image
    # plt.close()
    