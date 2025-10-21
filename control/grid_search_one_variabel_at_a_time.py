import numpy as np
import torch
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.stats import qmc, norm
import sys
import random

sys.path.append(r'/home/lukas-zeh/Documents/cablempc/utils') 
from IK_NS import InverseKinematicsController
from getDataset import SimulationDataset
from logging_utils import log_results_and_parameters
sys.path.append(r'/home/lukas-zeh/Documents/cablempc/models/biLSTM') 
from biLSTM_module import biLSTM
sys.path.append(r'/home/lukas-zeh/Documents/cablempc/control/MPPI')
from mpc_mujoco_mppi_franka import mpc_franka
# sys.path.append("./models/MLP")
# from MLP import MLPModel



dataset = SimulationDataset(r'/home/lukas-zeh/Documents/cablempc/simulationData/target.npz')

# Load your model
filepath = r"/home/lukas-zeh/Documents/cablempc/res/scene.xml"
model = mujoco.MjModel.from_xml_path(filepath)
data = mujoco.MjData(model)
# Set additional model settings if needed, e.g. gravity compensation
model.opt.timestep = 0.001

# Controller configuration parameters
site_name = "attachment_site"           # Name of the end-effector site
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
mocap_body_name = "target_EE"             # Name of the mocap body
q0 = np.array([0.108, 0.0716, 0.132, -2.71, -0.000547, 2.79, -0.545])  # Initial joint positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# biLSTM model
nn_model = biLSTM()
nn_model.to(device)
model_path = r'/home/lukas-zeh/Documents/cablempc/models/biLSTM/models/BEST_bi_LSTM_2_rel_vel_bs128_hs256_lr00001_nl3_epochs50_10k.pth'
nn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
nn_model.eval()

# Parameter lists
param_options = {
    "horizon": [5, 10, 20, 50],
    "dt": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    "control_dim": [4],
    "NUM_SAMPLES": [20, 10, 50, 100, 200],
    "TEMPERATURE": [0.002, 0.001, 0.005, 0.01, 0.05],
    "std_x": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2],
    "std_y": [0.2, 0.3, 0.4, 0.5, 1],
    "std_z": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5],
    "std_d": [0.5, 1, 2, 3, 4, 5, 10],
    "Q1": [18],
    "Q2": [50, 150, 300, 600, 1000],
    "S1": [1, 5, 10, 20, 50],
    "R1": [4],
    "R2": [5, 10, 20, 50, 100]
}

# Set default values (first element of each list)
default_params = {k: v[0] for k, v in param_options.items()}
for key, value in default_params.items():
    print(f"{key}: {value}")

# Step 1: Run with default values
print("Initial run with default values:")
mpc_franka(**default_params)

# Step 2: Iterate through each parameter and change only one at a time
for param_name, values in param_options.items():
    for i in range(1, len(values)):
        test_params = default_params.copy()
        test_params[param_name] = values[i]
        print(f"\nRunning with {param_name} set to {values[i]} (index {i}):")
        mpc_franka(**test_params)