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
import itertools

sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\utils') 
from IK_NS import InverseKinematicsController
from getDataset import SimulationDataset
from logging_utils import log_results_and_parameters
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\models\biLSTM') 
from biLSTM_module import biLSTM
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\control\MPPI')
from mpc_mujoco_mppi_franka import mpc_franka
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

# Define your best parameters here
best_params = {
    "horizon": 20,
    "dt": 0.1,
    "control_dim": 4,
    "NUM_SAMPLES": 100,
    "TEMPERATURE": 0.005,
    "std_x": 0.2,
    "std_y": 0.3,
    "std_z": 1.0,
    "std_d": 2,
    "Q1": 18,
    "Q2": 300,
    "S1": 10,
    "R1": 4,
    "R2": 20,
    "cutoff": 2.0,
    "fs": 10.0,  # Assuming fs is the sampling frequency
    "device": device
}

# Run only once with best parameters
print("=== Running with best parameters ===")
for k, v in best_params.items():
    print(f"  {k}: {v}")

mpc_franka(**best_params)
