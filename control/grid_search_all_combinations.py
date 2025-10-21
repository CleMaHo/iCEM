import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use('Agg')
import sys
import itertools

sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\utils') 
from IK_NS import InverseKinematicsController
from getDataset import SimulationDataset
from logging_utils import log_results_and_parameters
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\models\biLSTM') 
from biLSTM_module import biLSTM
sys.path.append(r'C:\Users\cleme\Desktop\HiWi-Stellen\ISW\CableML\cablempc\control\MPPI')
from mpc_mujoco_mppi_franka_function import mpc_franka_lp

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

# Define parameter grid
param_options = {
    "horizon": [20],
    "dt": [0.2],
    "control_dim": [4],
    "NUM_SAMPLES": [200],
    "TEMPERATURE": [0.002],
    "std_x": [0.4],
    "std_y": [0.2],
    "std_z": [0.2],
    "std_d": [0.5],
    "Q1": [18],
    "Q2": [300],
    "S1": [10],
    "R1": [4],
    "R2": [100],
    "cutoff": [0.5, 1.0, 2.0],
    "fs": [5.0],  # Assuming fs is the sampling frequency
    "filter_order": [2.0, 3.0, 4.0],
    "device": [device]
}

# Generate full parameter combinations (Cartesian product)
param_names = list(param_options.keys())
param_values = list(param_options.values())
all_combinations = list(itertools.product(*param_values))
print(f"Total available combinations: {len(all_combinations)}")

# # Randomly sample 450 unique combinations
# random.seed(42)  # for reproducibility
# sampled_combinations = random.sample(all_combinations, 450)
# print(f"Sampled 450 random combinations.")

# Run the sampled combinations
for idx, combination in enumerate(all_combinations):
    test_params = dict(zip(param_names, combination))
    print(f"\n=== Running combination {idx + 1}/{len(all_combinations)} ===")
    for k, v in test_params.items():
        print(f"  {k}: {v}")
    
    try:
        mpc_franka_lp(**test_params)
    except Exception as e:
        print(f"⚠️ Error in combination {idx + 1}: {e}")
