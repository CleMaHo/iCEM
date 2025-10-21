import random
import mujoco
import mujoco.viewer
import numpy as np
from tqdm import trange


class SimulationRecorder:
    def __init__(self, model, data, body_name, output_file='simulation_data_rollout_norot'):
        # Initialize the recorder with the model, data, body name, and output file name
        self.model = model
        self.data = data
        self.body_name = body_name
        self.body_index = model.body(body_name).id
        self.body_index_2 = model.body("Box_first").id
        self.cabel_first_index = model.body('CB0').id
        self.cabel_last_index = model.body('cube_body_1').id
        self.output_file = output_file
        self.records = []

    def record(self):
        # Record the current state of the simulation
        EE_euler = np.copy(self.data.qpos[self.body_index + 2:self.body_index + 5])
        EE_pos = np.copy(np.hstack([self.data.xpos[self.body_index], EE_euler]))
        # Record relevant simulation data (qpos and qvel for specific body, and other data)
        self.records.append({
            'time': self.data.time,
            'EE_pos': EE_pos,
            'EE_vel': np.copy(self.data.qvel[self.body_index_2 - 1:self.body_index_2 + 5]),
            'cable_pos': np.copy(self.data.xpos[self.cabel_first_index:self.cabel_last_index]),
        })

    def save(self):
        # Convert recorded data into NumPy arrays and save them as a single .npz file
        time_data = np.array([record['time'] for record in self.records])
        EE_pos_data = np.array([record['EE_pos'] for record in self.records])
        EE_vel_data = np.array([record['EE_vel'] for record in self.records])
        cable_pos_data = np.array([record['cable_pos'] for record in self.records])

        np.savez_compressed(f'{self.output_file}.npz', time=time_data, EE_pos=EE_pos_data, EE_vel=EE_vel_data, cable_pos=cable_pos_data)


# Set manipulation length boundaries in meters
x_min, x_max = -0.2, 0.2
y_min, y_max = 0.05, 0.35
z_min, z_max = 0, 0
rot_y_min, rot_y_max = 0, 0
rot_z_min, rot_z_max = 0, 0

# Load the model and data from XML file
m = mujoco.MjModel.from_xml_path('./res/vari_cable.xml')
d = mujoco.MjData(m)

# Number of datasets
num_datasets = 100
trials_per_dataset = 1  # Trials per dataset

for dataset_idx in range(1, num_datasets + 1):
    # Create Recorder with a unique filename for each dataset
    output_file = f'simulation_data_{dataset_idx:02d}'
    recorder = SimulationRecorder(m, d, 'CB0', output_file)

    print(f"Starting dataset {dataset_idx}/{num_datasets}...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        for element in trange(trials_per_dataset, desc=f"Simulating Dataset {dataset_idx}"):
            # Generate a random trajectory
            random_trajectorie = np.array([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                random.uniform(z_min, z_max),
                0,
                random.uniform(rot_y_min, rot_y_max),
                random.uniform(rot_z_min, rot_z_max),
            ])

            # Set control inputs
            d.ctrl = random_trajectorie
            mujoco.mj_forward(m, d)  # Forward physics to apply the changes

            while viewer.is_running() and d.time < 5:
                mujoco.mj_step(m, d)  # Step the physics simulation
                if d.time % 0.020 < m.opt.timestep:
                    recorder.record()

                viewer.sync()

            # Reset simulation for the next trial
            mujoco.mj_resetData(m, d)

    # Save the current dataset
    recorder.save()
    print(f"Dataset {dataset_idx} saved as {output_file}.npz")

print("All datasets have been created and saved.")
