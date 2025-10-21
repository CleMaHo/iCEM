import torch
from torch.utils import data as data_utils
import numpy as np
import json


class SimulationDataset(data_utils.Dataset):
    def __init__(self, file_path):

        data = np.load(file_path)
        self.EE_pos = torch.tensor(data['EE_pos'][:, [0, 1, 2, 4]], dtype=torch.float32)  # Form: (_, 4)
        self.EE_vel = torch.tensor(data['EE_vel'][:, [0, 1, 2, 5]], dtype=torch.float32) * 0.02 # Form: (_, 4)
        self.cable_pos = torch.tensor(data['cable_pos'], dtype=torch.float32)  # Form: (_, 50, 3)
        self.cable_pos_noise = torch.tensor(data['cable_pos'], dtype=torch.float32)  # Form: (_, 50, 3)
        self.add_noise(std_EE=0, std_vel=0, std_cable=0.001)
        self.relativ_cable_pos = self.precompute_relative_positions_tEE()


        # self.cable_vel mit verrauschten Daten berechnen
        self.cable_vel = torch.diff(self.cable_pos_noise, dim=0)  # Geändert von self.cable_pos
        self.cable_vel_selected = self.cable_vel[..., :2]  # Form: [1, 1, 2]

        # Normalizing
        self.EE_pos_mean = self.EE_pos.mean(dim=0)
        self.EE_pos_std = self.EE_pos.std(dim=0)
        self.EE_vel_mean = self.EE_vel.mean(dim=0)  # Shape: (D,)
        self.EE_vel_std = self.EE_vel.std(dim=0)    # Shape: (D,)
        self.l2_relativ_norm_cable_pos = torch.norm(self.relativ_cable_pos, dim=2, keepdim=True)
        self.cable_vel_mean = self.cable_vel.mean(dim=(0, 1), keepdim=True)  # Globaler Mittelwert
        self.cable_vel_std = self.cable_vel.std(dim=(0, 1), keepdim=True)    # Globale Std
        self.cable_vel_mean_2D = self.cable_vel_selected.mean(dim=(0, 1), keepdim=True)  # Globaler Mittelwert
        self.cable_vel_std_2D = self.cable_vel_selected.std(dim=(0, 1), keepdim=True)    # Globale Std

        eps = 1e-9
        # Normalisierung der Daten
        self.EE_pos_normalized = (self.EE_pos - self.EE_pos_mean) / (self.EE_pos_std + eps)


        self.EE_vel_normalized = (self.EE_vel - self.EE_vel_mean) / (self.EE_vel_std + eps)
        # Normalisierung mit epsilon
        self.cable_pos_normalized = self.relativ_cable_pos / (self.l2_relativ_norm_cable_pos + eps)  # Epsilon hinzugefügt
        self.cable_vel_normalized = (self.cable_vel - self.cable_vel_mean) / (self.cable_vel_std + eps)


        stats = {
            #"EE_pos_L2_norm": self.l2_norm_EE_pos.tolist(),
            "EE_vel_mean": self.EE_vel_mean.tolist(),
            "EE_vel_std": self.EE_vel_std.tolist(),
            #"cable_pos_L2": self.l2_norm_cable_pos.tolist(),
            "cable_vel_mean": self.cable_vel_mean.tolist(),
            "cable_vel_std": self.cable_vel_std.tolist()
        }
        with open("normalization_stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        self.num_markers = 9

        # Debug-Ausgaben
        print("EE_pos normalized Shape", self.EE_pos_normalized.shape)  # Erwartet: (_, 4)
        print("EE_vel normalized Shape", self.EE_vel_normalized.shape)  # Erwartet: (_, 4)
        print("cable_pos normalized Shape", self.cable_pos_normalized.shape)  # Erwartet: (_, 50, 3)

    def precompute_relative_positions(self):

            T = self.cable_pos_noise.shape[0]  # Anzahl Zeitschritte (Samples)
            all_rel_positions = []

            for t in range(T):
                # 1) Marker-Koordinaten auswählen
                coords = self.cable_pos_noise[t, :, :2]  # (50,2)
                coords = coords[6:-3]                    # z.B. Marker herausfiltern
                coords = coords[::5]                     # jeden 5. Marker => 9 Marker

                # 2) Differenzen zwischen benachbarten Markern
                relative_DLO = coords[1:] - coords[:-1]  # (8,2)

                # 3) Erster Marker relativ zur EE-Position
                EE_pos = self.EE_pos[t, :2]              # (2,)
                first_marker_relative_to_EE = coords[0] - EE_pos.unsqueeze(0)  # (1,2)

                # 4) Kombinieren => (9,2)
                rel_coords_t = torch.cat((first_marker_relative_to_EE, relative_DLO), dim=0)
                all_rel_positions.append(rel_coords_t)

            # Als 3D-Tensor stapeln: (T, 9, 2)
            precomputed_rel_positions = torch.stack(all_rel_positions, dim=0)
            return precomputed_rel_positions

    def precompute_relative_positions_tEE(self):
        T = self.cable_pos_noise.shape[0]
        all_rel_positions = []  # Liste für alle Zeitschritte

        for t in range(T):
            coords = self.cable_pos_noise[t, :, :2]  # (50,2)
            coords = coords[6:-3]                    # z.B. Marker herausfiltern
            coords = coords[::5]
            EE_pos = self.EE_pos[t, :2]
            rel_positions = coords - EE_pos.unsqueeze(0)  # (9, 2)

            all_rel_positions.append(rel_positions)  # Zur Liste hinzufügen

        # Jetzt erst stapeln: Liste -> Tensor (T, 9, 2)
        precomputed_rel_positions = torch.stack(all_rel_positions, dim=0)
        return precomputed_rel_positions

    def get_local_relativ_positions(self, idx):
        """
        Gibt die lokalen relativen 2D-Positionen zurück, einschließlich
        des ersten Markers relativ zur EE-Position.
        """
        coords = self.cable_pos_noise[idx, :, :2]  # Nur x, y-Koordinaten
        coords = coords[6:-3]  # Marker filtern
        coords = coords[::5]  # Marker auswählen (jeden 5.)

        # Differenzen zwischen benachbarten Markern berechnen
        relative_DLO = coords[1:] - coords[:-1]

        # Erster Marker relativ zur EE-Position
        EE_pos = self.EE_pos[idx, :2]  # Nur x, y-Koordinaten der EE-Position
        first_marker_relative_to_EE = coords[0] - EE_pos.unsqueeze(0)

        # Kombinieren: [Erster Marker relativ zur EE, dann Differenzen]
        relative_DLO = torch.cat((first_marker_relative_to_EE, relative_DLO), dim=0)

        return relative_DLO


    def add_noise(self, std_EE=0.001, std_vel=0.001, std_cable=0.002):
        noise_EE = torch.normal(0, std_EE, size=self.EE_pos.shape)
        noise_vel = torch.normal(0, std_vel, size=self.EE_vel.shape)
        noise_cable = torch.normal(0, std_cable, size=self.cable_pos.shape)

        self.EE_pos += noise_EE
        self.EE_vel += noise_vel
        self.cable_pos_noise += noise_cable

    def get_norm_absolute_coords(self, idx):
        absolute_coords = self.cable_pos_normalized[idx, :, :2]
        absolute_coords = absolute_coords[6:-3]
        absolute_coords = absolute_coords[::5]

        return absolute_coords.flatten()

    def get_absolute_coords(self, idx):
        absolute_coords = self.cable_pos[idx, :, :2]
        absolute_coords = absolute_coords[6:-3]
        absolute_coords = absolute_coords[::5]

        return absolute_coords.flatten()

    def get_norm_marker_velocity(self, idx):
        """
        Berechnet die normierte Geschwindigkeit der Marker.
        :param idx: Index des aktuellen Zeitpunkts
        :return: Normierte Geschwindigkeit (9 Marker, 2D)
        """
        if idx == 0:

            return torch.zeros(self.num_markers, 2)

        norm_velocity = self.cable_vel_normalized[idx, :, :2]
        norm_velocity = norm_velocity[6:-3]
        norm_velocity = norm_velocity[::5]

        return norm_velocity


    def __len__(self):
        return len(self.EE_pos) - 2

    def __getitem__(self, idx):
        # Roboterzustand
        current_robot_state_pos = self.EE_pos_normalized[idx].unsqueeze(0)  # Shape: (1, 4)
        current_robot_state_vel = self.EE_vel_normalized[idx].unsqueeze(0)  # Shape: (1, 4)
        current_robot_state = torch.cat((current_robot_state_pos, current_robot_state_vel), dim=1)  # Shape: (1, 8)

        # Markerzustand
        current_relative_dlo_state = self.cable_pos_normalized[idx]  # Shape: (9, 2)
        if idx > 0:
            current_relative_dlo_velocity = self.get_norm_marker_velocity(idx - 1)  # Shape: (9, 2)
        else:
            current_relative_dlo_velocity = torch.zeros(self.num_markers, 2)  # Shape: (9, 2)

        current_dlo_state = torch.cat((current_relative_dlo_state, current_relative_dlo_velocity), dim=1)  # Shape: (9, 4)

        # Wiederhole Roboterzustand für jeden Marker
        repeated_robot_state = current_robot_state.repeat(self.num_markers, 1)  # Shape: (9, 8)

        # Kombiniere Roboter- und Markerzustand
        inputs = torch.cat((repeated_robot_state, current_dlo_state), dim=1)  # Shape: (9, 12)
        inputs = inputs.reshape(-1)  # Flache Eingabe für MLP: Shape: (num_markers * combined_features,)

        # Zielwerte vorbereiten
        target = self.get_norm_marker_velocity(idx).reshape(-1)  # Shape: (num_markers * 2,)

        return inputs, target
