import time
import torch
import numpy as np
from scipy.stats import norm, qmc
import json
import os
import colorednoise
import sys
import matplotlib.pyplot as plt

class MPPIController:
    
    def __init__(self, model, device,
                 Q, R, S, horizon, control_dim,
                 num_samples, std_dev, temperature, model_type="biLSTM"):
        """
        Args:
            model: Das neuronale Netz für die Vorhersage (nn.Module)
            pred_mocap_ids: Benötigte IDs für die Vorhersage
            data: Zusätzliche Daten, die im Forward-Model benötigt werden
            dataset: Datensatz, der u.a. Normierungsparameter liefert
            device: torch.device (z.B. "cuda" oder "cpu")
            Q, R, S: Kostenmatrizen
            horizon (int): Planungshorizont (Anzahl der Zeitschritte)
            control_dim (int): Dimension der Steuerung
            num_samples (int): Anzahl der Samples (Batch-Größe) in MPPI
            std_dev: Standardabweichung (Tensor oder numpy-Array) für die Steuerungen
            temperature (float): Temperaturparameter für die Gewichtung
        """
        self.model = model
        self.device = device

        self.Q = torch.eye(18).to(device) * Q
        self.R = torch.eye(4).to(device) * R
        self.S = torch.eye(control_dim).to(device) * S
        self.model_type = model_type

        self.horizon = horizon
        self.control_dim = control_dim
        self.num_samples = num_samples
        self.STD_DEV = torch.tensor([0.2, 0.2, 0.0, 0.02], device=device) #torch.tensor([0.3, 0.3, 0.0, 0.03] Hier tunen um Exploration zu erhöhen oder zu verringern
        # exploration in x, y, z == 0, and rotation um z
        self.TEMPERATURE = temperature
        
        norms_path = "./FrankaControlInterface/dashboard/normalization_stats.json"
        with open(
            os.path.join(sys.path[0], norms_path) if norms_path[0] != "/" else norms_path, "r"
        ) as f:
            self.norm_data = json.load(f)

    def biLSTM_forward_model_batch(self, x_batch, u_batch, dt=0.02):
        """
        Führt einen Vorwärtsschritt für einen Batch von Zuständen x_batch
        und Steuerbefehlen u_batch durch.
        """

        batch_size = x_batch.shape[0]

        # Die EE-Information ist in jedem Marker identisch (weil sie durch repeat eingefügt wurde)
        # Es reicht also, z. B. den ersten Marker zu verwenden:
        ee_pos = x_batch[:, 0:4]  # (batch_size, 4)

        ee_vel = x_batch[:, 4:8]  # (batch_size, 4)


        # DLO (Marker) Information: jeweils 2 Werte für Position und 2 für Velocity pro Marker
        dlo_pos = x_batch[:, 8:26].view(batch_size, 9, 2)

        dlo_vel = x_batch[:, 26:44].view(batch_size, 9, 2)

        EE_Vel_mean = torch.tensor(self.norm_data["EE_vel_mean"]).to(self.device)
        EE_Vel_std = torch.tensor(self.norm_data["EE_vel_std"]).to(self.device)
        DLO_Vel_mean = torch.tensor(self.norm_data["cable_vel_mean"]).to(self.device)
        DLO_Vel_std = torch.tensor(self.norm_data["cable_vel_std"]).to(self.device)
        L2_norm_data = torch.norm(ee_pos, p=2, dim=1, keepdim=True)  # (batch_size, 1)     
        EE_Vel_normal = (ee_vel - EE_Vel_mean) / (EE_Vel_std + 1e-8) # Add 1e-8 to avoid division by zero

        
        # Berechne neuen EE-Zustand
        ee_vel_new = ee_vel + u_batch  # (batch_size, 4)
        v_max = 0.05
        ee_vel_new = torch.clamp(ee_vel_new, -v_max, v_max)
        ee_pos_new = ee_pos + ee_vel_new * dt  # (batch_size, 4)

        # Normalisierung (Kabel- sowie EE-Zustände)
        eps = 1e-9
        l2_norm_cable_pos = torch.norm(dlo_pos, p=2, dim=2, keepdim=True)  # (batch_size, 9, 1)

        ee_pos_normalized = ee_pos / (L2_norm_data + 1e-8) # Add 1e-8 to avoid division by zero
        ee_vel_normalized = (ee_vel_new - EE_Vel_mean) / (EE_Vel_std + eps)

        cable_pos_normalized = dlo_pos / (l2_norm_cable_pos + eps)
        cable_vel_normalized = (dlo_vel - DLO_Vel_mean) / (DLO_Vel_std + eps)

        # Wiederhole EE-Zustände für jeden Marker
        robot_state_pos = ee_pos_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
        robot_state_vel = ee_vel_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
        robot_state = torch.cat((robot_state_pos, robot_state_vel), dim=2)  # (batch_size, 9, 8)

        # Kombiniere mit Kabelzuständen
        dlo_state = torch.cat((cable_pos_normalized, cable_vel_normalized), dim=2)  # (batch_size, 9, 4)
        inputs = torch.cat((robot_state, dlo_state), dim=2)  # (batch_size, 9, 12)

        # Umformen für das Netzwerk: (batch_size * 9, 12)
        inputs = inputs.view(-1, 12).float() 

        # Batch-Inferenz
        with torch.no_grad():
            norm_predicted_velocity = self.model(inputs)  # (batch_size * 9, 2)
        norm_predicted_velocity = norm_predicted_velocity.view(batch_size, 9, 2)

        # Denormalisierung der Vorhersage
        predicted_velocity = norm_predicted_velocity * DLO_Vel_std + DLO_Vel_mean

        # Zustandsaktualisierung
        predicted_feature_pos = dlo_pos + predicted_velocity  # (batch_size, 9, 2)
        x_next_batch = torch.cat([
            ee_pos_new,
            ee_vel_new,
            predicted_feature_pos.view(batch_size, -1),
            predicted_velocity.view(batch_size, -1)
        ], dim=1)  # (batch_size, 44)
        return x_next_batch
    
    
    def MLP_forward_model_batch(self, x_batch, u_batch, dt=0.02):
        """
        Führt einen Vorwärtsschritt für einen Batch von Zuständen x_batch
        und Steuerbefehlen u_batch durch.
        """

        batch_size = x_batch.shape[0]

        # Die EE-Information ist in jedem Marker identisch (weil sie durch repeat eingefügt wurde)
        # Es reicht also, z. B. den ersten Marker zu verwenden:
        ee_pos = x_batch[:, 0:4]  # (batch_size, 4)

        ee_vel = x_batch[:, 4:8]  # (batch_size, 4)


        # DLO (Marker) Information: jeweils 2 Werte für Position und 2 für Velocity pro Marker
        dlo_pos = x_batch[:, 8:26].view(batch_size, 9, 2)

        dlo_vel = x_batch[:, 26:44].view(batch_size, 9, 2)

        EE_Vel_mean = torch.tensor(self.norm_data["EE_vel_mean"]).to(self.device)
        EE_Vel_std = torch.tensor(self.norm_data["EE_vel_std"]).to(self.device)
        DLO_Vel_mean = torch.tensor(self.norm_data["cable_vel_mean"]).to(self.device)
        DLO_Vel_std = torch.tensor(self.norm_data["cable_vel_std"]).to(self.device)
        L2_norm_data = torch.norm(ee_pos, p=2, dim=1, keepdim=True)  # (batch_size, 1)     
        EE_Vel_normal = (ee_vel - EE_Vel_mean) / (EE_Vel_std + 1e-8) # Add 1e-8 to avoid division by zero

        
        # Berechne neuen EE-Zustand
        ee_vel_new = ee_vel + u_batch  # (batch_size, 4)
        v_max = 0.05
        ee_vel_new = torch.clamp(ee_vel_new, -v_max, v_max)
        ee_pos_new = ee_pos + ee_vel_new * dt  # (batch_size, 4)

        # Normalisierung (Kabel- sowie EE-Zustände)
        eps = 1e-9
        l2_norm_cable_pos = torch.norm(dlo_pos, p=2, dim=2, keepdim=True)  # (batch_size, 9, 1)

        ee_pos_normalized = ee_pos / (L2_norm_data + 1e-8) # Add 1e-8 to avoid division by zero
        ee_vel_normalized = (ee_vel_new - EE_Vel_mean) / (EE_Vel_std + eps)

        cable_pos_normalized = dlo_pos / (l2_norm_cable_pos + eps)
        cable_vel_normalized = (dlo_vel - DLO_Vel_mean) / (DLO_Vel_std + eps)

        # Wiederhole EE-Zustände für jeden Marker
        robot_state_pos = ee_pos_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
        robot_state_vel = ee_vel_normalized.unsqueeze(1).repeat(1, 9, 1)  # (batch_size, 9, 4)
        robot_state = torch.cat((robot_state_pos, robot_state_vel), dim=2)  # (batch_size, 9, 8)

        # Kombiniere mit Kabelzuständen
        dlo_state = torch.cat((cable_pos_normalized, cable_vel_normalized), dim=2)  # (batch_size, 9, 4)
        inputs = torch.cat((robot_state, dlo_state), dim=2)  # (batch_size, 9, 12)
        
        #flatten inputs
        inputs = inputs.flatten(start_dim=1)
        inputs = inputs.to(dtype=torch.float32) 

        # Umformen für das Netzwerk: (batch_size * 9, 12)
        #inputs = inputs.view(-1, 12).float() 

        # Batch-Inferenz
        with torch.no_grad():
            norm_predicted_velocity = self.model(inputs)  # (batch_size * 9, 2)
        norm_predicted_velocity = norm_predicted_velocity.view(batch_size, 9, 2)

        # Denormalisierung der Vorhersage
        predicted_velocity = norm_predicted_velocity * DLO_Vel_std + DLO_Vel_mean

        # Zustandsaktualisierung
        predicted_feature_pos = dlo_pos + predicted_velocity  # (batch_size, 9, 2)
        x_next_batch = torch.cat([
            ee_pos_new,
            ee_vel_new,
            predicted_feature_pos.view(batch_size, -1),
            predicted_velocity.view(batch_size, -1)
        ], dim=1)  # (batch_size, 44)
        return x_next_batch

    def cost_function_batch(self, x_batch, u_batch, p_target, u_prev_batch=None):
        """
        Berechnet die Kosten für einen Batch.
        Args:
            x_batch: (batch_size, state_dim)
            u_batch: (batch_size, control_dim)
            p_target: Zielzustand (tensor, bereits flach oder wird geflacht)
            u_prev_batch: Vorheriger Steuerbefehl (oder None)
        Returns:
            Tensor der Form (batch_size,)
        """
        batch_size = x_batch.shape[0]
        p = x_batch[:, 8:26].view(batch_size, -1)  # (batch_size, 18)


        diff = (p - p_target).float()


        shape_cost = 0.5 * (diff @ self.Q * diff).sum(dim=1) # (batch_size,)
        control_cost = 1 * (u_batch @ self.R * u_batch).sum(dim=1)  # (batch_size,)

        return shape_cost + control_cost

    def halton_noise(self, num_samples, horizon, control_dim, std_dev):
        """
        Generiert Halton-basierte Samples und transformiert diese in normalverteilte Samples.
        Returns:
            Tensor der Form (num_samples, horizon, control_dim)
        """
        d = horizon * control_dim
        sampler = qmc.Halton(d=d, scramble=True)
        samples = sampler.random(n=num_samples)  # (num_samples, d)
        samples = samples.reshape(num_samples, horizon, control_dim)
        samples_normal = norm.ppf(samples)
        samples_tensor = torch.tensor(samples_normal, dtype=torch.float32, device=self.device)
        samples_tensor = samples_tensor * std_dev
        return samples_tensor

    def colored_noise(self, num_samples, horizon, control_dim, std_dev, mean, beta=1):
        
        noise = colorednoise.powerlaw_psd_gaussian(beta, size=(num_samples, control_dim, horizon))
        noise = np.transpose(noise, (0, 2, 1))
        noise_tensor = torch.tensor(noise, dtype=torch.float32, device=self.device)
        samples = noise_tensor * std_dev.unsqueeze(0).unsqueeze(0) #+ mean.unsqueeze(0)
        return samples
    
    
    def mppi_optimize(self, x0, p_target, u_init=None, u_prev=None):
        # Initialisierung wie zuvor
        if u_init is None:
            u_init = torch.zeros((self.horizon, self.control_dim), device=self.device)
        else:
            u_init = torch.roll(u_init, -1, dims=0)
            u_init[-1] = 0

        all_costs = []
        p_target_tensor = p_target.to(self.device).flatten()
        
        # Nur eine Iteration (Random Shooting)
        for mppi_iter in range(25):
            # Verwende höhere Standardabweichung für breitere Exploration
            #noise = self.halton_noise(self.num_samples, self.horizon, self.control_dim, self.STD_DEV)
            noise = self.colored_noise(self.num_samples, self.horizon, self.control_dim, self.STD_DEV, u_init, beta=1)
            samples = u_init.unsqueeze(0) + noise
            
            # Forward Rollouts
            x0 = x0.squeeze().unsqueeze(0)
            x_batch = x0.clone().repeat(self.num_samples, 1)
            total_costs = torch.zeros(self.num_samples, device=self.device)
            u_prev_batch = u_prev.clone().repeat(self.num_samples, 1) if u_prev is not None else None

            # Bewerte alle Samples durch Forward-Simulationen
            for t in range(self.horizon):
                u_t = samples[:, t]
                if self.model_type == "biLSTM":
                    x_batch = self.biLSTM_forward_model_batch(x_batch, u_t, dt=0.02)
                else:
                    x_batch = self.MLP_forward_model_batch(x_batch, u_t, dt=0.02)
                total_costs += self.cost_function_batch(x_batch, u_t, p_target_tensor, u_prev_batch)
                u_prev_batch = u_t.clone()
            
            # Verwende sehr niedrigen Temperaturwert für "härtere" Auswahl
            temperature = self.TEMPERATURE # Sehr niedrig → stärkere Gewichtung der besten Samples
            min_cost = torch.min(total_costs)
            weights = torch.exp(-(total_costs - min_cost) / temperature)
            
            # Normalisiere Gewichte
            weights = weights / (torch.sum(weights) + 1e-10)
            weights = weights.view(-1, 1, 1)
            
            # Gewichtete Summe aller Samples
            u_init = torch.sum(weights * samples, dim=0)
            all_costs.append(torch.mean(total_costs).item())
            
            # Für Debugging: Zeige, wie viele Samples einen signifikanten Beitrag leisten
            effective_samples = 1.0 / torch.sum(weights**2)
            print(f"Effektive Sample-Anzahl: {effective_samples.item():.1f} von {self.num_samples}")

        return u_init.cpu().numpy(), all_costs
