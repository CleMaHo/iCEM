import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn
import torch

# ===============================
# Simple mock dynamics + cost
# ===============================

def mock_dynamics(x0_batch, u_samples, dt=0.1):
    """
    Fake dynamics: propagate each control sequence over the horizon.

    Args:
        x0_batch: np.ndarray, shape (num_samples, control_dim)
            Initial state for each sample.
        u_samples: np.ndarray, shape (num_samples, horizon, control_dim)
            Control sequences for each sample.
        dt: float, time step multiplier

    Returns:
        x_trajectory: np.ndarray, shape (num_samples, horizon+1, control_dim)
            Predicted states for each sample at each timestep.
            First timestep is the initial state.
    """
    num_samples, horizon, control_dim = u_samples.shape
    x_trajectory = np.zeros((num_samples, horizon + 1, control_dim))
    x_trajectory[:, 0, :] = x0_batch  # set initial states

    for t in range(horizon):
        x_prev = x_trajectory[:, t, :]
        u_t = u_samples[:, t, :]
        noise = np.random.normal(0, 0.01, u_t.shape)
        x_next = x_prev + dt * u_t + noise
        x_trajectory[:, t + 1, :] = x_next

    return x_trajectory

def cost_function_batch(x_batch, p_target, Q):
    """Quadratic cost combining state error and control effort."""
    diff = x_batch - p_target
    shape_cost = 0.5 * np.sum(diff @ Q * diff, axis=1)
    #control_cost = 0.5 * np.sum(u_batch @ R * u_batch, axis=1)
    #return shape_cost + control_cost
    return shape_cost

def colored_samples(beta, num_samples, horizon, control_dim=4, sigma=None, mu=None, device="cpu"):
    """
    Generate colored (1/f^β) noise samples for control sequences using PyTorch.

    Args:
        beta (float): Power-law exponent (0 = white, 1 = pink, 2 = brown)
        num_samples (int): Number of trajectories to sample
        horizon (int): Number of timesteps per trajectory
        control_dim (int): Control dimensions (e.g., x, y, z, rot)
        sigma (torch.Tensor or float, optional): Std dev for each control dim
        mu (torch.Tensor or float, optional): Mean for each control dim
        device (str or torch.device): Target device

    Returns:
        torch.Tensor: (num_samples, horizon, control_dim)
    """
    # White Gaussian noise in frequency domain
    white = torch.randn(num_samples, control_dim, horizon, device=device)

    # Compute frequency scaling for power-law
    freqs = torch.fft.rfftfreq(horizon, d=1.0).to(device)  # shape (horizon//2 + 1,)
    freqs[0] = 1e-6  # avoid division by zero

    # 1/f^β amplitude scaling
    amplitude = 1.0 / (freqs ** (beta / 2.0))  # sqrt because power ∝ amplitude^2
    amplitude = amplitude.view(1, 1, -1)  # for broadcasting

    # FFT → filter → IFFT
    white_fft = torch.fft.rfft(white, dim=2)
    colored_fft = white_fft * amplitude
    colored = torch.fft.irfft(colored_fft, n=horizon, dim=2)

    # Reorder to (num_samples, horizon, control_dim)
    colored = colored.permute(0, 2, 1)

    # Apply scaling and mean
    if sigma is not None:
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        colored = colored * sigma.view(1, 1, -1)
    if mu is not None:
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu, dtype=torch.float32, device=device)
        colored = colored + mu.view(1, 1, -1)

    return colored


# Core iCEM algorithm

def icem_optimize(x_current, p_target, horizon=20, control_dim=4, num_iterations=500,
                  N_init=200, K=20, y=1.25, sigma_init=1.0, min_sigma=0.05, beta_sigma=0.8, elitefrac=0.3):
    """
    Simplified iCEM mockup returning only the first action of the best control sequence.
    """

    # --- Initialization ---
    mu = torch.zeros(horizon, control_dim, device=device)
    sigma = torch.ones(horizon, control_dim, device=device) * sigma_init

    best_cost_prev = np.inf
    Q = np.eye(control_dim)
    R = 0.01 * np.eye(control_dim)

    history_mean = []
    history_cost = []
    elites = None

    # --- Single optimization iteration (can increase if needed) ---
    for it in range(1):
        

        # iCEM horizon loop
        for t in range(horizon):
            
            all_costs = []
            all_samples = []
        
            # --- Adaptive number of samples ---
            N_i = int(max(N_init * y ** (-t), 2 * K))

            # Add elites from previous step (30%)
            if elites is not None:
                N_elites = int(K * elitefrac)
                N_i = max(2 * K, N_i - N_elites)
            else:
                N_elites = 0

            # --- Sample controls ---
            u_samples = colored_samples(beta=1.0, num_samples=N_i,
                            horizon=horizon, control_dim=control_dim,
                            mu=mu, sigma=sigma, device=device)
            # u_samples.shape -> (N_i, horizon, control_dim)


            # Add previous elites if available
            if elites is not None:
                u_samples = np.vstack([u_samples, elites[:N_elites]])

            # Add mean action for stability at last step
            if t == horizon - 1:
                u_samples = np.vstack([u_samples, np.mean(u_samples, axis=0)])

            # --- Simulate (mock) ---
            x_batch = mock_dynamics(x_current, u_samples)

            # --- Evaluate cost ---
            costs = cost_function_batch(x_batch, u_samples, p_target, Q)

            # --- Store ---
            all_costs.append(costs)
            all_samples.append(u_samples)

            # --- Select elites ---
            elite_ids = np.argsort(costs)[:K]
            elites = u_samples[elite_ids]

            # --- Update distribution ---
            mu[t] = np.mean(elites, axis=0)
            sigma_new = np.std(elites, axis=0)
            sigma[t] = beta_sigma * sigma[t] + (1 - beta_sigma) * sigma_new

            # --- At final horizon step ---
            if t == horizon - 1:
                # add mean action sample
                elites_mean = np.mean(elites, axis=0, keepdims=True)
                elites_mean_cost = cost_function_batch(mock_dynamics(x_current, elites_mean), p_target, Q)
                u_samples = np.vstack([u_samples, elites_mean])
                costs = np.hstack([costs, elites_mean_cost])
                
                # Select best overall
                u_best_id = np.argmin(costs)
                u_best = u_samples[u_best_id:u_best_id + 1]  # shape (1, control_dim)
                best_cost = costs[u_best_id]

                # Simulate full best trajectory
                # x_best = mock_dynamics(x_current, u_best)
                # x_final = x_best[:, -1, :]  # final state

    # --- Return first control action (1, control_dim) ---
    if u_best.ndim == 3:
        u_first = u_best[0, 0, :]     # shape (4,)
    else:
        u_first = u_best[0]           # shape (4,)


    return u_first, best_cost