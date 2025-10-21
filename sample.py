import colorednoise as cn
import matplotlib.pyplot as plt
import numpy as np

def colored_samples(beta, num_samples, sigma=None, mu=None):

    noise = cn.powerlaw_psd_gaussian(beta, num_samples)
    
    if sigma is None or mu is None:
        return noise
    else:
        return noise * sigma + mu

def cost_function(x):
    return 5-x

def find_elites(samples, costs, num_elites):
    elite_indices = costs.argsort()[:num_elites]
    elites = samples[elite_indices]
    return elites

def draw_samples(mean, std_dev, num_samples):
    return np.random.normal(mean, std_dev, num_samples)