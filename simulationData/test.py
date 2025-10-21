import numpy as np

# Load the .npz file
data = np.load('/home/lukas-zeh/Documents/cablempc/simulationData/target.npz')

# Print the keys (array names) in the file
print("Arrays in the file:", data.files)

# If samples/trajectories are stored with the first dimension as the number of samples
# This is the most common structure
for key in data.files:
    print(f"{key}: shape {data[key].shape}")
    
    # If this is a multidimensional array, the first dimension is usually the number of samples
    if len(data[key].shape) > 1:
        print(f"{key}: number of samples {data[key].shape[0]}")
    else:
        print(f"{key}: single array of length {len(data[key])}")

# If all arrays have the same first dimension, that's likely the number of samples
if len(data.files) > 0:
    first_dims = [data[key].shape[0] for key in data.files if len(data[key].shape) > 0]
    if len(set(first_dims)) == 1:
        print(f"\nAll arrays have the same first dimension: {first_dims[0]} (likely number of samples/trajectories)")