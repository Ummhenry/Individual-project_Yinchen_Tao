import numpy as np
import matplotlib.pyplot as plt

def normalize_and_scale(data, low, high, target_low, target_high):
    return ((data - low) / (high - low)) * (target_high - target_low) + target_low

def compute_distance(point1, point2, scale_factors=(1.0, 1.0, 1.0)):
    dx, dy, dz = point1
    sx, sy, sz = point2
    scale_x, scale_y, scale_z = scale_factors
    distance_squared = ((dx - sx) * scale_x) ** 2 + ((dy - sy) * scale_y) ** 2 + ((dz - sz) * scale_z) ** 2
    return np.sqrt(distance_squared)

# Create a 3D array
volume_data = np.zeros([100, 100, 150])

# Define core positions
core1 = (50, 50, 50)
core2 = (50, 50, 110)

# Populate the data array
for i in range(100):
    for j in range(100):
        for k in range(150):
            noise = np.random.normal(scale=0.25)
            if compute_distance((i, j, k), core1) <= 21:
                volume_data[i, j, k] = 1
            elif k <= 63 and compute_distance((i, j, k), core1, (1.0, 1.0, 1.1)) < 42:
                volume_data[i, j, k] = 2
            elif k >= 119 and compute_distance((i, j, k), core2, (1.0, 1.0, 1.1)) < 19:
                volume_data[i, j, k] = 2
            elif 63 < k < 119:
                radius = normalize_and_scale(k, 119, 63, 18.2, 40.7)
                if compute_distance((i, j, 0), (50, 50, 0)) < radius:
                    volume_data[i, j, k] = 2
            if volume_data[i, j, k] == 0:
                noise = np.random.normal(scale=0.4)
                if k <= 63 and compute_distance((i, j, k), core1, (1.0, 1.0, 1.1)) < 47 + noise:
                    volume_data[i, j, k] = 4
                if k >= 119 and compute_distance((i, j, k), core2, (1.0, 1.0, 1.1)) < 24 + noise:
                    volume_data[i, j, k] = 4
                if 63 < k < 119:
                    radius = normalize_and_scale(k, 119, 63, 23.4, 45.9)
                    if compute_distance((i, j, 0), (50, 50, 0)) < radius + noise:
                        volume_data[i, j, k] = 4

# Plotting slices
plt.figure(figsize=(12, 4))
slices = [
    np.transpose(volume_data[50, :, :]),
    np.transpose(volume_data[:, 50, :]),
    volume_data[:, :, 50]
]
titles = ['Vertical Slice along X-axis (x=50)', 'Vertical Slice along Y-axis (y=50)', 'Slice along Z-axis (z=50)']

for index, slice_data in enumerate(slices):
    plt.subplot(1, 3, index + 1)
    plt.imshow(slice_data, cmap='gray')
    plt.title(titles[index])

plt.tight_layout()
plt.show()

# Function to save slices as PGM files
def save_slice_as_pgm(slice_data, filename):
    max_value = 255
    image_max = slice_data.max()
    normalized = (slice_data / image_max * max_value) if image_max > 0 else np.zeros_like(slice_data)
    normalized = np.nan_to_num(normalized).astype(int)
    header = f'P2\n{slice_data.shape[1]} {slice_data.shape[0]}\n{max_value}\n'
    with open(filename, 'w') as file:
        file.write(header)
        for row in normalized:
            file.write(' '.join(str(v) for v in row) + '\n')

# Save slices
dimensions = ['x', 'y', 'z']
for dim_index, dim in enumerate([volume_data.shape[0], volume_data.shape[1], volume_data.shape[2]]):
    for slice_index in range(dim):
        if dimensions[dim_index] == 'x':
            slice_to_save = np.transpose(volume_data[slice_index, :, :])
        elif dimensions[dim_index] == 'y':
            slice_to_save = np.transpose(volume_data[:, slice_index, :])
        else:
            slice_to_save = volume_data[:, :, slice_index]
        save_slice_as_pgm(slice_to_save, f'healthy_{dimensions[dim_index]}_{slice_index}.pgm')
