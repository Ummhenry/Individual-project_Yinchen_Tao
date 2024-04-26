import numpy as np
import matplotlib.pyplot as plt

def scale_value(x, min_input, max_input, min_target, max_target):
    # Rescale a value from one range to another
    return (x - min_input) / (max_input - min_input) * (max_target - min_target) + min_target

def calculate_distance(point1, point2, compression=(1.0, 1.0, 1.0)):
    # Calculate the compressed Euclidean distance between two points
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x_comp, y_comp, z_comp = compression
    distance_squared = ((x1 - x2) * x_comp) ** 2 + ((y1 - y2) * y_comp) ** 2 + ((z1 - z2) * z_comp) ** 2
    return np.sqrt(distance_squared)

# Initialize a 3D numpy array with zeros
grid = np.zeros([100, 100, 150])

# Define the central points for the spheres
center1 = (50, 50, 50)
center2 = (50, 50, 110)

# Populate the grid with different values based on distance from centers
for x in range(100):
    for y in range(100):
        for z in range(150):
            # Assign values based on proximity to center points
            if calculate_distance((x, y, z), center1) <= 21:
                grid[x, y, z] = 1
            elif z <= 64 and calculate_distance((x, y, z), center1, (1.0, 1.0, 1.1)) < 43:
                grid[x, y, z] = 2
            elif z >= 119 and calculate_distance((x, y, z), center2, (1.0, 1.0, 1.1)) < 20:
                grid[x, y, z] = 2
            elif 64 < z < 119:
                radius = scale_value(z, 64, 119, 41.7, 19.2)
                if calculate_distance((x, y, 0), (50, 50, 0)) < radius:
                    grid[x, y, z] = 2

            # Add noise to regions outside the defined structures
            if grid[x, y, z] == 0:
                noise = np.random.normal(scale=0.4)
                if z <= 60 and calculate_distance((x, y, z), center1, (1.0, 1.0, 1.1)) < 47 + noise:
                    grid[x, y, z] = 4
                if z >= 115 and calculate_distance((x, y, z), center2, (1.0, 1.0, 1.1)) < 24 + noise:
                    grid[x, y, z] = 4
                if 60 < z < 115:
                    noise_radius = scale_value(z, 64, 119, 45.9, 23.4)
                    if calculate_distance((x, y, 0), (50, 50, 0)) < noise_radius + noise:
                        grid[x, y, z] = 4
# rotten part gen
xr,yr= np.random.uniform(70,93.0,[2])
zr=np.random.uniform(11.0,110.0)
rr=np.random.uniform(5.0,15.0)
cr =(xr,yr,zr)
for i in range(30):
 xr +=np.random.uniform(-rr,rr)
 yr +=np.random.uniform(-rr,rr)
 zr +=np.random.uniform(-rr,rr)
 cr=(xr,yr,zr)
 rr=np.random.uniform(5.0,15.0)
 if xr<0 or yr<0 or zr<0:
  continue
 if xr>=100 or yr >=100 or zr>=150:
  continue
 for x in range(100):
  for y in range(100):
   for z in range(150):
    if grid[x,y,z]!= 0 and grid[x,y,z]!=1 and calculate_distance((x,y,z),cr) <=rr:
     grid[x,y,z]=3

# Random movement and interaction simulation
for _ in range(50):
    x, y = np.random.uniform(0, 100, 2)
    direction = np.random.uniform(0, 2 * np.pi)
    vx, vy = np.cos(direction), np.sin(direction)
    for z in range(150):
        ax, ay = np.random.normal(scale=0.1, size=2)
        vx += ax
        vy += ay
        velocity = np.sqrt(vx**2 + vy**2)
        if velocity > 1.2:
            vx, vy = vx / velocity * 1.2, vy / velocity * 1.2
        x += vx
        y += vy
        x = max(0, min(x, 99))
        y = max(0, min(y, 99))
        for cx in range(int(x) - 3, int(x) + 4):
            for cy in range(int(y) - 3, int(y) + 4):
                for cz in range(z - 3, z + 4):
                    if 0 <= cx < 100 and 0 <= cy < 100 and 0 <= cz < 150:
                        if grid[cx, cy, cz] == 2 and calculate_distance((x, y, z), (cx, cy, cz)) <= 3.0:
                            grid[cx, cy, cz] = 3

# Display the slices of the grid
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(grid[50, :, :]), cmap='gray')
plt.title('Vertical Slice along X-axis (x=50)')
plt.subplot(1, 3, 2)
plt.imshow(np.transpose(grid[:, 50, :]), cmap='gray')
plt.title('Vertical Slice along Y-axis (y=50)')
plt.subplot(1, 3, 3)
plt.imshow(grid[:, :, 75], cmap='gray')
plt.title('Slice along Z-axis (z=75)')
plt.tight_layout()
plt.show()
