import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('filtered_velocity_original.csv', header=None)

# Extract the data
time = data[0]
velocity = data[1]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time, velocity, linestyle='-', color='b')
plt.title('Filtered Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
#plt.show()
plt.savefig("fig2.jpg")