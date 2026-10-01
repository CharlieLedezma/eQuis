import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Parameter Setup (Mechanical properties)
link_length = 2.0  # Length of the rotating link
omega = 2.0        # Angular velocity (rad/s)
time_vector = np.linspace(0, 10, 300) # 300 Simulation intervals

# 2. Kinematics Equations (Compute coordinates over time)
# For a rotating link, x = L*cos(theta) and y = L*sin(theta)
theta = omega * time_vector
x_coords = link_length * np.cos(theta)
y_coords = link_length * np.sin(theta)

# 3. Setting up the Plot Window
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Kinematic Link Simulation", fontsize=12)

# Initialize graphic elements (Ground joint, link path, and moving pin)
ground_joint, = ax.plot(0, 0, 'ko', markersize=8, label='Ground Pivot')
link_line, = ax.plot([], [], 'b-', linewidth=3, label='Kinematic Link')
pin_joint, = ax.plot([], [], 'ro', markersize=8, label='Pin Joint')
ax.legend(loc='upper right')

# 4. Initialization Function
def init():
    link_line.set_data([], [])
    pin_joint.set_data([], [])
    return link_line, pin_joint

# 5. Frame Update Function (Calculates state at each frame index)
def update(frame):
    # Origin to current x,y coordinate
    link_line.set_data([0, x_coords[frame]], [0, y_coords[frame]])
    pin_joint.set_data([x_coords[frame]], [y_coords[frame]])
    return link_line, pin_joint

# 6. Execute and Render the Animation
# 'blit=True' maximizes performance by only updating changed parts of the canvas
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=len(time_vector), 
    init_func=init, 
    blit=True, 
    interval=20, # Time delay between frames in milliseconds (~50 FPS)
    repeat=True
)

plt.show()
