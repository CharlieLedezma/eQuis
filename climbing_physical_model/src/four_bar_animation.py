import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 1. LINKAGE GEOMETRY (Adjust dimensions based on your specific problem)
# =============================================================================
r1 = 2.5  # Ground link (Link 1)
r2 = 1.0  # Crank / Input link (Link 2)
r3 = 2.5  # Coupler link (Link 3)
r4 = 2.0  # Rocker / Output link (Link 4)

# Check Grashof condition: Shortest + Longest <= Sum of other two
lengths = sorted([r1, r2, r3, r4])
if lengths[0] + lengths[-1] <= lengths[1] + lengths[2]:
    print("Grashof Linkage: At least one link can fully rotate.")
else:
    print("Non-Grashof Linkage: Links will only oscillate.")

# =============================================================================
# 2. ANALYTICAL POSITION ANALYSIS (Vector Loop Equations)
# =============================================================================
def solve_four_bar(theta2, r1, r2, r3, r4, mode=-1):
    """
    Solves for theta3 and theta4 given the input crank angle theta2.
    mode = 1 (Open / Elbow Up configuration)
    mode = -1 (Crossed / Elbow Down configuration)
    """
    # Vector from joint O2 to joint O4
    # Assuming O2 is at (0,0) and O4 is at (r1, 0)
    xA = r2 * np.cos(theta2)
    yA = r2 * np.sin(theta2)
    
    # Distance from A to O4
    d_sq = (r1 - xA)**2 + yA**2
    d = np.sqrt(d_sq)
    
    # Check if the triangle can close
    if d > (r3 + r4) or d < abs(r3 - r4):
        return None, None, None, None

    # Angle of vector A -> O4
    alpha = np.arctan2(-yA, r1 - xA)
    
    # Law of cosines to find internal angles
    cos_beta = (r3**2 + d_sq - r4**2) / (2 * r3 * d)
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    
    # Calculate theta3 and theta4 based on the assembly configuration mode
    theta3 = alpha + mode * beta
    
    # Coordinates of joint B
    xB = xA + r3 * np.cos(theta3)
    yB = yA + r3 * np.sin(theta3)
    
    theta4 = np.arctan2(yB, xB - r1)
    
    return xA, yA, xB, yB

# =============================================================================
# 3. ANIMATION SETUP VIA MATPLOTLIB
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-r2 - 1, r1 + r4 + 1)
ax.set_ylim(-max(r2, r3, r4) - 1, max(r2, r3, r4) + 1)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("Four-Bar Linkage Animation (Marghitu Kinematics)")

# Visual elements representing the bars
line, = ax.plot([], [], 'o-', lw=3, color='#1f77b4', markersize=8, label="Mechanism Links")
coupler_point, = ax.plot([], [], 'r-', alpha=0.4, label="Coupler Path")

# Stationary Ground points O2 and O4
ax.plot([0, r1], [0, 0], 'ks', markersize=10, label="Ground Joints (O2, O4)")
ax.legend(loc="upper right")

# Track the historical path of the coupler joint (B)
path_x, path_y = [], []

def init():
    line.set_data([], [])
    coupler_point.set_data([], [])
    return line, coupler_point

def animate(frame):
    # Drive the crank angle theta2 smoothly from 0 to 2*pi
    theta2 = np.radians(frame)
    
    # Call the analytical solver (mode=1 for open configuration)
    result = solve_four_bar(theta2, r1, r2, r3, r4, mode=1)
    
    if result[0] is not None:
        xA, yA, xB, yB = result
        
        # Link coordinates sequence: O2 -> A -> B -> O4
        x_coords = [0, xA, xB, r1]
        y_coords = [0, yA, yB, 0]
        
        line.set_data(x_coords, y_coords)
        
        # Keep record of Coupler joint path
        path_x.append(xB)
        path_y.append(yB)
        # Limit history to avoid memory explosion
        if len(path_x) > 360:
            path_x.pop(0)
            path_y.pop(0)
        coupler_point.set_data(path_x, path_y)
        
    return line, coupler_point

# Create the live loop (360 frames for a full rotation)
ani = animation.FuncAnimation(
    fig, animate, frames=np.arange(0, 360, 2),
    init_func=init, blit=True, interval=20
)

plt.show()
