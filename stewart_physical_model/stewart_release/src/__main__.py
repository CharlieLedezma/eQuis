import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from scipy.integrate import solve_ivp
from .config import *
from . import workspace_analysis
from .stewart_dynamic import stewart_dynamic_model



def kinematic():
 #Inverse Kinematics(IK)
 #Use the getIK method to compute the leg lengths for a given pose (position and orientation).
     #pose = [0.2, 0, 0.6, 10, 20, 0]  # [x, y, z, roll, pitch, yaw]
    pose = [0, 0, 0, 0, 30, 0]  # [x, y, z, roll, pitch, yaw]
    leg_lengths = platform.getIK(pose)
    platform.plot()
    print("Leg Lengths:", leg_lengths)

# Forward Kinematic (FK)
    #starting_pose = [0, 0, 0.2, 0, 0, 0]  # Initial guess for the pose
    #lengths_desired = np.linalg.norm(leg_lengths,axis=1)  # Use the lengths obtained from IK
    #plot=True
    #estimated_pose = platform.getFK(starting_pose, lengths_desired, plot)


# KINEMATIC AND FORCE ANALYSIS
# Get Singular Value Index
# measures drive capability of the platform, finds max q_dot under unitary x_dot
    singular_value_index = platform.getSingularValueIndex()
    print("Singular Value Index:", singular_value_index)

    # Get Manipulability Index
    # Measures manipulability of manipulator, can be used to optimize it's configuration
    manipulability_index = platform.getManipulabilityIndex()
    print("Manipulability Index:", manipulability_index)

    # Get Condition Number
    # Measures closeness to isotropic configuration [1,+ inf)
    condition_number = platform.getConditionNumber()
    print("Condition Number:", condition_number)

    # Get Local Condition Index
    # Measures closeness to isotropic configuration (0,1]
    local_condition_index = platform.getLocalConditionIndex()
    print("Local Condition Index:", local_condition_index)

    # Calculate Platform Forces given Actuator Forces
    F_actuators = [10, 10, 10, 10, 10, 10]  # Example actuator forces
    F_platform = platform.getPlatformForces(F_actuators)
    print("Platform Forces:", F_platform)

    # Calculate Actuator Forces given Platform Forces
    F_platform = [10, 10, 10, 10, 10, 10]  # Example platform forces
    F_actuators = platform.getActuatorForces(F_platform)
    print("Actuator Forces:", F_actuators)

    # Get Force Ellipsoid
    force_ellipsoid = platform.getForceEllipsoid()
    print("Force Ellipsoid:", force_ellipsoid)

    # Get Local Design Index (LDI)
    # Local design index for Force transmittability (actuator design)
    ldi = platform.getLDI()
    print("Local Design Index:", ldi)

    #Print Jacobian
    jac = platform.getJacobian()
    print("Jacobian", jac)
    return platform
 
if __name__ == "__main__":
    kinematic()
    workspace_analysis.ws_analysis()
    
    