import numpy as np
#from scipy.integrate import solve_ivp
#from stewart_release import stewart_dynamic_model
from stewart_dynamic import stewart_dynamic_model

def main():
#if __name__ == "__main__":
    # Create an instance of stewart_dynamic_model

# Define parameters
    r_b = 1.3  # Radius of base
    phi_b = 50  # Angle between base joints
    r_p = 1.3  # Radius of platform
    phi_p = 80  # Angle between platform joints

# Create Stewart Platform instance
    platform = stewart_dynamic_model(r_b, phi_b, r_p, phi_p)
    pose = [0.2, 0, 0.6, 10, 20, 0]  # [x, y, z, roll, pitch, yaw]
    leg_lengths = platform.getIK(pose)
    platform.plot()
    print("Leg Lengths:", leg_lengths)

# Forward Kinematic
    #starting_pose = [0, 0, 0.2, 0, 0, 0]  # Initial guess for the pose
    #lengths_desired = np.linalg.norm(leg_lengths,axis=1)  # Use the lengths obtained from IK
    #plot=True
    #estimated_pose = platform.getFK(starting_pose, lengths_desired, plot)
# Kinematic and Force Analysis
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

if __name__ == "__main__":
    main()