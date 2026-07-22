from .stewart_dynamic import stewart_dynamic_model


r_b = 1.3  # Radius of base
phi_b = 50  # Angle between base joints
r_p = 1.7  # Radius of platform
phi_p = 80  # Angle between platform joints

# Initialization: create Stewart Platform instance

platform = stewart_dynamic_model(r_b, phi_b, r_p, phi_p)