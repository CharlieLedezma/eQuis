import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from __main__ import platform

#WORKSPACE ANALYSIS
#Use getIndexWorkspacePosition to calculate an index over a range of positions in the workspace.
#Use getIndexWorkspaceOrientation to calculate an index over a range of orientations for a given position.

def do_work(vault_instance):
    #platform = vault_instance
    return vault_instance

workspace_limits = [-0.5, 0.5, -0.5, 0.5, 0.1, 0.6]
RPY = [0, 0, 0]  # Fixed orientation (roll, pitch, yaw)
N = 10  # Number of points in each dimension
choice = 4  # Choice of index calculation (1: Singular Value Index, etc.)
            # self.options = {
            #     1: self.getSingularValueIndex, # measures drive capability of the platform, finds max q_dot under unitary x_dot
            #     2: self.getManipulabilityIndex,# Measures manipulability of manipulator, can be used to optimize it's configuration
            #     3: self.getConditionNumber,# Measures closeness to isotropic configuration [1,+ inf)
            #     4: self.getLocalConditionIndex,# Measures closeness to isotropic configuration (0,1]
            #     5: self.getLDI # Local design index for Force transmittability (actuator design)
            # }

platform = do_work()
workspace_indices_position = platform.getIndexWorkspacePosition(workspace_limits, RPY, N, choice)
print("Workspace Indices (Position):", workspace_indices_position)

# Define orientation limits [roll_min, roll_max, pitch_min, pitch_max, yaw_min, yaw_max]
orientation_limits = [-10, 10, -10, 10, -10, 10]
position = [0, 0, 0.4]  # Fixed position

workspace_indices_orientation = platform.getIndexWorkspaceOrientation(position, orientation_limits, N, choice)
print("Workspace Indices (Orientation):", workspace_indices_orientation)


# @title Plotly
values = workspace_indices_position[:, 3]
X = workspace_indices_position[:, 0]
Y = workspace_indices_position[:, 1]
Z = workspace_indices_position[:, 2]

x_min, x_max = np.min(X), np.max(X)
y_min, y_max = np.min(Y), np.max(Y)
z_min, z_max = np.min(Z), np.max(Z)

isomin_val, isomax_val = np.min(values), np.max(values)

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    opacity=0.2,  # needs to be small to see through all surfaces
    surface_count=20,  # needs to be a large number for good volume rendering
    isomin=isomin_val,
    isomax=isomax_val,
    caps=dict(x_show=False, y_show=False, z_show=False, x_fill=1),  # with caps (default mode) (Uncomment to see all values)
))

fig.update_layout(
    title='Worspace Position',
    scene=dict(
        xaxis=dict(nticks=N, range=[x_min, x_max]),
        yaxis=dict(nticks=N, range=[y_min, y_max]),
        zaxis=dict(nticks=N, range=[z_min, z_max]),
    ),
    width=700,
    margin=dict(r=0, l=0, b=0, t=40)
)

fig.show()