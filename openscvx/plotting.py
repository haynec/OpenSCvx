from plotly.subplots import make_subplots
import random
import plotly.graph_objects as go
import numpy as np
import pickle

from openscvx.utils import qdcm
from openscvx.config import Config

def full_subject_traj_time(results, params):
    x_full = results['state']
    x_nodes = results['scp_trajs'][-1]
    t_nodes = x_nodes[:,params.veh.t_inds]
    t_full = results['t_full']
    subs_traj = []
    subs_traj_node = []
    subs_traj_sen = []
    subs_traj_sen_node = []
    
    if hasattr(params.veh, 'get_kp_pose'):
        subs_traj.append(params.veh.get_kp_pose(t_full))
        subs_traj_node.append(params.veh.get_kp_pose(t_nodes))
    elif hasattr(params.veh, 'init_poses'):
        for pose in params.veh.init_poses:
            # repeat the pose for all time steps
            pose_full = np.repeat(pose[:,np.newaxis], x_full.shape[0], axis=1).T
            subs_traj.append(pose_full)
            
            pose_node = np.repeat(pose[:,np.newaxis], x_nodes.shape[0], axis=1).T
            subs_traj_node.append(pose_node)
    else:
        raise ValueError("No valid method to get keypoint poses.")

    R_sb = params.veh.R_sb
    for sub_traj in subs_traj:
        sub_traj_sen = []
        for i in range(x_full.shape[0]):
            sub_pose = sub_traj[i]
            sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
        subs_traj_sen.append(np.array(sub_traj_sen).squeeze())

    for sub_traj_node in subs_traj_node:
        sub_traj_sen_node = []
        for i in range(x_nodes.shape[0]):
            sub_pose = sub_traj_node[i]
            sub_traj_sen_node.append(R_sb @ qdcm(x_nodes[i, 6:10]).T @ (sub_pose - x_nodes[i, 0:3]))
        subs_traj_sen_node.append(np.array(sub_traj_sen_node).squeeze())
    return subs_traj, subs_traj_sen, subs_traj_node, subs_traj_sen_node

def save_gate_parameters(gates, params: Config):
    gate_centers = []
    gate_vertices = []
    for gate in gates:
        gate_centers.append(gate.center)
        gate_vertices.append(gate.vertices)
    gate_params = dict(
        gate_centers = gate_centers,
        gate_vertices = gate_vertices
    )

    # Use pickle to save the gate parameters
    with open('results/gate_params.pickle', 'wb') as f:
        pickle.dump(gate_params, f)

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }

def plot_constraint_violation(result, params: Config):
    fig = make_subplots(rows=2, cols=3, subplot_titles=(r'$\text{Obstacle Violation}$', r'$\text{Sub VP Violation}$', r'$\text{Sub Min Violation}$', r'$\text{Sub Max Violation}$', r'$\text{Sub Direc Violation}$', r'$\text{State Bound Violation}$', r'$\text{Total Violation}$'))
    fig.update_layout(template='plotly_dark', title=r'$\text{Constraint Violation}$')

    if "obs_vio" in result:
        obs_vio = result["obs_vio"]
        for i in range(obs_vio.shape[0]):
            color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
            fig.add_trace(go.Scatter(y=obs_vio[i], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=1, col=1)
        i = 0
    else:
        print("'obs_vio' not found in result dictionary.")

    # Make names of each state in the state vector
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz', 'ctcs']

    if "sub_vp_vio" in result and "sub_min_vio" in result and "sub_max_vio" in result:
        sub_vp_vio = result["sub_vp_vio"]
        sub_min_vio = result["sub_min_vio"]
        sub_max_vio = result["sub_max_vio"]
        for i in range(sub_vp_vio.shape[0]):
            color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
            fig.add_trace(go.Scatter(y=sub_vp_vio[i], mode='lines', showlegend=True, name = 'LoS ' + str(i) + ' Error', line=dict(color=color, width = 2)), row=1, col=2)
            if params.vp.tracking:
                fig.add_trace(go.Scatter(y=sub_min_vio[i], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=1, col=3)
                fig.add_trace(go.Scatter(y=sub_max_vio[i], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(y=[], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=1, col=3)
                fig.add_trace(go.Scatter(y=[], mode='lines', showlegend=False, line=dict(color=color, width = 2)), row=2, col=1)
        i = 0
    else:
        print("'sub_vp_vio', 'sub_min_vio', or 'sub_max_vio' not found in result dictionary.")
    
    if "sub_direc_vio" in result:
        sub_direc_vio = result["sub_direc_vio"]
        # fig.add_trace(go.Scatter(y=sub_direc_vio, mode='lines', showlegend=False, line=dict(color='red', width = 2)), row=2, col=2)
    else:
        print("'sub_direc_vio' not found in result dictionary.")

    if "state_bound_vio" in result:
        state_bound_vio = result["state_bound_vio"]
        for i in range(state_bound_vio.shape[0]):
            color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
            fig.add_trace(go.Scatter(y=state_bound_vio[:,i], mode='lines', showlegend=True, name = state_names[i] + ' Error', line=dict(color=color, width = 2)), row=2, col=3)
    else:
        print("'state_bound_vio' not found in result dictionary.")

    fig.show()

def plot_initial_guess(result, params: Config):
    x_positions = result["x"][0:3]
    x_attitude = result["x"][6:10]
    subs_positions = result["sub_positions"]

    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2)))

    # Plot the position of the drone
    fig.add_trace(go.Scatter3d(x=x_positions[0], y=x_positions[1], z=x_positions[2], mode='lines+markers', line=dict(color='green', width = 5)))

    # Plot the attitude of the drone
    # Draw drone attitudes as axes
    step = 1
    indices = np.array(list(range(x_positions.shape[1])))
    for i in range(0, len(indices), step):
        att = x_attitude[:, indices[i]]

        # Convert quaternion to rotation matrix
        rotation_matrix = qdcm(att)

        # Extract axes from rotation matrix
        axes = 2 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotated_axes = np.dot(rotation_matrix, axes).T

        colors = ['#FF0000', '#00FF00', '#0000FF']

        for k in range(3):
            axis = rotated_axes[k]
            color = colors[k]

            fig.add_trace(go.Scatter3d(
                x=[x_positions[0, indices[i]], x_positions[0, indices[i]] + axis[0]],
                y=[x_positions[1, indices[i]], x_positions[1, indices[i]] + axis[1]],
                z=[x_positions[2, indices[i]], x_positions[2, indices[i]] + axis[2]],
                mode='lines+text',
                line=dict(color=color, width=4),
                showlegend=False
            ))
    
    fig.update_layout(template='plotly_dark')
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Plot the keypoint
    for sub_positions in subs_positions:
        fig.add_trace(go.Scatter3d(x=sub_positions[:,0], y=sub_positions[:,1], z=sub_positions[:,2], mode='lines+markers', line=dict(color='red', width = 5), name='Subject'))
    fig.show()

def plot_camera_view(result: dict, params: Config) -> None:
    title = r'$\text{Camera View}$'
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result["x_full"], params, False)
    fig = go.Figure()

    # Create a cone plot
    A = np.diag([1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)])  # Conic Matrix

    # Meshgrid
    if params.vp.tracking:
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        z = np.linspace(-10, 10, 100)
    else:
        x = np.linspace(-80, 80, 100)
        y = np.linspace(-80, 80, 100)
        z = np.linspace(-80, 80, 100)
 
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = []
    for x_val in x:
        for y_val in y:
            if params.vp.norm == 'inf':
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = np.inf))
            else:
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = params.vp.norm))
    z = np.array(z)

    # Extract the points from the meshgrid
    X = X.flatten()
    Y = Y.flatten()
    Z = z.flatten()
    
    # Normalize the coordinates by the Z value
    X = X / Z
    Y = Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X = X[order]
    Y = Y[order]

    # Repeat the first point to close the cone
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', line=dict(color='red', width=5), name = r'$\text{Camera Frame}$'))

    sub_idx = 0
    for sub_traj in sub_positions_sen:
        color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
        sub_traj = np.array(sub_traj)
        sub_traj[:,0] = sub_traj[:,0] / sub_traj[:,2]
        sub_traj[:,1] = sub_traj[:,1] / sub_traj[:,2]
        fig.add_trace(go.Scatter(x=sub_traj[:, 0], y=sub_traj[:, 1], mode='lines',line=dict(color=color, width=3), name = r'$\text{Subject }' + str(sub_idx) + '$'))
        
        sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])
        sub_traj_nodal[:,0] = sub_traj_nodal[:,0] / sub_traj_nodal[:,2]
        sub_traj_nodal[:,1] = sub_traj_nodal[:,1] / sub_traj_nodal[:,2]
        fig.add_trace(go.Scatter(x=sub_traj_nodal[:, 0], y=sub_traj_nodal[:, 1], mode='markers',marker=dict(color=color, size=20), name = r'$\text{Subject }' + str(sub_idx) + r'\text{ Node}$'))
        sub_idx += 1
    
    # Center the title for the plot
    fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template='simple_white')

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # fig.update_yaxes(scaleanchor="x", scaleratio=1,)
    fig.update_layout(height=600)

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.0, 1.0])
    fig.update_yaxes(range=[-1.0, 1.0])
    # Set aspect ratio to be equal
    fig.update_layout(autosize=False, width=800, height=800)

    # Save figure as svg
    fig.write_image("figures/camera_view.svg")

    fig.show()

def plot_camera_animation(result: dict, params, path="") -> None:
    title = r'$\text{Camera Animation}$'
    _, subs_positions_sen, _, subs_positions_sen_node = full_subject_traj_time(result, params)
    fig = go.Figure()

    # Add blank plots for the subjects
    for _ in range(50):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='blue', width=2)))

    # Create a cone plot
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag([1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])])  # Conic Matrix
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")

    # Meshgrid
    range_limit = 10 if hasattr(params.veh, 'get_kp_pose') else 80
    x = np.linspace(-range_limit, range_limit, 50)
    y = np.linspace(-range_limit, range_limit, 50)
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    if "norm_type" in result:
        z = np.array([np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=(np.inf if result["norm_type"] == 'inf' else result["norm_type"])) for x_val in x for y_val in y])
    else:
        raise ValueError("`norm_type` not found in result dictionary.")

    # Extract the points from the meshgrid
    X, Y, Z = X.flatten(), Y.flatten(), z.flatten()

    # Normalize the coordinates by the Z value
    X, Y = X / Z, Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]

    # Repeat the first point to close the cone
    X, Y = np.append(X, X[0]), np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', line=dict(color='red', width=5), name=r'$\text{Camera Frame}$', showlegend=False))

    # Choose a random color for each subject
    colors = [f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})' for _ in subs_positions_sen]

    frames = []
    # Animate the subjects along their trajectories
    for i in range(0, len(subs_positions_sen[0]), 2):
        frame_data = []
        for sub_idx, sub_traj in enumerate(subs_positions_sen):
            color = colors[sub_idx]
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(subs_positions_sen_node[sub_idx])
            sub_traj[:, 0] /= sub_traj[:, 2]
            sub_traj[:, 1] /= sub_traj[:, 2]
            frame_data.append(go.Scatter(x=sub_traj[:i+1, 0], y=sub_traj[:i+1, 1], mode='lines', line=dict(color=color, width=3), showlegend=False))

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]
            sub_node_plot[:, 0] /= sub_node_plot[:, 2]
            sub_node_plot[:, 1] /= sub_node_plot[:, 2]
            frame_data.append(go.Scatter(x=sub_node_plot[:, 0], y=sub_node_plot[:, 1], mode='markers', marker=dict(color=color, size=10), showlegend=False))

        frames.append(go.Frame(name=str(i), data=frame_data))

    fig.frames = frames

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.15,
            "steps": [
                {
                    "args": [[f.name], frame_args(500)],  # Use the frame name as the argument
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.15,
                                }
                            ],
                            sliders=sliders
                        )

    fig.update_layout(sliders=sliders)
    
    # Center the title for the plot
    fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template='plotly_dark')
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Remove center line
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # Remove the axis numbers
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Remove ticks enrtirely
    fig.update_xaxes(ticks="outside", tickwidth=0, tickcolor='black')
    fig.update_yaxes(ticks="outside", tickwidth=0, tickcolor='black')
    

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    # Move Title down
    fig.update_layout(title_y=0.9)

    # Set aspect ratio to be equal
    # fig.update_layout(autosize=False, width=650, height=650)
    # Remove marigns
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # # Make the background transparent
    # fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    # # Make the axis backgrounds transparent
    # fig.update_layout(scene=dict(
    #     xaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey')
    # ))
    # # Remove the plot background
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # # Make ticks themselves transparent
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))

    # # Remove the paper background
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    fig.show()  

def plot_camera_polytope_animation(result: dict, params, path="") -> None:
    title = r'$\text{Camera Animation}$'
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result["x_full"], params, False)
    fig = go.Figure()

    # Add blank plots for the subjects
    for _ in range(500):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='blue', width=2)))

    # Create a cone plot
    A = np.diag([1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)])  # Conic Matrix

    # Meshgrid
    range_limit = 10 if params.vp.tracking else 80
    x = np.linspace(-range_limit, range_limit, 50)
    y = np.linspace(-range_limit, range_limit, 50)
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = np.array([np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord=(np.inf if params.vp.norm == 'inf' else params.vp.norm)) for x_val in x for y_val in y])

    # Extract the points from the meshgrid
    X, Y, Z = X.flatten(), Y.flatten(), z.flatten()

    # Normalize the coordinates by the Z value
    X, Y = X / Z, Y / Z

    # Order the points so they are connected in radial order about the origin
    order = np.argsort(np.arctan2(Y, X))
    X, Y = X[order], Y[order]

    # Repeat the first point to close the cone
    X, Y = np.append(X, X[0]), np.append(Y, Y[0])

    # Plot the points on a red scatter plot
    fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', line=dict(color='red', width=5), name=r'$\text{Camera Frame}$', showlegend=False))

    # Choose a random color for each subject
    colors = [f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})' for _ in sub_positions_sen]

    frames = []
    # Animate the subjects along their trajectories
    for i in range(0, len(sub_positions_sen[0]), 2):
        frame_data = []
        for sub_idx, sub_traj in enumerate(sub_positions_sen):
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])
            sub_traj[:, 0] /= sub_traj[:, 2]
            sub_traj[:, 1] /= sub_traj[:, 2]
            frame_data.append(go.Scatter(x=sub_traj[:i+1, 0], y=sub_traj[:i+1, 1], mode='lines', line=dict(color='darkblue', width=3), showlegend=False))

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0] / sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]
            sub_node_plot[:, 0] /= sub_node_plot[:, 2]
            sub_node_plot[:, 1] /= sub_node_plot[:, 2]
            frame_data.append(go.Scatter(x=sub_node_plot[:, 0], y=sub_node_plot[:, 1], mode='markers', marker=dict(color='darkblue', size=10), showlegend=False))

        # Connect each of the polytope vertices or subjects of polytope to eachother at each time i, don't use sub_traj_nodal
        
        # Connect 0 to 16, 8, 12
        frame_data.append(go.Scatter(x=[sub_positions_sen[0][i][0]/sub_positions_sen[0][i][2], sub_positions_sen[16][i][0]/sub_positions_sen[16][i][2]], y=[sub_positions_sen[0][i][1]/sub_positions_sen[0][i][2], sub_positions_sen[16][i][1]/sub_positions_sen[16][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[0][i][0]/sub_positions_sen[0][i][2], sub_positions_sen[8][i][0]/sub_positions_sen[8][i][2]], y=[sub_positions_sen[0][i][1]/sub_positions_sen[0][i][2], sub_positions_sen[8][i][1]/sub_positions_sen[8][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[0][i][0]/sub_positions_sen[0][i][2], sub_positions_sen[12][i][0]/sub_positions_sen[12][i][2]], y=[sub_positions_sen[0][i][1]/sub_positions_sen[0][i][2], sub_positions_sen[12][i][1]/sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 1 to 17, 9, 12
        frame_data.append(go.Scatter(x=[sub_positions_sen[1][i][0]/sub_positions_sen[1][i][2], sub_positions_sen[17][i][0]/sub_positions_sen[17][i][2]], y=[sub_positions_sen[1][i][1]/sub_positions_sen[1][i][2], sub_positions_sen[17][i][1]/sub_positions_sen[17][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[1][i][0]/sub_positions_sen[1][i][2], sub_positions_sen[9][i][0]/sub_positions_sen[9][i][2]], y=[sub_positions_sen[1][i][1]/sub_positions_sen[1][i][2], sub_positions_sen[9][i][1]/sub_positions_sen[9][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[1][i][0]/sub_positions_sen[1][i][2], sub_positions_sen[12][i][0]/sub_positions_sen[12][i][2]], y=[sub_positions_sen[1][i][1]/sub_positions_sen[1][i][2], sub_positions_sen[12][i][1]/sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 2 to 16, 13, 10
        frame_data.append(go.Scatter(x=[sub_positions_sen[2][i][0]/sub_positions_sen[2][i][2], sub_positions_sen[16][i][0]/sub_positions_sen[16][i][2]], y=[sub_positions_sen[2][i][1]/sub_positions_sen[2][i][2], sub_positions_sen[16][i][1]/sub_positions_sen[16][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[2][i][0]/sub_positions_sen[2][i][2], sub_positions_sen[13][i][0]/sub_positions_sen[13][i][2]], y=[sub_positions_sen[2][i][1]/sub_positions_sen[2][i][2], sub_positions_sen[13][i][1]/sub_positions_sen[13][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[2][i][0]/sub_positions_sen[2][i][2], sub_positions_sen[10][i][0]/sub_positions_sen[10][i][2]], y=[sub_positions_sen[2][i][1]/sub_positions_sen[2][i][2], sub_positions_sen[10][i][1]/sub_positions_sen[10][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 3 to 17, 11, 13
        frame_data.append(go.Scatter(x=[sub_positions_sen[3][i][0]/sub_positions_sen[3][i][2], sub_positions_sen[17][i][0]/sub_positions_sen[17][i][2]], y=[sub_positions_sen[3][i][1]/sub_positions_sen[3][i][2], sub_positions_sen[17][i][1]/sub_positions_sen[17][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[3][i][0]/sub_positions_sen[3][i][2], sub_positions_sen[11][i][0]/sub_positions_sen[11][i][2]], y=[sub_positions_sen[3][i][1]/sub_positions_sen[3][i][2], sub_positions_sen[11][i][1]/sub_positions_sen[11][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[3][i][0]/sub_positions_sen[3][i][2], sub_positions_sen[13][i][0]/sub_positions_sen[13][i][2]], y=[sub_positions_sen[3][i][1]/sub_positions_sen[3][i][2], sub_positions_sen[13][i][1]/sub_positions_sen[13][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 4 to 18, 14, 8
        frame_data.append(go.Scatter(x=[sub_positions_sen[4][i][0]/sub_positions_sen[4][i][2], sub_positions_sen[18][i][0]/sub_positions_sen[18][i][2]], y=[sub_positions_sen[4][i][1]/sub_positions_sen[4][i][2], sub_positions_sen[18][i][1]/sub_positions_sen[18][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[4][i][0]/sub_positions_sen[4][i][2], sub_positions_sen[14][i][0]/sub_positions_sen[14][i][2]], y=[sub_positions_sen[4][i][1]/sub_positions_sen[4][i][2], sub_positions_sen[14][i][1]/sub_positions_sen[14][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[4][i][0]/sub_positions_sen[4][i][2], sub_positions_sen[8][i][0]/sub_positions_sen[8][i][2]], y=[sub_positions_sen[4][i][1]/sub_positions_sen[4][i][2], sub_positions_sen[8][i][1]/sub_positions_sen[8][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        
        # Connect 5 to 19, 9, 14
        frame_data.append(go.Scatter(x=[sub_positions_sen[5][i][0]/sub_positions_sen[5][i][2], sub_positions_sen[19][i][0]/sub_positions_sen[19][i][2]], y=[sub_positions_sen[5][i][1]/sub_positions_sen[5][i][2], sub_positions_sen[19][i][1]/sub_positions_sen[19][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[5][i][0]/sub_positions_sen[5][i][2], sub_positions_sen[9][i][0]/sub_positions_sen[9][i][2]], y=[sub_positions_sen[5][i][1]/sub_positions_sen[5][i][2], sub_positions_sen[9][i][1]/sub_positions_sen[9][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[5][i][0]/sub_positions_sen[5][i][2], sub_positions_sen[14][i][0]/sub_positions_sen[14][i][2]], y=[sub_positions_sen[5][i][1]/sub_positions_sen[5][i][2], sub_positions_sen[14][i][1]/sub_positions_sen[14][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 6 to 18, 15, 10
        frame_data.append(go.Scatter(x=[sub_positions_sen[6][i][0]/sub_positions_sen[6][i][2], sub_positions_sen[18][i][0]/sub_positions_sen[18][i][2]], y=[sub_positions_sen[6][i][1]/sub_positions_sen[6][i][2], sub_positions_sen[18][i][1]/sub_positions_sen[18][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[6][i][0]/sub_positions_sen[6][i][2], sub_positions_sen[15][i][0]/sub_positions_sen[15][i][2]], y=[sub_positions_sen[6][i][1]/sub_positions_sen[6][i][2], sub_positions_sen[15][i][1]/sub_positions_sen[15][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[6][i][0]/sub_positions_sen[6][i][2], sub_positions_sen[10][i][0]/sub_positions_sen[10][i][2]], y=[sub_positions_sen[6][i][1]/sub_positions_sen[6][i][2], sub_positions_sen[10][i][1]/sub_positions_sen[10][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 7 to 19, 11, 15
        frame_data.append(go.Scatter(x=[sub_positions_sen[7][i][0]/sub_positions_sen[7][i][2], sub_positions_sen[19][i][0]/sub_positions_sen[19][i][2]], y=[sub_positions_sen[7][i][1]/sub_positions_sen[7][i][2], sub_positions_sen[19][i][1]/sub_positions_sen[19][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[7][i][0]/sub_positions_sen[7][i][2], sub_positions_sen[11][i][0]/sub_positions_sen[11][i][2]], y=[sub_positions_sen[7][i][1]/sub_positions_sen[7][i][2], sub_positions_sen[11][i][1]/sub_positions_sen[11][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[7][i][0]/sub_positions_sen[7][i][2], sub_positions_sen[15][i][0]/sub_positions_sen[15][i][2]], y=[sub_positions_sen[7][i][1]/sub_positions_sen[7][i][2], sub_positions_sen[15][i][1]/sub_positions_sen[15][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 8 to 0, 4, 12 -> 10 BAD
        frame_data.append(go.Scatter(x=[sub_positions_sen[8][i][0]/sub_positions_sen[8][i][2], sub_positions_sen[0][i][0]/sub_positions_sen[0][i][2]], y=[sub_positions_sen[8][i][1]/sub_positions_sen[8][i][2], sub_positions_sen[0][i][1]/sub_positions_sen[0][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[8][i][0]/sub_positions_sen[8][i][2], sub_positions_sen[4][i][0]/sub_positions_sen[4][i][2]], y=[sub_positions_sen[8][i][1]/sub_positions_sen[8][i][2], sub_positions_sen[4][i][1]/sub_positions_sen[4][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[8][i][0]/sub_positions_sen[8][i][2], sub_positions_sen[10][i][0]/sub_positions_sen[10][i][2]], y=[sub_positions_sen[8][i][1]/sub_positions_sen[8][i][2], sub_positions_sen[10][i][1]/sub_positions_sen[10][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 9 to 1, 5, 13 -> 11 BAD
        frame_data.append(go.Scatter(x=[sub_positions_sen[9][i][0]/sub_positions_sen[9][i][2], sub_positions_sen[1][i][0]/sub_positions_sen[1][i][2]], y=[sub_positions_sen[9][i][1]/sub_positions_sen[9][i][2], sub_positions_sen[1][i][1]/sub_positions_sen[1][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[9][i][0]/sub_positions_sen[9][i][2], sub_positions_sen[5][i][0]/sub_positions_sen[5][i][2]], y=[sub_positions_sen[9][i][1]/sub_positions_sen[9][i][2], sub_positions_sen[5][i][1]/sub_positions_sen[5][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[9][i][0]/sub_positions_sen[9][i][2], sub_positions_sen[11][i][0]/sub_positions_sen[11][i][2]], y=[sub_positions_sen[9][i][1]/sub_positions_sen[9][i][2], sub_positions_sen[11][i][1]/sub_positions_sen[11][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 10 to 8, 2, 6
        frame_data.append(go.Scatter(x=[sub_positions_sen[10][i][0]/sub_positions_sen[10][i][2], sub_positions_sen[8][i][0]/sub_positions_sen[8][i][2]], y=[sub_positions_sen[10][i][1]/sub_positions_sen[10][i][2], sub_positions_sen[8][i][1]/sub_positions_sen[8][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[10][i][0]/sub_positions_sen[10][i][2], sub_positions_sen[2][i][0]/sub_positions_sen[2][i][2]], y=[sub_positions_sen[10][i][1]/sub_positions_sen[10][i][2], sub_positions_sen[2][i][1]/sub_positions_sen[2][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[10][i][0]/sub_positions_sen[10][i][2], sub_positions_sen[6][i][0]/sub_positions_sen[6][i][2]], y=[sub_positions_sen[10][i][1]/sub_positions_sen[10][i][2], sub_positions_sen[6][i][1]/sub_positions_sen[6][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 11 to 3, 7, 15 -> 9 BAD
        frame_data.append(go.Scatter(x=[sub_positions_sen[11][i][0]/sub_positions_sen[11][i][2], sub_positions_sen[3][i][0]/sub_positions_sen[3][i][2]], y=[sub_positions_sen[11][i][1]/sub_positions_sen[11][i][2], sub_positions_sen[3][i][1]/sub_positions_sen[3][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[11][i][0]/sub_positions_sen[11][i][2], sub_positions_sen[7][i][0]/sub_positions_sen[7][i][2]], y=[sub_positions_sen[11][i][1]/sub_positions_sen[11][i][2], sub_positions_sen[7][i][1]/sub_positions_sen[7][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[11][i][0]/sub_positions_sen[11][i][2], sub_positions_sen[9][i][0]/sub_positions_sen[9][i][2]], y=[sub_positions_sen[11][i][1]/sub_positions_sen[11][i][2], sub_positions_sen[9][i][1]/sub_positions_sen[9][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 12 to 0, 1, 8 -> 14 BAD
        frame_data.append(go.Scatter(x=[sub_positions_sen[12][i][0]/sub_positions_sen[12][i][2], sub_positions_sen[0][i][0]/sub_positions_sen[0][i][2]], y=[sub_positions_sen[12][i][1]/sub_positions_sen[12][i][2], sub_positions_sen[0][i][1]/sub_positions_sen[0][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[12][i][0]/sub_positions_sen[12][i][2], sub_positions_sen[1][i][0]/sub_positions_sen[1][i][2]], y=[sub_positions_sen[12][i][1]/sub_positions_sen[12][i][2], sub_positions_sen[1][i][1]/sub_positions_sen[1][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[12][i][0]/sub_positions_sen[12][i][2], sub_positions_sen[14][i][0]/sub_positions_sen[14][i][2]], y=[sub_positions_sen[12][i][1]/sub_positions_sen[12][i][2], sub_positions_sen[14][i][1]/sub_positions_sen[14][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 13 to 2, 3, 9 -> 15 BAD
        frame_data.append(go.Scatter(x=[sub_positions_sen[13][i][0]/sub_positions_sen[13][i][2], sub_positions_sen[2][i][0]/sub_positions_sen[2][i][2]], y=[sub_positions_sen[13][i][1]/sub_positions_sen[13][i][2], sub_positions_sen[2][i][1]/sub_positions_sen[2][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[13][i][0]/sub_positions_sen[13][i][2], sub_positions_sen[3][i][0]/sub_positions_sen[3][i][2]], y=[sub_positions_sen[13][i][1]/sub_positions_sen[13][i][2], sub_positions_sen[3][i][1]/sub_positions_sen[3][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[13][i][0]/sub_positions_sen[13][i][2], sub_positions_sen[15][i][0]/sub_positions_sen[15][i][2]], y=[sub_positions_sen[13][i][1]/sub_positions_sen[13][i][2], sub_positions_sen[15][i][1]/sub_positions_sen[15][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 14 to 4, 5, 10 -> 12 BAD
        frame_data.append(go.Scatter(x=[sub_positions_sen[14][i][0]/sub_positions_sen[14][i][2], sub_positions_sen[4][i][0]/sub_positions_sen[4][i][2]], y=[sub_positions_sen[14][i][1]/sub_positions_sen[14][i][2], sub_positions_sen[4][i][1]/sub_positions_sen[4][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[14][i][0]/sub_positions_sen[14][i][2], sub_positions_sen[5][i][0]/sub_positions_sen[5][i][2]], y=[sub_positions_sen[14][i][1]/sub_positions_sen[14][i][2], sub_positions_sen[5][i][1]/sub_positions_sen[5][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[14][i][0]/sub_positions_sen[14][i][2], sub_positions_sen[12][i][0]/sub_positions_sen[12][i][2]], y=[sub_positions_sen[14][i][1]/sub_positions_sen[14][i][2], sub_positions_sen[12][i][1]/sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 15 to 13, 6, 7 
        frame_data.append(go.Scatter(x=[sub_positions_sen[15][i][0]/sub_positions_sen[15][i][2], sub_positions_sen[13][i][0]/sub_positions_sen[13][i][2]], y=[sub_positions_sen[15][i][1]/sub_positions_sen[15][i][2], sub_positions_sen[13][i][1]/sub_positions_sen[13][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[15][i][0]/sub_positions_sen[15][i][2], sub_positions_sen[6][i][0]/sub_positions_sen[6][i][2]], y=[sub_positions_sen[15][i][1]/sub_positions_sen[15][i][2], sub_positions_sen[6][i][1]/sub_positions_sen[6][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[15][i][0]/sub_positions_sen[15][i][2], sub_positions_sen[7][i][0]/sub_positions_sen[7][i][2]], y=[sub_positions_sen[15][i][1]/sub_positions_sen[15][i][2], sub_positions_sen[7][i][1]/sub_positions_sen[7][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 16 to 0, 2, 17
        frame_data.append(go.Scatter(x=[sub_positions_sen[16][i][0]/sub_positions_sen[16][i][2], sub_positions_sen[0][i][0]/sub_positions_sen[0][i][2]], y=[sub_positions_sen[16][i][1]/sub_positions_sen[16][i][2], sub_positions_sen[0][i][1]/sub_positions_sen[0][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[16][i][0]/sub_positions_sen[16][i][2], sub_positions_sen[2][i][0]/sub_positions_sen[2][i][2]], y=[sub_positions_sen[16][i][1]/sub_positions_sen[16][i][2], sub_positions_sen[2][i][1]/sub_positions_sen[2][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[16][i][0]/sub_positions_sen[16][i][2], sub_positions_sen[17][i][0]/sub_positions_sen[17][i][2]], y=[sub_positions_sen[16][i][1]/sub_positions_sen[16][i][2], sub_positions_sen[17][i][1]/sub_positions_sen[17][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 17 to 1, 3, 16
        frame_data.append(go.Scatter(x=[sub_positions_sen[17][i][0]/sub_positions_sen[17][i][2], sub_positions_sen[1][i][0]/sub_positions_sen[1][i][2]], y=[sub_positions_sen[17][i][1]/sub_positions_sen[17][i][2], sub_positions_sen[1][i][1]/sub_positions_sen[1][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[17][i][0]/sub_positions_sen[17][i][2], sub_positions_sen[3][i][0]/sub_positions_sen[3][i][2]], y=[sub_positions_sen[17][i][1]/sub_positions_sen[17][i][2], sub_positions_sen[3][i][1]/sub_positions_sen[3][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[17][i][0]/sub_positions_sen[17][i][2], sub_positions_sen[16][i][0]/sub_positions_sen[16][i][2]], y=[sub_positions_sen[17][i][1]/sub_positions_sen[17][i][2], sub_positions_sen[16][i][1]/sub_positions_sen[16][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 18 to 4, 6, 19
        frame_data.append(go.Scatter(x=[sub_positions_sen[18][i][0]/sub_positions_sen[18][i][2], sub_positions_sen[4][i][0]/sub_positions_sen[4][i][2]], y=[sub_positions_sen[18][i][1]/sub_positions_sen[18][i][2], sub_positions_sen[4][i][1]/sub_positions_sen[4][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[18][i][0]/sub_positions_sen[18][i][2], sub_positions_sen[6][i][0]/sub_positions_sen[6][i][2]], y=[sub_positions_sen[18][i][1]/sub_positions_sen[18][i][2], sub_positions_sen[6][i][1]/sub_positions_sen[6][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[18][i][0]/sub_positions_sen[18][i][2], sub_positions_sen[19][i][0]/sub_positions_sen[19][i][2]], y=[sub_positions_sen[18][i][1]/sub_positions_sen[18][i][2], sub_positions_sen[19][i][1]/sub_positions_sen[19][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 19 to 5, 7, 18
        frame_data.append(go.Scatter(x=[sub_positions_sen[19][i][0]/sub_positions_sen[19][i][2], sub_positions_sen[5][i][0]/sub_positions_sen[5][i][2]], y=[sub_positions_sen[19][i][1]/sub_positions_sen[19][i][2], sub_positions_sen[5][i][1]/sub_positions_sen[5][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[19][i][0]/sub_positions_sen[19][i][2], sub_positions_sen[7][i][0]/sub_positions_sen[7][i][2]], y=[sub_positions_sen[19][i][1]/sub_positions_sen[19][i][2], sub_positions_sen[7][i][1]/sub_positions_sen[7][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        frame_data.append(go.Scatter(x=[sub_positions_sen[19][i][0]/sub_positions_sen[19][i][2], sub_positions_sen[18][i][0]/sub_positions_sen[18][i][2]], y=[sub_positions_sen[19][i][1]/sub_positions_sen[19][i][2], sub_positions_sen[18][i][1]/sub_positions_sen[18][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        frames.append(go.Frame(name=str(i), data=frame_data))

    fig.frames = frames

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.15,
            "steps": [
                {
                    "args": [[f.name], frame_args(500)],  # Use the frame name as the argument
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.15,
                                }
                            ],
                            sliders=sliders
                        )

    fig.update_layout(sliders=sliders)
    
    # Center the title for the plot
    # fig.update_layout(title=title, title_x=0.5)
    fig.update_layout(template='plotly_dark')
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Remove center line
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)

    # Increase title size
    fig.update_layout(title_font_size=20)

    # Increase legend size
    fig.update_layout(legend_font_size=15)

    # Remove the axis numbers
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Remove ticks enrtirely
    fig.update_xaxes(ticks="outside", tickwidth=0, tickcolor='black')
    fig.update_yaxes(ticks="outside", tickwidth=0, tickcolor='black')
    

    # Set x axis and y axis limits
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    # Move Title down
    fig.update_layout(title_y=0.9)

    # Set aspect ratio to be equal
    # fig.update_layout(autosize=False, width=650, height=650)
    # Remove marigns
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # # Make the background transparent
    # fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    # # Make the axis backgrounds transparent
    # fig.update_layout(scene=dict(
    #     xaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey')
    # ))
    # # Remove the plot background
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # # Make ticks themselves transparent
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))

    # # Remove the paper background
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig.show()  

def plot_conic_view_animation(result: dict, params, path="") -> None:
    title = r'$\text{Conic Constraint}$'
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result["x_full"], params, False)
    fig = go.Figure()
    for i in range(100):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='blue', width = 2)))


    # Create a cone plot
    if "alpha_x" in result and "alpha_y" in result:
        A = np.diag([1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])])  # Conic Matrix
    else:
        raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")

    # Meshgrid
    if hasattr(params.veh, 'get_kp_pose'):
        x = np.linspace(-6, 6, 20)
        y = np.linspace(-6, 6, 20)
        z = np.linspace(-6, 6, 20)
    else:
        x = np.linspace(-80, 80, 20)
        y = np.linspace(-80, 80, 20)
        z = np.linspace(-80, 80, 20)
 
    X, Y = np.meshgrid(x, y)

    if "norm_type" in result:
        # Define the condition for the second order cone
        z = []
        for x_val in x:
            for y_val in y:
                if result["norm_type"] == 'inf':
                    z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = np.inf))
                else:
                    z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = result["norm_type"]))
        z = np.array(z)
    
        fig.add_trace(go.Surface(x=X, y=Y, z=z.reshape(20,20), opacity = 0.25, showscale=False))
        frames = []

        if hasattr(params.veh, 'get_kp_pose'):
            x_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:,0])
            y_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:,0])
        else:
            x_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:,0])
            y_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:,0])

        # Add the projection of the second order cone onto the x-z plane
        z = []
        for x_val in x:
            if result["norm_type"] == 'inf':
                z.append(np.linalg.norm(A @ np.array([x_val, 0]), axis=0, ord = np.inf))
            else:
                z.append(np.linalg.norm(A @ np.array([x_val, 0]), axis=0, ord = result["norm_type"]))
        z = np.array(z)
        fig.add_trace(go.Scatter3d(y=x, x=y_vals, z=z, mode='lines', showlegend=False, line=dict(color='grey', width=3)))

        # Add the projection of the second order cone onto the y-z plane
        z = []
        for y_val in y:
            if result["norm_type"] == 'inf':
                z.append(np.linalg.norm(A @ np.array([0, y_val]), axis=0, ord = np.inf))
            else:
                z.append(np.linalg.norm(A @ np.array([0, y_val]), axis=0, ord = result["norm_type"]))
        z = np.array(z)
        fig.add_trace(go.Scatter3d(y=x_vals, x=y, z=z, mode='lines', showlegend=False, line=dict(color='grey', width=3)))
    else:
        raise ValueError("`norm_type` not found in result dictionary.")

    # Choose a random color for each subject
    colors = []
    for sub_traj in sub_positions_sen:
        color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
        colors.append(color)

    color_background = f'rgb({150}, {150}, {150})'
    sub_node_plot = []
    sub_node_idx = 0
    for i in range(0, len(sub_positions_sen[0]), 4):
        frame = go.Frame(name = str(i))
        data = []
        sub_idx = 0

        for sub_traj in sub_positions_sen:
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])

            if hasattr(params.veh, 'get_kp_pose'):
                x_vals = 12 * np.ones_like(sub_traj[:i+1, 0])
                y_vals = 12 * np.ones_like(sub_traj[:i+1, 0])
            else:
                x_vals = 110 * np.ones_like(sub_traj[:i+1, 0])
                y_vals = 110 * np.ones_like(sub_traj[:i+1, 0])

            data.append(go.Scatter3d(x = sub_traj[:i+1, 0], y = y_vals, z=sub_traj[:i+1, 2], mode='lines', showlegend=False, line=dict(color='grey', width=4)))
            data.append(go.Scatter3d(x = x_vals, y = sub_traj[:i+1, 1], z=sub_traj[:i+1, 2], mode='lines', showlegend=False, line=dict(color='grey', width=4)))

            # Add subject position to data
            # color = f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'
            sub_traj = np.array(sub_traj)
            data.append(go.Scatter3d(x=sub_traj[:i+1, 0], y=sub_traj[:i+1, 1], z=sub_traj[:i+1, 2], mode='lines',line=dict(color=colors[sub_idx], width=3), showlegend=False))

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0]/sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]

            data.append(go.Scatter3d(x=sub_node_plot[:, 0], y=sub_node_plot[:, 1], z=sub_node_plot[:, 2], mode='markers', marker=dict(color=colors[sub_idx], size=5), showlegend=False))

            sub_idx += 1
        
        frame.data = data
        frames.append(frame)
    
    fig.frames = frames

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.32,
            "steps": [
                {
                    "args": [[f.name], frame_args(500)],  # Use the frame name as the argument
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.32,
                                }
                            ],
                            sliders=sliders
                        )

    fig.update_layout(sliders=sliders)

    # Set camera position
    fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=10), center=dict(x=-2, y=0, z=-3), eye=dict(x=-28, y=-22, z=15)))

    # Set axis labels 
    fig.update_layout(scene=dict(xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)'))

    fig.update_layout(template='plotly_dark')
    
    # Make only the grid lines thicker in the template
    fig.update_layout(scene=dict(xaxis=dict(showgrid=True, gridwidth=5),
                                yaxis=dict(showgrid=True, gridwidth=5),
                                zaxis=dict(showgrid=True, gridwidth=5)))


    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=20, y=20, z=20)))
    # fig.update_layout(autosize=False, width=600, height=600)

    # Remove marigns
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # # Make the background transparent
    # fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    # # Make the axis backgrounds transparent
    # fig.update_layout(scene=dict(
    #     xaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey')
    # ))
    # # Remove the plot background
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # # Make ticks themselves transparent
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))

    # # Remove the paper background
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    fig.show()

def plot_conic_view_polytope_animation(result: dict, params, path="") -> None:
    title = r'$\text{Conic Constraint}$'
    sub_positions_sen, _, sub_positions_sen_node = full_subject_traj_time(result["x_full"], params, False)
    fig = go.Figure()
    for i in range(500):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='blue', width = 2)))


    # Create a cone plot
    A = np.diag([1 / np.tan(np.pi / params.vp.alpha_y), 1 / np.tan(np.pi / params.vp.alpha_x)])  # Conic Matrix

    # Meshgrid
    if params.vp.tracking:
        x = np.linspace(-6, 6, 20)
        y = np.linspace(-6, 6, 20)
        z = np.linspace(-6, 6, 20)
    else:
        x = np.linspace(-80, 80, 20)
        y = np.linspace(-80, 80, 20)
        z = np.linspace(-80, 80, 20)
 
    X, Y = np.meshgrid(x, y)

    # Define the condition for the second order cone
    z = []
    for x_val in x:
        for y_val in y:
            if params.vp.norm == 'inf':
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = np.inf))
            else:
                z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = params.vp.norm))
    z = np.array(z)
    
    fig.add_trace(go.Surface(x=X, y=Y, z=z.reshape(20,20), opacity = 0.25, showscale=False))
    frames = []

    if params.vp.tracking:
        x_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:,0])
        y_vals = 12 * np.ones_like(np.array(sub_positions_sen[0])[:,0])
    else:
        x_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:,0])
        y_vals = 110 * np.ones_like(np.array(sub_positions_sen[0])[:,0])

    # Add the projection of the second order cone onto the x-z plane
    z = []
    for x_val in x:
        if params.vp.norm == 'inf':
            z.append(np.linalg.norm(A @ np.array([x_val, 0]), axis=0, ord = np.inf))
        else:
            z.append(np.linalg.norm(A @ np.array([x_val, 0]), axis=0, ord = params.vp.norm))
    z = np.array(z)
    fig.add_trace(go.Scatter3d(y=x, x=y_vals, z=z, mode='lines', showlegend=False, line=dict(color='grey', width=3)))

    # Add the projection of the second order cone onto the y-z plane
    z = []
    for y_val in y:
        if params.vp.norm == 'inf':
            z.append(np.linalg.norm(A @ np.array([0, y_val]), axis=0, ord = np.inf))
        else:
            z.append(np.linalg.norm(A @ np.array([0, y_val]), axis=0, ord = params.vp.norm))
    z = np.array(z)
    fig.add_trace(go.Scatter3d(y=x_vals, x=y, z=z, mode='lines', showlegend=False, line=dict(color='grey', width=3)))

    # Choose a random color for each subject
    colors = []
    for sub_traj in sub_positions_sen:
        color = f'rgb({random.randint(10,255)}, {random.randint(10,255)}, {random.randint(10,255)})'
        colors.append(color)

    color_background = f'rgb({150}, {150}, {150})'
    sub_node_plot = []
    sub_node_idx = 0
    for i in range(0, len(sub_positions_sen[0]), 4):
        frame = go.Frame(name = str(i))
        data = []
        sub_idx = 0

        for sub_traj in sub_positions_sen:
            sub_traj = np.array(sub_traj)
            sub_traj_nodal = np.array(sub_positions_sen_node[sub_idx])

            if params.vp.tracking:
                x_vals = 12 * np.ones_like(sub_traj[:i+1, 0])
                y_vals = 12 * np.ones_like(sub_traj[:i+1, 0])
            else:
                x_vals = 110 * np.ones_like(sub_traj[:i+1, 0])
                y_vals = 110 * np.ones_like(sub_traj[:i+1, 0])

            data.append(go.Scatter3d(x = sub_traj[:i+1, 0], y = y_vals, z=sub_traj[:i+1, 2], mode='lines', showlegend=False, line=dict(color='grey', width=4)))
            data.append(go.Scatter3d(x = x_vals, y = sub_traj[:i+1, 1], z=sub_traj[:i+1, 2], mode='lines', showlegend=False, line=dict(color='grey', width=4)))

            # Add subject position to data
            # color = f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'
            sub_traj = np.array(sub_traj)
            data.append(go.Scatter3d(x=sub_traj[:i+1, 0], y=sub_traj[:i+1, 1], z=sub_traj[:i+1, 2], mode='lines',line=dict(color='darkblue', width=3), showlegend=False))

            # Add in node when loop has reached point where node is present
            scaled_index = int((i // (sub_traj.shape[0]/sub_traj_nodal.shape[0])) + 1)
            sub_node_plot = sub_traj_nodal[:scaled_index]

            # data.append(go.Scatter3d(x=sub_node_plot[:, 0], y=sub_node_plot[:, 1], z=sub_node_plot[:, 2], mode='markers', marker=dict(color='darkblue', size=5), showlegend=False))

            sub_idx += 1
        
        # Connect 0 to 16, 8, 12
        data.append(go.Scatter3d(x=[sub_positions_sen[0][i][0], sub_positions_sen[16][i][0]], y=[sub_positions_sen[0][i][1], sub_positions_sen[16][i][1]], z=[sub_positions_sen[0][i][2], sub_positions_sen[16][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[0][i][0], sub_positions_sen[8][i][0]], y=[sub_positions_sen[0][i][1], sub_positions_sen[8][i][1]], z=[sub_positions_sen[0][i][2], sub_positions_sen[8][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[0][i][0], sub_positions_sen[12][i][0]], y=[sub_positions_sen[0][i][1], sub_positions_sen[12][i][1]], z=[sub_positions_sen[0][i][2], sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 1 to 17, 9, 12
        data.append(go.Scatter3d(x=[sub_positions_sen[1][i][0], sub_positions_sen[17][i][0]], y=[sub_positions_sen[1][i][1], sub_positions_sen[17][i][1]], z=[sub_positions_sen[1][i][2], sub_positions_sen[17][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[1][i][0], sub_positions_sen[9][i][0]], y=[sub_positions_sen[1][i][1], sub_positions_sen[9][i][1]], z=[sub_positions_sen[1][i][2], sub_positions_sen[9][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[1][i][0], sub_positions_sen[12][i][0]], y=[sub_positions_sen[1][i][1], sub_positions_sen[12][i][1]], z=[sub_positions_sen[1][i][2], sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 2 to 16, 13, 10
        data.append(go.Scatter3d(x=[sub_positions_sen[2][i][0], sub_positions_sen[16][i][0]], y=[sub_positions_sen[2][i][1], sub_positions_sen[16][i][1]], z=[sub_positions_sen[2][i][2], sub_positions_sen[16][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[2][i][0], sub_positions_sen[13][i][0]], y=[sub_positions_sen[2][i][1], sub_positions_sen[13][i][1]], z=[sub_positions_sen[2][i][2], sub_positions_sen[13][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[2][i][0], sub_positions_sen[10][i][0]], y=[sub_positions_sen[2][i][1], sub_positions_sen[10][i][1]], z=[sub_positions_sen[2][i][2], sub_positions_sen[10][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 3 to 17, 11, 13
        data.append(go.Scatter3d(x=[sub_positions_sen[3][i][0], sub_positions_sen[17][i][0]], y=[sub_positions_sen[3][i][1], sub_positions_sen[17][i][1]], z=[sub_positions_sen[3][i][2], sub_positions_sen[17][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[3][i][0], sub_positions_sen[11][i][0]], y=[sub_positions_sen[3][i][1], sub_positions_sen[11][i][1]], z=[sub_positions_sen[3][i][2], sub_positions_sen[11][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[3][i][0], sub_positions_sen[13][i][0]], y=[sub_positions_sen[3][i][1], sub_positions_sen[13][i][1]], z=[sub_positions_sen[3][i][2], sub_positions_sen[13][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 4 to 18, 14, 8
        data.append(go.Scatter3d(x=[sub_positions_sen[4][i][0], sub_positions_sen[18][i][0]], y=[sub_positions_sen[4][i][1], sub_positions_sen[18][i][1]], z=[sub_positions_sen[4][i][2], sub_positions_sen[18][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[4][i][0], sub_positions_sen[14][i][0]], y=[sub_positions_sen[4][i][1], sub_positions_sen[14][i][1]], z=[sub_positions_sen[4][i][2], sub_positions_sen[14][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[4][i][0], sub_positions_sen[8][i][0]], y=[sub_positions_sen[4][i][1], sub_positions_sen[8][i][1]], z=[sub_positions_sen[4][i][2], sub_positions_sen[8][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 5 to 19, 9, 14
        data.append(go.Scatter3d(x=[sub_positions_sen[5][i][0], sub_positions_sen[19][i][0]], y=[sub_positions_sen[5][i][1], sub_positions_sen[19][i][1]], z=[sub_positions_sen[5][i][2], sub_positions_sen[19][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[5][i][0], sub_positions_sen[9][i][0]], y=[sub_positions_sen[5][i][1], sub_positions_sen[9][i][1]], z=[sub_positions_sen[5][i][2], sub_positions_sen[9][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[5][i][0], sub_positions_sen[14][i][0]], y=[sub_positions_sen[5][i][1], sub_positions_sen[14][i][1]], z=[sub_positions_sen[5][i][2], sub_positions_sen[14][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        
        # Connect 6 to 18, 15, 10
        data.append(go.Scatter3d(x=[sub_positions_sen[6][i][0], sub_positions_sen[18][i][0]], y=[sub_positions_sen[6][i][1], sub_positions_sen[18][i][1]], z=[sub_positions_sen[6][i][2], sub_positions_sen[18][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[6][i][0], sub_positions_sen[15][i][0]], y=[sub_positions_sen[6][i][1], sub_positions_sen[15][i][1]], z=[sub_positions_sen[6][i][2], sub_positions_sen[15][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[6][i][0], sub_positions_sen[10][i][0]], y=[sub_positions_sen[6][i][1], sub_positions_sen[10][i][1]], z=[sub_positions_sen[6][i][2], sub_positions_sen[10][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 7 to 19, 11, 15
        data.append(go.Scatter3d(x=[sub_positions_sen[7][i][0], sub_positions_sen[19][i][0]], y=[sub_positions_sen[7][i][1], sub_positions_sen[19][i][1]], z=[sub_positions_sen[7][i][2], sub_positions_sen[19][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[7][i][0], sub_positions_sen[11][i][0]], y=[sub_positions_sen[7][i][1], sub_positions_sen[11][i][1]], z=[sub_positions_sen[7][i][2], sub_positions_sen[11][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[7][i][0], sub_positions_sen[15][i][0]], y=[sub_positions_sen[7][i][1], sub_positions_sen[15][i][1]], z=[sub_positions_sen[7][i][2], sub_positions_sen[15][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 8 to 0, 4, 12
        data.append(go.Scatter3d(x=[sub_positions_sen[8][i][0], sub_positions_sen[0][i][0]], y=[sub_positions_sen[8][i][1], sub_positions_sen[0][i][1]], z=[sub_positions_sen[8][i][2], sub_positions_sen[0][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[8][i][0], sub_positions_sen[4][i][0]], y=[sub_positions_sen[8][i][1], sub_positions_sen[4][i][1]], z=[sub_positions_sen[8][i][2], sub_positions_sen[4][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[8][i][0], sub_positions_sen[12][i][0]], y=[sub_positions_sen[8][i][1], sub_positions_sen[12][i][1]], z=[sub_positions_sen[8][i][2], sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 9 to 1, 5, 11
        data.append(go.Scatter3d(x=[sub_positions_sen[9][i][0], sub_positions_sen[1][i][0]], y=[sub_positions_sen[9][i][1], sub_positions_sen[1][i][1]], z=[sub_positions_sen[9][i][2], sub_positions_sen[1][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[9][i][0], sub_positions_sen[5][i][0]], y=[sub_positions_sen[9][i][1], sub_positions_sen[5][i][1]], z=[sub_positions_sen[9][i][2], sub_positions_sen[5][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[9][i][0], sub_positions_sen[11][i][0]], y=[sub_positions_sen[9][i][1], sub_positions_sen[11][i][1]], z=[sub_positions_sen[9][i][2], sub_positions_sen[11][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 10 to 8, 2, 6
        data.append(go.Scatter3d(x=[sub_positions_sen[10][i][0], sub_positions_sen[8][i][0]], y=[sub_positions_sen[10][i][1], sub_positions_sen[8][i][1]], z=[sub_positions_sen[10][i][2], sub_positions_sen[8][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[10][i][0], sub_positions_sen[2][i][0]], y=[sub_positions_sen[10][i][1], sub_positions_sen[2][i][1]], z=[sub_positions_sen[10][i][2], sub_positions_sen[2][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[10][i][0], sub_positions_sen[6][i][0]], y=[sub_positions_sen[10][i][1], sub_positions_sen[6][i][1]], z=[sub_positions_sen[10][i][2], sub_positions_sen[6][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 11 to 3, 7, 9
        data.append(go.Scatter3d(x=[sub_positions_sen[11][i][0], sub_positions_sen[3][i][0]], y=[sub_positions_sen[11][i][1], sub_positions_sen[3][i][1]], z=[sub_positions_sen[11][i][2], sub_positions_sen[3][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[11][i][0], sub_positions_sen[7][i][0]], y=[sub_positions_sen[11][i][1], sub_positions_sen[7][i][1]], z=[sub_positions_sen[11][i][2], sub_positions_sen[7][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[11][i][0], sub_positions_sen[9][i][0]], y=[sub_positions_sen[11][i][1], sub_positions_sen[9][i][1]], z=[sub_positions_sen[11][i][2], sub_positions_sen[9][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        
        # Connect 12 to 0, 1, 14
        data.append(go.Scatter3d(x=[sub_positions_sen[12][i][0], sub_positions_sen[0][i][0]], y=[sub_positions_sen[12][i][1], sub_positions_sen[0][i][1]], z=[sub_positions_sen[12][i][2], sub_positions_sen[0][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[12][i][0], sub_positions_sen[1][i][0]], y=[sub_positions_sen[12][i][1], sub_positions_sen[1][i][1]], z=[sub_positions_sen[12][i][2], sub_positions_sen[1][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[12][i][0], sub_positions_sen[14][i][0]], y=[sub_positions_sen[12][i][1], sub_positions_sen[14][i][1]], z=[sub_positions_sen[12][i][2], sub_positions_sen[14][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 13 to 2, 3, 15
        data.append(go.Scatter3d(x=[sub_positions_sen[13][i][0], sub_positions_sen[2][i][0]], y=[sub_positions_sen[13][i][1], sub_positions_sen[2][i][1]], z=[sub_positions_sen[13][i][2], sub_positions_sen[2][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[13][i][0], sub_positions_sen[3][i][0]], y=[sub_positions_sen[13][i][1], sub_positions_sen[3][i][1]], z=[sub_positions_sen[13][i][2], sub_positions_sen[3][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[13][i][0], sub_positions_sen[15][i][0]], y=[sub_positions_sen[13][i][1], sub_positions_sen[15][i][1]], z=[sub_positions_sen[13][i][2], sub_positions_sen[15][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 14 to 4, 5, 12
        data.append(go.Scatter3d(x=[sub_positions_sen[14][i][0], sub_positions_sen[4][i][0]], y=[sub_positions_sen[14][i][1], sub_positions_sen[4][i][1]], z=[sub_positions_sen[14][i][2], sub_positions_sen[4][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[14][i][0], sub_positions_sen[5][i][0]], y=[sub_positions_sen[14][i][1], sub_positions_sen[5][i][1]], z=[sub_positions_sen[14][i][2], sub_positions_sen[5][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[14][i][0], sub_positions_sen[12][i][0]], y=[sub_positions_sen[14][i][1], sub_positions_sen[12][i][1]], z=[sub_positions_sen[14][i][2], sub_positions_sen[12][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 15 to 13, 6, 7
        data.append(go.Scatter3d(x=[sub_positions_sen[15][i][0], sub_positions_sen[13][i][0]], y=[sub_positions_sen[15][i][1], sub_positions_sen[13][i][1]], z=[sub_positions_sen[15][i][2], sub_positions_sen[13][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[15][i][0], sub_positions_sen[6][i][0]], y=[sub_positions_sen[15][i][1], sub_positions_sen[6][i][1]], z=[sub_positions_sen[15][i][2], sub_positions_sen[6][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[15][i][0], sub_positions_sen[7][i][0]], y=[sub_positions_sen[15][i][1], sub_positions_sen[7][i][1]], z=[sub_positions_sen[15][i][2], sub_positions_sen[7][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 16 to 0, 2, 17
        data.append(go.Scatter3d(x=[sub_positions_sen[16][i][0], sub_positions_sen[0][i][0]], y=[sub_positions_sen[16][i][1], sub_positions_sen[0][i][1]], z=[sub_positions_sen[16][i][2], sub_positions_sen[0][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[16][i][0], sub_positions_sen[2][i][0]], y=[sub_positions_sen[16][i][1], sub_positions_sen[2][i][1]], z=[sub_positions_sen[16][i][2], sub_positions_sen[2][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[16][i][0], sub_positions_sen[17][i][0]], y=[sub_positions_sen[16][i][1], sub_positions_sen[17][i][1]], z=[sub_positions_sen[16][i][2], sub_positions_sen[17][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 17 to 1, 3, 16
        data.append(go.Scatter3d(x=[sub_positions_sen[17][i][0], sub_positions_sen[1][i][0]], y=[sub_positions_sen[17][i][1], sub_positions_sen[1][i][1]], z=[sub_positions_sen[17][i][2], sub_positions_sen[1][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[17][i][0], sub_positions_sen[3][i][0]], y=[sub_positions_sen[17][i][1], sub_positions_sen[3][i][1]], z=[sub_positions_sen[17][i][2], sub_positions_sen[3][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[17][i][0], sub_positions_sen[16][i][0]], y=[sub_positions_sen[17][i][1], sub_positions_sen[16][i][1]], z=[sub_positions_sen[17][i][2], sub_positions_sen[16][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        
        # Connect 18 to 4, 6, 19
        data.append(go.Scatter3d(x=[sub_positions_sen[18][i][0], sub_positions_sen[4][i][0]], y=[sub_positions_sen[18][i][1], sub_positions_sen[4][i][1]], z=[sub_positions_sen[18][i][2], sub_positions_sen[4][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[18][i][0], sub_positions_sen[6][i][0]], y=[sub_positions_sen[18][i][1], sub_positions_sen[6][i][1]], z=[sub_positions_sen[18][i][2], sub_positions_sen[6][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[18][i][0], sub_positions_sen[19][i][0]], y=[sub_positions_sen[18][i][1], sub_positions_sen[19][i][1]], z=[sub_positions_sen[18][i][2], sub_positions_sen[19][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))

        # Connect 19 to 5, 7, 18
        data.append(go.Scatter3d(x=[sub_positions_sen[19][i][0], sub_positions_sen[5][i][0]], y=[sub_positions_sen[19][i][1], sub_positions_sen[5][i][1]], z=[sub_positions_sen[19][i][2], sub_positions_sen[5][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[19][i][0], sub_positions_sen[7][i][0]], y=[sub_positions_sen[19][i][1], sub_positions_sen[7][i][1]], z=[sub_positions_sen[19][i][2], sub_positions_sen[7][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))
        data.append(go.Scatter3d(x=[sub_positions_sen[19][i][0], sub_positions_sen[18][i][0]], y=[sub_positions_sen[19][i][1], sub_positions_sen[18][i][1]], z=[sub_positions_sen[19][i][2], sub_positions_sen[18][i][2]], mode='lines', line=dict(color='red', width=3), showlegend=False))


        frame.data = data
        frames.append(frame)
    
    fig.frames = frames

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.32,
            "steps": [
                {
                    "args": [[f.name], frame_args(500)],  # Use the frame name as the argument
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.32,
                                }
                            ],
                            sliders=sliders
                        )

    fig.update_layout(sliders=sliders)

    # Set camera position
    fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=10), center=dict(x=-2, y=0, z=-3), eye=dict(x=-28, y=-22, z=15)))

    # Set axis labels 
    fig.update_layout(scene=dict(xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)'))

    fig.update_layout(template='plotly_dark')
    
    # Make only the grid lines thicker in the template
    fig.update_layout(scene=dict(xaxis=dict(showgrid=True, gridwidth=5),
                                yaxis=dict(showgrid=True, gridwidth=5),
                                zaxis=dict(showgrid=True, gridwidth=5)))


    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=20, y=20, z=20)))
    # fig.update_layout(autosize=False, width=600, height=600)

    # Remove marigns
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # # Make the background transparent
    # fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    # # Make the axis backgrounds transparent
    # fig.update_layout(scene=dict(
    #     xaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey')
    # ))
    # # Remove the plot background
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # # Make ticks themselves transparent
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))

    # # Remove the paper background
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    # # Generate embded html
    # html_str = fig.to_html(full_html=False, include_plotlyjs='cdn', auto_play=False)
    # # Save the html string to a file
    # with open(f'{path}results/conic_animation.html', 'w') as f:
    #     f.write(html_str)

    fig.show()

def plot_animation(result: dict,
                   params,
                   path="",
                   ) -> None:
    tof = result["t_final"]
    # Make title say quadrotor simulation and insert the variable tof into the title
    # title = 'Quadrotor Simulation: Time of Flight = ' + str(tof) + 's'
    drone_positions = result["x_full"][:, :3]
    drone_velocities = result["x_full"][:, 3:6]
    drone_attitudes = result["x_full"][:, 6:10]
    drone_forces = result["u_full"][:, :3]
    if hasattr(params.veh, 'get_kp_pose') or "init_poses" in result:
        subs_positions, _, _, _ = full_subject_traj_time(result, params)

    step = 2
    indices = np.array(list(range(drone_positions.shape[0]-1)[::step]) + [drone_positions.shape[0]-1])

    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2)))
    for i in range(100):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='red', width = 2)))
    
    frames = []
    i = 0
    # Generate a color for each keypoint
    if "init_poses" in result or hasattr(params.veh, 'get_kp_pose'):
        color_kp = []
        if "init_poses" in result:
            for j in range(len(result["init_poses"])):
                color_kp.append(f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})')
        else:
            for j in range(1):
                color_kp.append(f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})')

    # Draw drone attitudes as axes
    for i in range(0, len(indices)-1, step):
        att = drone_attitudes[indices[i]]
        frame = go.Frame(name=str(i))

        subs_pose = []

        if hasattr(params.veh, 'get_kp_pose') or "init_poses" in result:
            for sub_positions in subs_positions:
                subs_pose.append(sub_positions[indices[i]])

        # Convert quaternion to rotation matrix
        rotation_matrix = qdcm(att)

        force = 0.5 * rotation_matrix @ drone_forces[indices[i]]

        # Extract axes from rotation matrix
        axes = 20 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
        rotated_axes = np.dot(rotation_matrix, axes).T

        # Meshgrid
        if hasattr(params.veh, 'get_kp_pose'):    
            x = np.linspace(-5, 5, 20)
            y = np.linspace(-5, 5, 20)
            z = np.linspace(-5, 5, 20)
        elif hasattr(params.veh, 'covariance') and params.veh.covariance:
            x = np.linspace(-2000, 2000, 20)
            y = np.linspace(-2000, 2000, 20)
            z = np.linspace(-2000, 2000, 20)
        else:
            x = np.linspace(-30, 30, 20)
            y = np.linspace(-30, 30, 20)
            z = np.linspace(-30, 30, 20)
        
        
        X, Y = np.meshgrid(x, y)

        data = []

        # Define the condition for the second order cone
        if (hasattr(params.veh, 'init_poses') or hasattr(params.veh, 'get_kp_pose')):
            if "alpha_x" in result and "alpha_y" in result:
                A = np.diag([1 / np.tan(np.pi / result["alpha_y"]), 1 / np.tan(np.pi / result["alpha_x"])])  # Conic Matrix
            else:
                raise ValueError("`alpha_x` and `alpha_y` not found in result dictionary.")
            if "norm_type" in result:
                z = []
                for x_val in x:
                    for y_val in y:
                        if result["norm_type"] == 'inf':
                            z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = np.inf))
                        else:
                            z.append(np.linalg.norm(A @ np.array([x_val, y_val]), axis=0, ord = result["norm_type"]))
                Z = np.array(z).reshape(20,20)
            else:
                raise ValueError("`norm_type` not found in result dictionary.")

            # Transform X,Y, and Z from the Sensor frame to the Body frame using R_sb
            if "R_sb" in result:
                R_sb = result["R_sb"]
            else:
                raise ValueError("`R_sb` not found in result dictionary.")
            X, Y, Z = R_sb.T @ np.array([X.flatten(), Y.flatten(), Z.flatten()])
            # Transform X,Y, and Z from the Body frame to the Inertial frame
            R_bi = qdcm(drone_attitudes[indices[i]])
            X, Y, Z = R_bi @ np.array([X, Y, Z])
            # Shift the meshgrid to the drone position
            X += drone_positions[indices[i], 0]
            Y += drone_positions[indices[i], 1]
            Z += drone_positions[indices[i], 2]

            # Make X, Y, Z back into a meshgrid
            X = X.reshape(20,20)
            Y = Y.reshape(20,20)
            Z = Z.reshape(20,20)

            data.append(go.Surface(x=X, y=Y, z=Z, opacity = 0.5, showscale=False, showlegend=True, name='Viewcone'))

        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFFFF']
        labels = ['X', 'Y', 'Z', 'Force']

        for k in range(3):
            if k < 3:
                axis = rotated_axes[k]
            color = colors[k]
            label = labels[k]

            if label == 'Force':
                data.append(go.Scatter3d(
                    x=[drone_positions[indices[i], 0], drone_positions[indices[i], 0] + force[0]],
                    y=[drone_positions[indices[i], 1], drone_positions[indices[i], 1] + force[1]],
                    z=[drone_positions[indices[i], 2], drone_positions[indices[i], 2] + force[2]],
                    mode='lines',
                    line=dict(color=color, width=4),
                    showlegend=False
                ))
            else:
                data.append(go.Scatter3d(
                    x=[drone_positions[indices[i], 0], drone_positions[indices[i], 0] + axis[0]],
                    y=[drone_positions[indices[i], 1], drone_positions[indices[i], 1] + axis[1]],
                    z=[drone_positions[indices[i], 2], drone_positions[indices[i], 2] + axis[2]],
                    mode='lines+text',
                    line=dict(color=color, width=4),
                    showlegend=False
                ))

        # Add subject position to data
        j = 0
        for sub_pose in subs_pose:
            # Use color iter to change the color of the subject in rgb
            data.append(go.Scatter3d(x=[sub_pose[0]], y=[sub_pose[1]], z=[sub_pose[2]], mode='markers', marker=dict(size=10, color=color_kp[j]), showlegend=False, name='Subject'))
            # if params.vp.n_subs != 1:
            j += 1
    
        data.append(go.Scatter3d(
            x=drone_positions[:indices[i]+1,0], 
            y=drone_positions[:indices[i]+1,1], 
            z=drone_positions[:indices[i]+1,2], 
            mode='markers',
            marker=dict(
                size=5,
                color=np.linalg.norm(drone_velocities[:indices[i]+1], axis = 1), # set color to an array/list of desired values
                colorscale='Viridis', # choose a colorscale
                colorbar=dict(title='Velocity Norm (m/s)', x=0.02, y=0.55, len=0.75) # add colorbar
            ),
            name='Nonlinear Propagation'
        ))
        

        # Make the subject draw a line as it moves
        if hasattr(params.veh, 'get_kp_pose'):
            for sub_positions in subs_positions:
                data.append(go.Scatter3d(x=sub_positions[:indices[i]+1,0], y=sub_positions[:indices[i]+1,1], z=sub_positions[:indices[i]+1,2], mode='lines', line=dict(color='red', width = 10), name='Subject Position'))
                
                sub_position = sub_positions[indices[i]]

                # Plot two spheres as a surface at the location of the subject to represent the minimum and maximum allowed range from the subject
                n = 20
                # Generate points on the unit sphere
                u = np.linspace(0, 2 * np.pi, n)
                v = np.linspace(0, np.pi, n)

                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))

                if "min_range" in result and "max_range" in result:
                    # Scale points by minimum range
                    x_min = result["min_range"] * x
                    y_min = result["min_range"] * y
                    z_min = result["min_range"] * z

                    # Scale points by maximum range
                    x_max = result["max_range"] * x
                    y_max = result["max_range"] * y
                    z_max = result["max_range"] * z
                else:
                    raise ValueError("`min_range` and `max_range` not found in result dictionary.")

                # Rotate and translate points
                points_min = np.array([x_min.flatten(), y_min.flatten(), z_min.flatten()])
                points_max = np.array([x_max.flatten(), y_max.flatten(), z_max.flatten()])
                
                points_min = points_min.T + sub_position
                points_max = points_max.T + sub_position

                data.append(go.Surface(x=points_min[:, 0].reshape(n,n), y=points_min[:, 1].reshape(n,n), z=points_min[:, 2].reshape(n,n), opacity = 0.2, colorscale='reds', name='Minimum Range', showlegend=True, showscale=False))
                data.append(go.Surface(x=points_max[:, 0].reshape(n,n), y=points_max[:, 1].reshape(n,n), z=points_max[:, 2].reshape(n,n), opacity = 0.2, colorscale='blues', name='Maximum Range', showlegend=True, showscale=False))


        frame.data = data
        frames.append(frame)

    fig.frames = frames

    if "obstacles_centers" in result:
        for center, axes, radius in zip(result['obstacles_centers'], result['obstacles_axes'], result['obstacles_radii']):
                n = 20
                # Generate points on the unit sphere
                u = np.linspace(0, 2 * np.pi, n)
                v = np.linspace(0, np.pi, n)

                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))

                # Scale points by radii
                x = 1/radius[0] * x
                y = 1/radius[1] * y
                z = 1/radius[2] * z

                # Rotate and translate points
                points = np.array([x.flatten(), y.flatten(), z.flatten()])
                points = axes @ points
                points = points.T + center

                fig.add_trace(go.Surface(x=points[:, 0].reshape(n,n), y=points[:, 1].reshape(n,n), z=points[:, 2].reshape(n,n), opacity = 0.5, showscale=False))

    if "vertices" in result:
        for vertices in result["vertices"]:
            # Plot a line through the vertices of the gate
            fig.add_trace(go.Scatter3d(x=[vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0], vertices[0][0]], y=[vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1], vertices[0][1]], z=[vertices[0][2], vertices[1][2], vertices[2][2], vertices[3][2], vertices[0][2]], mode='lines', showlegend=False, line=dict(color='blue', width=10)))

    # Add ground plane
    fig.add_trace(go.Surface(x=[-200, 200, 200, -200], y=[-200, -200, 200, 200], z=[[0, 0], [0, 0], [0, 0], [0, 0]], opacity=0.3, showscale=False, colorscale='Greys', showlegend = True, name='Ground Plane'))

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.32,
            "steps": [
                {
                    "args": [[f.name], frame_args(500)],  # Use the frame name as the argument
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.1,
                                    "y": 0,
                                }
                            ],
                            sliders=sliders
                        )

    fig.update_layout(sliders=sliders)

    fig.update_layout(template='plotly_dark') #, title=title)
    
    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    
    # Check if params.veh.covairance exists
    if hasattr(params.veh, 'covariance'):
        fig.update_layout(scene=dict(xaxis=dict(range=[0, 4000]), yaxis=dict(range=[0, 4000]), zaxis=dict(range=[-1000, 3000])))
    else:
        fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    



    # Overlay the sliders and buttons onto the plot
    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.22,
                                    "y": 0.37,
                                }
                            ],
                            sliders=sliders
                        )
    
    

    # Show the legend overlayed on the plot
    fig.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.75))

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Rmeove the background from the legend
    fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)'))

    fig.update_xaxes(
        dtick=1.0,
        showline=False
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=False,
        dtick=1.0
    )

    # Rotate the camera view to the left
    # if not params.vp.tracking and params.vp.n_subs != 0:
    #     fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=90), center=dict(x=1, y=0.3, z=1), eye=dict(x=-1, y=2, z=1)))

    # # Make the background transparent
    # fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    # # Make the axis backgrounds transparent
    # fig.update_layout(scene=dict(
    #     xaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey')
    # ))
    # # Remove the plot background
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # # Make ticks themselves transparent
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))

    # # Remove the paper background
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    fig.show()

def plot_scp_animation(result: dict,
                       params = None,
                       path=""):
    tof = result["t_final"]
    title = f'SCP Simulation: {tof} seconds'
    drone_positions = result["x_full"][:, :3]
    drone_attitudes = result["x_full"][:, 6:10]
    drone_forces = result["u_full"][:, :3]
    scp_interp_trajs = scp_traj_interp(result["x_history"], params)
    scp_ctcs_trajs = result["x_history"]
    scp_multi_shoot = result["discretization"]
    # obstacles = result_ctcs["obstacles"]
    # gates = result_ctcs["gates"]
    if hasattr(params.veh, 'get_kp_pose') or hasattr(params.veh, 'init_poses'):
        subs_positions, _, _, _ = full_subject_traj_time(result, params)
    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2), name='SCP Iterations'))
    for j in range(200):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', line=dict(color='gray', width = 2)))

    # fig.update_layout(height=1000)

    fig.add_trace(go.Scatter3d(x=drone_positions[:,0], y=drone_positions[:,1], z=drone_positions[:,2], mode='lines', line=dict(color='green', width = 5), name='Nonlinear Propagation'))

    fig.update_layout(template='plotly_dark', title=title)

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # Plot the attitudes of the SCP Trajs
    frames = []
    traj_iter = 0

    for scp_traj in scp_ctcs_trajs:
        drone_positions = scp_traj[:,0:3]
        drone_attitudes = scp_traj[:,6:10]
        frame = go.Frame(name=str(traj_iter))
        data = []
        # Plot the multiple shooting trajectories
        pos_traj = []
        if traj_iter < len(scp_multi_shoot):
            for i_multi in range(scp_multi_shoot[traj_iter].shape[1]):
                pos_traj.append(scp_multi_shoot[traj_iter][:,i_multi].reshape(-1, i5)[:,0:3])
            pos_traj = np.array(pos_traj)
            
            for j in range(pos_traj.shape[1]):
                if j == 0:
                    data.append(go.Scatter3d(x=pos_traj[:,j, 0], y=pos_traj[:,j, 1], z=pos_traj[:,j, 2], mode='lines', legendgroup='Multishot Trajectory', name='Multishot Trajectory ' + str(traj_iter), showlegend=True, line=dict(color='blue', width = 5)))
                else:
                    data.append(go.Scatter3d(x=pos_traj[:,j, 0], y=pos_traj[:,j, 1], z=pos_traj[:,j, 2], mode='lines', legendgroup='Multishot Trajectory', showlegend=False, line=dict(color='blue', width = 5)))
        
            
        for i in range(drone_attitudes.shape[0]):
            att = drone_attitudes[i]

            # Convert quaternion to rotation matrix
            rotation_matrix = qdcm(att)

            # Extract axes from rotation matrix
            axes = 2 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            rotated_axes = np.dot(rotation_matrix, axes.T).T

            colors = ['#FF0000', '#00FF00', '#0000FF']

            for k in range(3):
                axis = rotated_axes[k]
                color = colors[k]

                data.append(go.Scatter3d(
                    x=[scp_traj[i, 0], scp_traj[i, 0] + axis[0]],
                    y=[scp_traj[i, 1], scp_traj[i, 1] + axis[1]],
                    z=[scp_traj[i, 2], scp_traj[i, 2] + axis[2]],
                    mode='lines+text',
                    line=dict(color=color, width=4),
                    showlegend=False
                ))
        traj_iter += 1  
        frame.data = data
        frames.append(frame)
    fig.frames = frames 

    i = 1
    if "obstacles_centers" in result:
        for center, axes, radius in zip(result['obstacles_centers'], result['obstacles_axes'], result['obstacles_radii']):
            n = 20
            # Generate points on the unit sphere
            u = np.linspace(0, 2 * np.pi, n)
            v = np.linspace(0, np.pi, n)

            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))

            # Scale points by radii
            x = 1/radius[0] * x
            y = 1/radius[1] * y
            z = 1/radius[2] * z

            # Rotate and translate points
            points = np.array([x.flatten(), y.flatten(), z.flatten()])
            points = axes @ points
            points = points.T + center

            fig.add_trace(go.Surface(x=points[:, 0].reshape(n,n), y=points[:, 1].reshape(n,n), z=points[:, 2].reshape(n,n), opacity = 0.5, showscale=False))

    if "vertices" in result:
        for vertices in result["vertices"]:
            # Plot a line through the vertices of the gate
            fig.add_trace(go.Scatter3d(x=[vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0], vertices[0][0]], y=[vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1], vertices[0][1]], z=[vertices[0][2], vertices[1][2], vertices[2][2], vertices[3][2], vertices[0][2]], mode='lines', showlegend=False, line=dict(color='blue', width=10)))
            
    # Add the subject positions
    if "n_subs" in result and result["n_subs"] != 0:     
        if hasattr(params.veh, 'get_kp_pose'):
            for sub_positions in subs_positions:
                fig.add_trace(go.Scatter3d(x=sub_positions[:,0], y=sub_positions[:,1], z=sub_positions[:,2], mode='lines', line=dict(color='red', width = 5), showlegend=False))
        else:
            # Plot the subject positions as points
            for sub_positions in subs_positions:
                fig.add_trace(go.Scatter3d(x=sub_positions[:,0], y=sub_positions[:,1], z=sub_positions[:,2], mode='markers', marker=dict(size=10, color='red'), showlegend=False))


    fig.add_trace(go.Surface(x=[-200, 200, 200, -200], y=[-200, -200, 200, 200], z=[[0, 0], [0, 0], [0, 0], [0, 0]], opacity=0.3, showscale=False, colorscale='Greys', showlegend = True, name='Ground Plane'))

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.8,
            "x": 0.15,
            "y": 0.32,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": f.name,
                    "method": "animate",
                } for f in fig.frames
            ]
        }
    ]

    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.32,
                                }
                            ],
                            sliders=sliders
                        )
    fig.update_layout(sliders=sliders)

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=10, y=10, z=10)))
    fig.update_layout(scene=dict(xaxis=dict(range=[-200, 200]), yaxis=dict(range=[-200, 200]), zaxis=dict(range=[-200, 200])))

    # Overlay the title onto the plot
    fig.update_layout(title_y=0.95, title_x=0.5)

    # Overlay the sliders and buttons onto the plot
    fig.update_layout(updatemenus = [{"buttons":[
                                        {
                                            "args": [None, frame_args(50)],
                                            "label": "Play",
                                            "method": "animate",
                                        },
                                        {
                                            "args": [[None], frame_args(0)],
                                            "label": "Pause",
                                            "method": "animate",
                                    }],

                                    "direction": "left",
                                    "pad": {"r": 10, "t": 70},
                                    "type": "buttons",
                                    "x": 0.15,
                                    "y": 0.32,
                                }
                            ],
                            sliders=sliders
                        )
    
    

    # Show the legend overlayed on the plot
    fig.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.75))

    # fig.update_layout(height=450, width = 800)

    # Remove the black border around the fig
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Rmeove the background from the legend
    fig.update_layout(legend=dict(bgcolor='rgba(0,0,0,0)'))

    fig.update_xaxes(
        dtick=1.0,
        showline=False
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=False,
        dtick=1.0
    )

    # Rotate the camera view to the left
    if not hasattr(params.veh, 'get_kp_pose'):
        fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=90), center=dict(x=1, y=0.3, z=1), eye=dict(x=-1, y=2, z=1)))

    # # Make the background transparent
    # fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    # # Make the axis backgrounds transparent
    # fig.update_layout(scene=dict(
    #     xaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     yaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey'),
    #     zaxis=dict(backgroundcolor='rgba(0,0,0,0)', showbackground=False, showgrid=True, gridcolor='grey')
    # ))
    # # Remove the plot background
    # fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    # # Make ticks themselves transparent
    # fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))

    # # Remove the paper background
    # fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    fig.show()

def scp_traj_interp(scp_trajs, params):
    scp_prop_trajs = []
    for traj in scp_trajs:
        states = []
        for k in range(params.scp.n):
            traj_temp = np.repeat(np.expand_dims(traj[k], axis = 1), params.sim.inter_sample - 1, axis = 1)
            for i in range(1, params.sim.inter_sample - 1):
                states.append(traj_temp[:,i])
        scp_prop_trajs.append(np.array(states))
    return scp_prop_trajs

def plot_state(result, params: Config):
    scp_trajs = scp_traj_interp(result["x_history"], params)
    x_full = result["x_full"]

    fig = make_subplots(rows=2, cols=7, subplot_titles=('X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity', 'CTCS Augmentation', 'Q1', 'Q2', 'Q3', 'Q4', 'X Angular Rate', 'Y Angular Rate', 'Z Angular Rate'))
    fig.update_layout(title_text="State Trajectories", template='plotly_dark')

    # Plot the position
    x_min = params.sim.min_state[0]
    x_max = params.sim.max_state[0]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,0], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(y=x_full[:,0], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=1)
    fig.add_hline(y=x_min, line=dict(color='red', width=2), row = 1, col = 1)
    fig.add_hline(y=x_max, line=dict(color='red', width=2), row = 1, col = 1)
    fig.update_yaxes(range=[x_min, x_max], row=1, col=1)

    y_min = params.sim.min_state[1]
    y_max = params.sim.max_state[1]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,1], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(y=x_full[:,1], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=2)
    fig.add_hline(y=y_min, line=dict(color='red', width=2), row = 1, col = 2)
    fig.add_hline(y=y_max, line=dict(color='red', width=2), row = 1, col = 2)

    z_min = params.sim.min_state[2]
    z_max = params.sim.max_state[2]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,2], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=3)
    fig.add_trace(go.Scatter(y=x_full[:,2], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=3)
    fig.add_hline(y=z_min, line=dict(color='red', width=2), row = 1, col = 3)
    fig.add_hline(y=z_max, line=dict(color='red', width=2), row = 1, col = 3)

    # Plot the velocity
    vx_min = params.sim.min_state[3]
    vx_max = params.sim.max_state[3]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,3], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=4)
    fig.add_trace(go.Scatter(y=x_full[:,3], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=4)
    fig.add_hline(y=vx_min, line=dict(color='red', width=2), row = 1, col = 4)
    fig.add_hline(y=vx_max, line=dict(color='red', width=2), row = 1, col = 4)

    vy_min = params.sim.min_state[4]
    vy_max = params.sim.max_state[4]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,4], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=5)
    fig.add_trace(go.Scatter(y=x_full[:,4], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=5)
    fig.add_hline(y=vy_min, line=dict(color='red', width=2), row = 1, col = 5)
    fig.add_hline(y=vy_max, line=dict(color='red', width=2), row = 1, col = 5)

    vz_min = params.sim.min_state[5]
    vz_max = params.sim.max_state[5]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,5], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=6)
    fig.add_trace(go.Scatter(y=x_full[:,5], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=6)
    fig.add_hline(y=vz_min, line=dict(color='red', width=2), row = 1, col = 6)
    fig.add_hline(y=vz_max, line=dict(color='red', width=2), row = 1, col = 6)

    # # Plot the norm of the quaternion
    # for traj in scp_trajs:
    #     fig.add_trace(go.Scatter(y=la.norm(traj[1:,6:10], axis = 1), mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=7)
    # fig.add_trace(go.Scatter(y=la.norm(x_full[1:,6:10], axis = 1), mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=7)
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,-1], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=7)
    fig.add_trace(go.Scatter(y=x_full[:,-1], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=7)
    # fig.add_hline(y=vz_min, line=dict(color='red', width=2), row = 1, col = 6)
    # fig.add_hline(y=vz_max, line=dict(color='red', width=2), row = 1, col = 6)

    # Plot the attitude
    q1_min = params.sim.min_state[6]
    q1_max = params.sim.max_state[6]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,6], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(y=x_full[:,6], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=1)
    fig.add_hline(y=q1_min, line=dict(color='red', width=2), row = 2, col = 1)
    fig.add_hline(y=q1_max, line=dict(color='red', width=2), row = 2, col = 1)

    q2_min = params.sim.min_state[7]
    q2_max = params.sim.max_state[7]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,7], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=2)
    fig.add_trace(go.Scatter(y=x_full[:,7], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=2)
    fig.add_hline(y=q2_min, line=dict(color='red', width=2), row = 2, col = 2)
    fig.add_hline(y=q2_max, line=dict(color='red', width=2), row = 2, col = 2)

    q3_min = params.sim.min_state[8]
    q3_max = params.sim.max_state[8]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,8], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=3)
    fig.add_trace(go.Scatter(y=x_full[:,8], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=3)
    fig.add_hline(y=q3_min, line=dict(color='red', width=2), row = 2, col = 3)
    fig.add_hline(y=q3_max, line=dict(color='red', width=2), row = 2, col = 3)

    q4_min = params.sim.min_state[9]
    q4_max = params.sim.max_state[9]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,9], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=4)
    fig.add_trace(go.Scatter(y=x_full[:,9], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=4)
    fig.add_hline(y=q4_min, line=dict(color='red', width=2), row = 2, col = 4)
    fig.add_hline(y=q4_max, line=dict(color='red', width=2), row = 2, col = 4)

    # Plot the angular rate
    wx_min = params.sim.min_state[10]
    wx_max = params.sim.max_state[10]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,10], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=5)
    fig.add_trace(go.Scatter(y=x_full[:,10], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=5)
    fig.add_hline(y=wx_min, line=dict(color='red', width=2), row = 2, col = 5)
    fig.add_hline(y=wx_max, line=dict(color='red', width=2), row = 2, col = 5)

    wy_min = params.sim.min_state[11]
    wy_max = params.sim.max_state[11]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,11], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=6)
    fig.add_trace(go.Scatter(y=x_full[:,11], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=6)
    fig.add_hline(y=wy_min, line=dict(color='red', width=2), row = 2, col = 6)
    fig.add_hline(y=wy_max, line=dict(color='red', width=2), row = 2, col = 6)

    wz_min = params.sim.min_state[12]
    wz_max = params.sim.max_state[12]
    for traj in scp_trajs:
        fig.add_trace(go.Scatter(y=traj[:,12], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=7)
    fig.add_trace(go.Scatter(y=x_full[:,12], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=7)
    fig.add_hline(y=wz_min, line=dict(color='red', width=2), row = 2, col = 7)
    fig.add_hline(y=wz_max, line=dict(color='red', width=2), row = 2, col = 7)
    fig.show()

def plot_control(result, params: Config):
    scp_controls = result["u_history"]
    u = result["u"]

    fx_min = params.sim.min_control[0]
    fx_max = params.sim.max_control[0]

    fy_min = params.sim.min_control[1]
    fy_max = params.sim.max_control[1]

    fz_min = params.sim.min_control[2]
    fz_max = params.sim.max_control[2]

    tau_x_min = params.sim.max_control[3]
    tau_x_max = params.sim.min_control[3]

    tau_y_min = params.sim.max_control[4]
    tau_y_max = params.sim.min_control[4]

    tau_z_min = params.sim.max_control[5]
    tau_z_max = params.sim.min_control[5]

    fig = make_subplots(rows=2, cols=3, subplot_titles=('X Force', 'Y Force', 'Z Force', 'X Torque', 'Y Torque', 'Z Torque'))
    fig.update_layout(title_text="Control Trajectories", template='plotly_dark')

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[0], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(y=u[0], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=1)
    fig.add_hline(y=fx_min, line=dict(color='red', width=2), row = 1, col = 1)
    fig.add_hline(y=fx_max, line=dict(color='red', width=2), row = 1, col = 1)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[1], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(y=u[1], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=2)
    fig.add_hline(y=fy_min, line=dict(color='red', width=2), row = 1, col = 2)
    fig.add_hline(y=fy_max, line=dict(color='red', width=2), row = 1, col = 2)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[2], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=1, col=3)
    fig.add_trace(go.Scatter(y=u[2], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=3)
    fig.add_hline(y=fz_min, line=dict(color='red', width=2), row = 1, col = 3)
    fig.add_hline(y=fz_max, line=dict(color='red', width=2), row = 1, col = 3)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[3], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(y=u[3], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=1)
    fig.add_hline(y=tau_x_min, line=dict(color='red', width=2), row = 2, col = 1)
    fig.add_hline(y=tau_x_max, line=dict(color='red', width=2), row = 2, col = 1)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[4], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=2)
    fig.add_trace(go.Scatter(y=u[4], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=2)
    fig.add_hline(y=tau_y_min, line=dict(color='red', width=2), row = 2, col = 2)
    fig.add_hline(y=tau_y_max, line=dict(color='red', width=2), row = 2, col = 2)

    for traj in scp_controls:
        fig.add_trace(go.Scatter(y=traj[5], mode='lines', showlegend=False, line=dict(color='gray', width = 0.5)), row=2, col=3)
    fig.add_trace(go.Scatter(y=u[5], mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=3)
    fig.add_hline(y=tau_z_min, line=dict(color='red', width=2), row = 2, col = 3)
    fig.add_hline(y=tau_z_max, line=dict(color='red', width=2), row = 2, col = 3)

    fig.show()

def plot_losses(result, params: Config):
    # Plot J_tr, J_vb, J_vc, J_vc_ctcs
    J_tr = result["J_tr_history"]
    J_vb = result["J_vb_history"]
    J_vc = result["J_vc_history"]
    J_vc_ctcs = result["J_vc_ctcs_vec"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=('J_tr', 'J_vb', 'J_vc', 'J_vc_ctcs'))
    fig.update_layout(title_text="Losses", template='plotly_dark')

    fig.add_trace(go.Scatter(y=J_tr, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=J_vb, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=J_vc, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=1)
    fig.add_trace(go.Scatter(y=J_vc_ctcs, mode='lines', showlegend=False, line=dict(color='green', width = 2)), row=2, col=2)

    # Set y-axis to log scale for each subplot
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=2)
    fig.update_yaxes(type='log', row=2, col=1)
    fig.update_yaxes(type='log', row=2, col=2)

    fig.show()