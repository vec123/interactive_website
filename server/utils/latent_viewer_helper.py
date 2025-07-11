import matplotlib.pyplot as plt
from utils import numpy_motion_data_helper as np_motion_helpers
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def output_to_joint_positions(output_vector, joints, feature_names, joint_names, data_dim):
    """
    Convert model output vector to a list of [x,y,z] joint positions.
    """
    # Reshape to (features, frames)
    kinematic_motion = output_vector.reshape((1, data_dim))
    
    # Convert back to AMC motion and Cartesian data
    amc_motions = np_motion_helpers.numpy_to_amc_kinematic_tree(
        kinematic_motion, feature_names, joints
    )
    
    spatial_motion_data, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(
        joints, amc_motions
    )
    
    # Extract the first frame
    num_joints = len(joint_names)
    frame_data = spatial_motion_data[0].reshape(num_joints, 3)
    
    # Swap Y/Z for visualization consistency
    # ys, zs = np_motion_helpers.swap_yz(frame_data[:,1], frame_data[:,2])
   
    # Return a list of positions: [[x,y,z], ...]
    positions = np.stack([frame_data[:,0], frame_data[:,1], frame_data[:,2]], axis=1)
    return positions

def plot_latent_space(latent_samples, callback):
    """
    Plot the latent space and allow updates on mouse click and motion.
    """
    fig_latent, ax_latent = plt.subplots(figsize=(8, 6))
    scatter = ax_latent.scatter(latent_samples[:, 0], latent_samples[:, 1], alpha=0.6)
    ax_latent.set_xlabel("Latent Dimension 1")
    ax_latent.set_ylabel("Latent Dimension 2")
    ax_latent.set_title("Latent Space (Click and Drag to Explore)")

    # State tracking for mouse button press
    mouse_pressed = [False]  # Use a mutable object to modify state inside event handlers

    def on_press(event):
        if event.inaxes == ax_latent:  # Check if event is in the correct figure
            mouse_pressed[0] = True
            if event.xdata is not None and event.ydata is not None:
                clicked_point = np.array([event.xdata, event.ydata])
                callback(clicked_point)  # Update skeleton

    def on_release(event):
        mouse_pressed[0] = False  # Reset when mouse button is released

    def on_motion(event):
        if mouse_pressed[0] and event.inaxes == ax_latent:  # Update only when button is pressed
            if event.xdata is not None and event.ydata is not None:
                dragged_point = np.array([event.xdata, event.ydata])
                callback(dragged_point)  # Update skeleton continuously

    # Connect event handlers
    fig_latent.canvas.mpl_connect('button_press_event', on_press)
    fig_latent.canvas.mpl_connect('button_release_event', on_release)
    fig_latent.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show(block=True)  # Keep the latent space plot open


# Global figure and axes
# Initialize a global figure and axes
fig, ax = None, None

def generate_and_plot_kinematic_tree_output(model, latent_point, joints, edges, feature_names, joint_names):
    """
    Generate a sample from the model based on the clicked latent point and update the 3D skeleton plot.
    """
    global fig, ax  # Use global figure and axes to persist between clicks

    # Generate motion data
    with torch.no_grad():
        latent_point_tensor = torch.tensor(latent_point, dtype=torch.float32).unsqueeze(0)
        kinematic_motion = model(latent_point_tensor).mean.cpu().numpy().T
        
    print("kinematic_motion.shape:", kinematic_motion.shape)
    # Convert back to AMC motion and Cartesian data
    amc_motions = np_motion_helpers.numpy_to_amc_kinematic_tree(kinematic_motion, feature_names, joints)
    spatial_motion_data, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(joints, amc_motions)

    print("spatial_motion_data.shape:", spatial_motion_data.shape)

    # Extract the first frame
    num_joints = len(joint_names)
    frame_data = spatial_motion_data[0].reshape(num_joints, 3)
    xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    ys, zs = np_motion_helpers.swap_yz(ys, zs)

    # Initialize figure and axes if not already initialized
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()  # Turn on interactive mode to dynamically update the figure

    # Clear the axes for updating
    ax.cla()

    # Plot the joints
    ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")

    # Plot edges
    edges_indices = edges[0]  # Use only the numerical edge indices
    for start, end in edges_indices:
        ax.plot(
            [xs[start], xs[end]],
            [ys[start], ys[end]],
            [zs[start], zs[end]],
            color='black'
        )

    # Update axis limits dynamically
    margin = 10
    ax.set_xlim(np.min(xs) - margin, np.max(xs) + margin)
    ax.set_ylim(np.min(ys) - margin, np.max(ys) + margin)
    ax.set_zlim(np.min(zs) - margin, np.max(zs) + margin)

    # Set plot properties
    ax.set_title("Updated Skeleton: Clicked Latent Point")
    ax.view_init(elev=30, azim=45)
    plt.draw()  # Redraw the figure
    plt.pause(0.1)  # Allow the figure to update


def generate_and_plot_kinematic_tree_output_numpy(model, latent_point, joints, edges, feature_names, joint_names):
    """
    Generate a sample from the model based on the clicked latent point and update the 3D skeleton plot.
    """
    global fig, ax  # Use global figure and axes to persist between clicks

    # Generate motion data
    with torch.no_grad():
        latent_point_tensor = torch.tensor(latent_point, dtype=torch.float32).unsqueeze(0)
        kinematic_motion = model(latent_point_tensor).mean.cpu().numpy().T

    # Convert back to AMC motion and Cartesian data
    print("kinematic_motion.shape", kinematic_motion.shape)
    print(err)
    amc_motions = np_motion_helpers.numpy_to_amc_kinematic_tree(kinematic_motion, feature_names, joints)
    spatial_motion_data, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(joints, amc_motions)

    print("spatial_motion_data.shape:", spatial_motion_data.shape)

    # Extract the first frame
    num_joints = len(joint_names)
    frame_data = spatial_motion_data[0].reshape(num_joints, 3)
    xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    ys, zs = np_motion_helpers.swap_yz(ys, zs)

    # Initialize figure and axes if not already initialized
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()  # Turn on interactive mode to dynamically update the figure

    # Clear the axes for updating
    ax.cla()

    # Plot the joints
    ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")

    # Plot edges
    edges_indices = edges[0]  # Use only the numerical edge indices
    for start, end in edges_indices:
        ax.plot(
            [xs[start], xs[end]],
            [ys[start], ys[end]],
            [zs[start], zs[end]],
            color='black'
        )

    # Update axis limits dynamically
    margin = 10
    ax.set_xlim(np.min(xs) - margin, np.max(xs) + margin)
    ax.set_ylim(np.min(ys) - margin, np.max(ys) + margin)
    ax.set_zlim(np.min(zs) - margin, np.max(zs) + margin)

    # Set plot properties
    ax.set_title("Updated Skeleton: Clicked Latent Point")
    ax.view_init(elev=30, azim=45)
    plt.draw()  # Redraw the figure
    plt.pause(0.1)  # Allow the figure to update

"""

def generate_and_plot_cartesian_output(latent_point, edges, skeleton_ax):

    with torch.no_grad():
        latent_point_tensor = torch.tensor(latent_point, dtype=torch.float32).unsqueeze(0)  # Shape (1, 2)
        output = model(latent_point_tensor).mean  # Predict the output using the model
        output = output.cpu().numpy().reshape(num_joints, 3)  # Reshape to (num_joints, 3)

    # Clear the skeleton plot and update
    skeleton_ax.clear()
    skeleton_ax.scatter(output[:, 0], output[:, 1], color='blue', label='Joints')
    skeleton_ax.scatter(output[7, 0], output[7, 1], c = "red",alpha=0.6)
    skeleton_ax.scatter(output[6, 0], output[6, 1], c = "red",alpha=0.6)
    skeleton_ax.scatter(output[1, 0], output[1, 1], c = "green",alpha=0.6)
    skeleton_ax.scatter(output[0, 0], output[0, 1], c = "green",alpha=0.6)
    
    for edge in edges:
        joint1, joint2 = edge
        skeleton_ax.plot(
            [output[joint1, 0], output[joint2, 0]],
            [output[joint1, 1], output[joint2, 1]],
            color='black'
        )
    skeleton_ax.set_title("Generated 2D Skeleton")
    skeleton_ax.set_xlabel("X")
    skeleton_ax.set_ylabel("Y")
    skeleton_ax.legend()
    plt.pause(0.1)

def generate_and_plot_kinematic_tree_output(model, latent_point, joints, edges, feature_names, joint_names):

    with torch.no_grad():
        latent_point_tensor = torch.tensor(latent_point, dtype=torch.float32).unsqueeze(0)  # Shape (1, 2)
        kinematic_motion = model(latent_point_tensor).mean  # Predict the output using the model
        kinematic_motion = kinematic_motion.cpu().numpy()  # Reshape to (num_joints, 3)
        kinematic_motion = kinematic_motion.T
    
    amc_motions = np_motion_helpers.numpy_to_amc_kinematic_tree(kinematic_motion, feature_names, joints) 
    spatial_motion_data, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(joints, amc_motions)

    print("spatial_motion_data.shape", spatial_motion_data.shape)
    np_motion_helpers.plot_motion_dynamic_(spatial_motion_data, joint_names, edges, frame_interval=50)
    
def plot_latent_space_(latent_samples, callback):

    fig_latent, ax_latent = plt.subplots(figsize=(8, 6))
    scatter = ax_latent.scatter(latent_samples[:, 0], latent_samples[:, 1], alpha=0.6)
    ax_latent.set_xlabel("Latent Dimension 1")
    ax_latent.set_ylabel("Latent Dimension 2")
    ax_latent.set_title("Latent Space (Click to Explore)")
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            clicked_point = np.array([event.xdata, event.ydata])
            callback(clicked_point)

    cid = fig_latent.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=False)  # Keep the latent space plot open


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_and_plot_kinematic_tree_output_(model, latent_point, joints, edges, feature_names, joint_names):

    with torch.no_grad():
        latent_point_tensor = torch.tensor(latent_point, dtype=torch.float32).unsqueeze(0)
        kinematic_motion = model(latent_point_tensor).mean.cpu().numpy().T

    # Convert back to AMC motion and Cartesian data
    amc_motions = np_motion_helpers.numpy_to_amc_kinematic_tree(kinematic_motion, feature_names, joints)
    spatial_motion_data, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(joints, amc_motions)

    print("spatial_motion_data.shape:", spatial_motion_data.shape)

    # Reshape spatial motion data
    num_frames, num_joints = spatial_motion_data.shape[0], len(joint_names)
    spatial_coords = spatial_motion_data.reshape(num_frames, num_joints, 3)

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_single_frame():

        frame_data = spatial_coords[0]
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
        ys, zs = np_motion_helpers.swap_yz(ys, zs)
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")
       
        # Extract only the numerical edge indices
        edges_indices = edges[0]
        for start, end in edges_indices:
            ax.plot(
                [xs[start], xs[end]],
                [ys[start], ys[end]],
                [zs[start], zs[end]],
                color='black'
            )

        ax.set_title("Static Frame: Single Frame Data")
        ax.view_init(elev=30, azim=45)
        plt.show()

    def update(frame_idx):

        ax.cla()  # Clear the previous frame
        frame_data = spatial_coords[frame_idx]
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

        # Set dynamic axis limits
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        z_min, z_max = np.min(zs), np.max(zs)
        margin = 10

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)

        
        # Plot joints
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")

        # Plot edges
        for edge in edges:
            start, end = edge
            ax.plot(
                [xs[start], xs[end]],
                [ys[start], ys[end]],
                [zs[start], zs[end]],
                color='black'
            )

        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")
        ax.view_init(elev=30, azim=45)

    # Handle single frame case
    if num_frames == 1:
        plot_single_frame()
    else:
        # Create animation for multiple frames
        anim = FuncAnimation(fig, update, frames=num_frames, interval=50)
        plt.show()
        return anim

"""