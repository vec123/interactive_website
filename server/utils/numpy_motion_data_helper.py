import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def swap_yz(ys, zs):
            ys_, zs = zs, ys
            return ys_, zs

def extract_cartesian_motion_data_from_amc(joints, motion_frames):
    """
    Extracts motion data as a NumPy array from joint hierarchy and motion frames.
    
    Parameters:
    - joints: Dictionary of Joint objects representing the skeleton.
    - motion_frames: List of dictionaries containing motion data for each frame.

    Returns:
    - A NumPy array of shape (num_frames, num_joints * 3) containing joint coordinates.
    """
    num_frames = len(motion_frames)
    num_joints = len(joints)
    motion_data = np.zeros((num_frames, num_joints * 3))
    
    joint_names = list(joints.keys())
    
    for frame_idx, motion_frame in enumerate(motion_frames):
        # Update joint positions for the current frame
        joints['root'].set_motion(motion_frame)
        
        # Extract coordinates for all joints
        frame_data = []
        for joint_name in joint_names:
            joint = joints[joint_name]
            frame_data.extend(joint.coordinate.flatten())  # Flatten 3D coordinate to [x, y, z]
        
        motion_data[frame_idx, :] = frame_data
    
    return np.array(motion_data), np.array(joint_names)

def extract_edges(joints):
    """
    Extracts edges (connections) between joints in the hierarchy.
    
    Parameters:
    - joints: Dictionary of Joint objects representing the skeleton.
    
    Returns:
    - edges: List of tuples (parent_index, child_index) representing connections between joints.
    """
    edges = []
    joint_names = list(joints.keys())
    joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    
    for joint_name, joint in joints.items():
        if joint.parent is not None:
            parent_idx = joint_name_to_index[joint.parent.name]
            child_idx = joint_name_to_index[joint_name]
            edges.append((parent_idx, child_idx))
    
    return np.array(edges), np.array(joint_names)

def plot_motion(motion_data, joint_names, edges, frame_interval=50):
    """
    Visualizes the motion data as a 3D animation.
    
    Parameters:
    - motion_data: NumPy array of shape (num_frames, num_joints * 3) containing joint coordinates.
    - joint_names: List of joint names corresponding to the columns in motion_data.
    - edges: List of tuples representing edges (connections between joints).
    - frame_interval: Time interval (in milliseconds) between frames in the animation.
    """
    num_frames, _ = motion_data.shape
    num_joints = len(joint_names)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.cla()  # Clear the plot for the current frame
        
        # Set axis limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        
        # Extract joint positions for the current frame
        frame_data = motion_data[frame_idx, :].reshape((num_joints, 3))
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

        def swap_yz(ys,zs):
            ys_ = zs
            zs = ys
            ys = ys_
            return ys, zs
        ys,zs = swap_yz(ys,zs)

        # Plot joints
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")
        
        # Plot edges (connections between joints)
        for edge in edges:
            start, end = edge
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'r')
        
        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")
    

    anim = FuncAnimation(fig, update, frames=num_frames, interval=frame_interval)
    
    plt.show()

def build_adjacency_matrix(joints, edges):
    """
    Constructs an adjacency matrix for the kinematic tree.

    Parameters:
    - joints: Dictionary of Joint objects representing the skeleton.
    - edges: List of tuples (parent_index, child_index) representing connections between joints.

    Returns:
    - adj_matrix: NumPy array of shape (num_joints, num_joints).
    """
    num_joints = len(joints)
    adj_matrix = np.zeros((num_joints, num_joints))
    for parent_idx, child_idx in edges:
        adj_matrix[parent_idx, child_idx] = 1
        adj_matrix[child_idx, parent_idx] = 1  # Assuming bidirectional connectivity
    
    return adj_matrix

def amc_to_numpy_kinematic_tree(motions, joints):
    """
    Converts AMC motion data into a NumPy array suitable for training.
    
    Parameters:
    - motions: List of dictionaries containing joint motions for each frame (parsed AMC data).
    - joints: Dictionary of Joint objects representing the skeleton.

    Returns:
    - motion_array: NumPy array of shape (num_frames, num_features).
      - First 6 columns: Root translation (3) + Root rotation (3).
      - Remaining columns: Joint rotations (degrees of freedom for each joint in hierarchical order).
    - feature_names: List of feature names corresponding to the columns in motion_array.
    """
    num_frames = len(motions)
    num_joints = len(joints)
    
    # Count total degrees of freedom (DOF) for all joints
    dof_per_joint = {name: len(joint.dof) for name, joint in joints.items()}
    total_dof = sum(dof_per_joint.values()) + 6  # 6 corresponds to Root translation (3) + Root rotation (3)

    # Initialize motion array
    motion_array = np.zeros((num_frames, total_dof))
    feature_names = []

    for frame_idx, motion in enumerate(motions):
        col_idx = 0
        
        # Process root joint (translation + rotation)
        if 'root' in motion:
            motion_array[frame_idx, col_idx:col_idx + 6] = motion['root']
            if frame_idx == 0:
                feature_names.extend(['root_tx', 'root_ty', 'root_tz', 'root_rx', 'root_ry', 'root_rz'])
            col_idx += 6

        # Process other joints
        for joint_name, joint in joints.items():
            if joint_name == 'root':
                continue
            dof = dof_per_joint[joint_name]
            if joint_name in motion:
                motion_array[frame_idx, col_idx:col_idx + dof] = motion[joint_name]
                if frame_idx == 0:
                    feature_names.extend([f"{joint_name}_{axis}" for axis in joint.dof])
            else:
                pass
                #print("joint {joint_name} is not in motion")
                # Fill missing joints with zeros (or default pose if needed)
                # motion_array[frame_idx, col_idx:col_idx + dof] = np.zeros(dof)
                #feature_names.extend([f"{joint_name}_{axis}" for axis in joint.dof])
                #raise ValueError(f"Missing motion data for joint {joint_name}")
            
            col_idx += dof
    return motion_array, feature_names

def numpy_to_amc_kinematic_tree(motion_array, feature_names, joints):
    """
    Converts a NumPy motion array back into AMC motion data format.

    Parameters:
    - motion_array: NumPy array of shape (num_frames, num_features).
    - feature_names: List of feature names corresponding to the columns in motion_array.
    - joints: Dictionary of Joint objects representing the skeleton (used for degree of freedom info).

    Returns:
    - motions: List of dictionaries containing joint motions for each frame.
    """
    motions = []
    num_frames = motion_array.shape[0]

    # Identify root indices for translation and rotation
    root_indices = {
        'root_tx': None, 'root_ty': None, 'root_tz': None,
        'root_rx': None, 'root_ry': None, 'root_rz': None
    }
    joint_indices = {joint_name: [] for joint_name in joints if joint_name != 'root'}

    # Parse feature names to get indices for root and joints
    for idx, feature in enumerate(feature_names):
        if feature in root_indices:
            root_indices[feature] = idx
        else:
            joint_name, dof = feature.rsplit('_', 1)
            if joint_name in joint_indices:
                joint_indices[joint_name].append((idx, dof))

    # Reconstruct motions frame by frame
    for frame_idx in range(num_frames):
        frame_motion = {}

        # Process root joint
        root_motion = [
            motion_array[frame_idx, root_indices[f'root_t{axis}']] for axis in 'xyz'
        ] + [
            motion_array[frame_idx, root_indices[f'root_r{axis}']] for axis in 'xyz'
        ]
        frame_motion['root'] = root_motion

        # Process other joints
        for joint_name, indices in joint_indices.items():
            joint_motion = []
            indices_sorted = sorted(indices, key=lambda x: x[1])  # Sort indices by dof order (e.g., rx, ry, rz)
            for idx, _ in indices_sorted:
                joint_motion.append(motion_array[frame_idx, idx])
            frame_motion[joint_name] = joint_motion

        motions.append(frame_motion)

    return motions

def plot_motion_dynamic(motion_data, edges, frame_interval=50):
    """
    Visualizes the motion data as a 3D animation, dynamically centering and adjusting the view.
    """
    num_frames, num_joint_coor = motion_data.shape
    if num_joint_coor % 3 != 0:
        raise ValueError("Number of joint coordinates must be divisible by 3.")
    num_joints = int(num_joint_coor/3)  # Each joint has 3 coordinates (x, y, z)

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()
        print("frame_idx", frame_idx)
        frame_data = motion_data[frame_idx, :].reshape((num_joints, 3))
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

        # Adjust axes dynamically
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")
        
        for edge in edges:
            start, end = edge
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'r')
        
        for i in range(num_joints):
            ax.text(xs[i], ys[i], zs[i], f'{i}', color='black', fontsize=10, ha='center')

        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")

    # Persist the animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=frame_interval)
    plt.show()
    return anim

def save_motion_dynamic(motion_data, edges, save_path, frame_interval=50, file_format='mp4', fps=30, dpi=300):
    """
    Visualizes the motion data as a 3D animation, dynamically centering and adjusting the view, and saves it to a specified path.
    
    Parameters:
        motion_data (numpy array): The 3D coordinates of joints over time (shape: [num_frames, num_joints * 3]).
        edges (list of tuples): A list of joint pairs to be connected by lines (edges).
        save_path (str): The path where the animation will be saved.
        frame_interval (int): The interval between frames in milliseconds.
        file_format (str): The file format to save the animation ('mp4', 'gif', etc.).
        fps (int): Frames per second for the saved animation.
        dpi (int): Dots per inch for the saved animation (higher DPI means better quality).
    """
    num_frames, num_joint_coor = motion_data.shape
    if num_joint_coor % 3 != 0:
        raise ValueError("Number of joint coordinates must be divisible by 3.")
    num_joints = int(num_joint_coor / 3)  # Each joint has 3 coordinates (x, y, z)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()  # Clear the previous plot
        print("frame_idx", frame_idx)
        frame_data = motion_data[frame_idx, :].reshape((num_joints, 3))
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

        # Plot joints
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")
        
        # Plot edges (skeleton connections)
        for edge in edges:
            start, end = edge
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'r')
        
        # Add joint labels
        for i in range(num_joints):
            ax.text(xs[i], ys[i], zs[i], f'{i}', color='black', fontsize=10, ha='center')

        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=frame_interval)
    
    # Save the animation
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    print(f"Animation saved to {save_path}")
    return anim

#-----------can be deleted:

def numpy_kinematic_tree_to_cartesian(motion_array, joints):

    num_frames, total_dof = motion_array.shape
    motions = []
    
    # Count total degrees of freedom (DOF) for all joints
    dof_per_joint = {name: len(joint.dof) for name, joint in joints.items()}
    
    # Extract root translation and rotation from the first 6 columns
    root_dof = 6
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame_motion = {}
        
        # Extract root motion (translation + rotation)
        root_motion = motion_array[frame_idx, :root_dof]
        frame_motion['root'] = root_motion
        
        # Extract joint motions
        col_idx = root_dof
        for joint_name, joint in joints.items():
            if joint_name == 'root':  # Skip root since it's already handled
                continue
            
            dof = dof_per_joint[joint_name]
            joint_motion = motion_array[frame_idx, col_idx:col_idx + dof]
            frame_motion[joint_name] = joint_motion
            
            col_idx += dof
        
        motions.append(frame_motion)
    
    return motions

def extract_cartesian_motion_data_from_numpy(joints, motion_array, feature_names):
    """
    Extracts Cartesian motion data from a NumPy array (converted from AMC) and joint hierarchy.

    Parameters:
    - joints: Dictionary of Joint objects representing the skeleton hierarchy.
    - motion_array: NumPy array of shape (num_frames, num_features) with joint motion data.
    - feature_names: List of feature names corresponding to columns in motion_array.

    Returns:
    - motion_data: NumPy array of shape (num_frames, num_joints * 3) containing joint coordinates.
    - joint_names: List of joint names in hierarchical order.
    """
    print("motion_array.shape", motion_array.shape)
    print("len(feature_names)", len(feature_names))
    num_frames = motion_array.shape[0]
    num_joints = len(joints)
    motion_data = np.zeros((num_frames, num_joints * 3))
    
    joint_names = list(joints.keys())

    # Map feature names to indices for efficient access
    feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
    print(f"Feature index map: {feature_idx_map}")
    for frame_idx in range(num_frames):
        # Update root joint translation and rotation
        # Extract root translation and rotation using feature names
        root_translation = motion_array[frame_idx, feature_idx_map['root_tx']:feature_idx_map['root_tz'] + 1]
        root_rotation = motion_array[frame_idx, feature_idx_map['root_rx']:feature_idx_map['root_rz'] + 1]

        # Format motion for compatibility with set_motion
        root_motion = {'root': np.concatenate([root_translation, root_rotation])}
        print("root_motion", root_motion)
        joints['root'].set_motion(root_motion)
        # Update and compute coordinates for all joints
        frame_data = []
        for joint_name in joint_names:
            joint = joints[joint_name]

            # Extract the joint's degrees of freedom (DOF)
            if joint_name == 'root':
                frame_data.extend(joint.coordinate.flatten())  # Root coordinates
                continue
            
            dof_names = [f"{joint_name}_{axis}" for axis in joint.dof]
            dof_values = [motion_array[frame_idx, feature_idx_map[name]] for name in dof_names]
            joint.set_motion({'dof': dof_values})

            # Append Cartesian coordinates
            frame_data.extend(joint.coordinate.flatten())
        
        motion_data[frame_idx, :] = frame_data

    return np.array(motion_data), np.array(joint_names)

def plot_motion_dynamic_old(motion_data, joint_names, edges, frame_interval=50):
    """
    Visualizes the motion data as a 3D animation, dynamically centering and adjusting the view.

    Parameters:
    - motion_data: NumPy array of shape (num_frames, num_joints * 3) containing joint coordinates.
    - joint_names: List of joint names corresponding to the columns in motion_data.
    - edges: List of tuples representing edges (connections between joints).
    - frame_interval: Time interval (in milliseconds) between frames in the animation.
    """
    num_frames, _ = motion_data.shape
    num_joints = len(joint_names)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()  # Clear the plot for the current frame

        # Extract joint positions for the current frame
        frame_data = motion_data[frame_idx, :].reshape((num_joints, 3))
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

        ys, zs = swap_yz(ys, zs)

        # Calculate center of the skeleton for dynamic centering
        x_center, y_center, z_center = np.mean(xs), np.mean(ys), np.mean(zs)
        margin = 20  # Add a margin around the skeleton

        # Dynamically adjust axis limits
        ax.set_xlim(x_center - margin, x_center + margin)
        ax.set_ylim(y_center - margin, y_center + margin)
        ax.set_zlim(z_center - margin, z_center + margin)

        # Set a consistent viewing angle
        ax.view_init(elev=30, azim=45)

        # Plot joints
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")

        # Plot edges (connections between joints)
        for edge in edges:
            start, end = edge
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'r')

        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")

    # Assign the animation to a variable to prevent garbage collection
    anim = FuncAnimation(fig, update, frames=num_frames, interval=frame_interval)

    plt.show()

def plot_motion_dynamic_old2(motion_data, joint_names, edges, frame_interval=50):
    """
    Visualizes the motion data as a 3D animation, dynamically centering and adjusting the view.

    Parameters:
    - motion_data: NumPy array of shape (num_frames, num_joints * 3) containing joint coordinates.
    - joint_names: List of joint names corresponding to the columns in motion_data.
    - edges: List of tuples representing edges (connections between joints).
    - frame_interval: Time interval (in milliseconds) between frames in the animation.
    """
    num_frames, _ = motion_data.shape
    num_joints = len(joint_names)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()  # Clear the plot for the current frame

        # Extract joint positions for the current frame
        frame_data = motion_data[frame_idx, :].reshape((num_joints, 3))
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

        def swap_yz(ys,zs):
            ys_ = zs
            zs = ys
            ys = ys_
            return ys, zs 
        ys,zs = swap_yz(ys,zs)

        # Calculate center of the skeleton for dynamic centering
        x_center, y_center, z_center = np.mean(xs), np.mean(ys), np.mean(zs)
        margin = 20  # Add a margin around the skeleton

        # Dynamically adjust axis limits
        ax.set_xlim(x_center - margin, x_center + margin)
        ax.set_ylim(y_center - margin, y_center + margin)
        ax.set_zlim(z_center - margin, z_center + margin)

        # Set a consistent viewing angle
        ax.view_init(elev=30, azim=45)

        # Plot joints
        ax.scatter(xs, ys, zs, c='b', s=20, label="Joints")

        # Plot edges (connections between joints)
        for edge in edges:
            start, end = edge
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'r')

        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")

    anim = FuncAnimation(fig, update, frames=num_frames, interval=frame_interval)
    plt.show()