
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D
import cv2


class Joint:
  
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None
    self.dof = dof

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self, output_file):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')
    plt.savefig(output_file)

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)

def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx

def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints

def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames

def plot_first():
  asf_path = 'allasfamc/all_asfamc/subjects/01/01.asf'
  amc_path = 'allasfamc/all_asfamc/subjects/01/01_01.amc'
  flag  = True
  if flag:
      print('parsing %s' % asf_path)
      joints = parse_asf(asf_path)
      motions = parse_amc(amc_path)
      joints['root'].set_motion(motions[0])
      joints['root'].draw()

      # for lv2 in lv2s:
      #   if lv2.split('.')[-1] != 'amc':
      #     continue
      #   amc_path = '%s/%s/%s' % (lv0, lv1, lv2)
      #   print('parsing amc %s' % amc_path)
      #   motions = parse_amc(amc_path)
      #   for idx, motion in enumerate(motions):
      #     print('setting motion %d' % idx)
      #     joints['root'].set_motion(motion)

def render_frame():
        """Render the current state of the skeleton into a frame."""
        ax.cla()
        ax.set_xlim3d(-50, 10)
        ax.set_ylim3d(-20, 40)
        ax.set_zlim3d(-20, 40)

        # Plot joints and connections
        joints = self.to_dict()
        xs, ys, zs = [], [], []
        for joint in joints.values():
            xs.append(joint.coordinate[0, 0])
            ys.append(joint.coordinate[1, 0])
            zs.append(joint.coordinate[2, 0])
        ax.plot(zs, xs, ys, 'b.')  # Joints as blue dots

        for joint in joints.values():
            child = joint
            if child.parent is not None:
                parent = child.parent
                xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
                ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
                zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
                ax.plot(zs, xs, ys, 'r')  # Connections as red lines

        # Convert rendered figure to a video frame
        fig.canvas.draw()
        frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return frame_image

def visualize_motion(joints, motion_frames, output_path):

    #Initialize 3D figure for plotting
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Video writer setup
    frame_width, frame_height = 640, 480
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps = 20, frameSize=frame_size)

    def render_frame():
      """Render the current state of the skeleton into a frame with dynamic camera movement."""
      ax.cla()

      # Gather joint positions to determine dynamic bounds
      xs, ys, zs = [], [], []
      for joint in joints.values():
          xs.append(joint.coordinate[0, 0])
          ys.append(joint.coordinate[1, 0])
          zs.append(joint.coordinate[2, 0])

      # Calculate the center of the skeleton
      x_center = np.mean(xs)
      y_center = np.mean(ys)
      z_center = np.mean(zs)

      # Define a margin around the skeleton
      margin = 40  # Adjust as needed
      ax.set_xlim3d(x_center - margin, x_center + margin)
      ax.set_ylim3d(y_center - margin, y_center + margin)
      ax.set_zlim3d(z_center - margin, z_center + margin)

      # Plot joints as blue dots
      ax.plot(zs, xs, ys, 'b.')

      # Plot connections between joints as red lines
      for joint in joints.values():
          child = joint
          if child.parent is not None:
              parent = child.parent
              xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
              ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
              zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
              ax.plot(zs, xs, ys, 'r')

      # Convert rendered figure to a video frame
      fig.canvas.draw()
      frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      return frame_image

    for frame_idx, motion_frame in enumerate(motion_frames):
        # Update joint positions for the current frame
        #print("frame_idx",frame_idx)
        joints['root'].set_motion(motion_frame)

        # Render the current frame
        frame_image = render_frame()

        # Convert frame to BGR format for OpenCV
        frame_image_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)

        # Resize frame to video resolution
        frame_resized = cv2.resize(frame_image_bgr, (frame_width, frame_height))

        # Write frame to video
        video_writer.write(frame_resized)
        #print(f"Processed frame {frame_idx + 1}/{len(motion_frames)}")
