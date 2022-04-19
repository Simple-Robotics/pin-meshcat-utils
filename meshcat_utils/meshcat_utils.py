import meshcat
import numpy as np
import pinocchio as pin

import meshcat.transformations as tf
import meshcat.geometry as g

from pinocchio.visualize import MeshcatVisualizer
from typing import List, Union


def set_bg(viewer):
    """Set the background."""
    col1 = [.98, .98, .98]
    viewer["/Background"].set_property("top_color", col1)
    col2 = [.8, .8, .8]
    viewer["/Background"].set_property("bottom_color", col2)


CAM_PRESETS = {
    0: [
        np.zeros(3),  # target
        [3., 1., 0.]  # anchor point (x, z, -y) lhs coords
    ],
    1: [
        np.zeros(3),
        [1., 1., -1.]
    ],
    2: [
        [0., 0.6, 0.],
        [0.8, 1.2, -1.]
    ],
    'acrobot': [
        [0., 0.1, 0.],
        [.5, 0.1, 0.]
    ],
    'cam_ur': [
        [0.4, 0.6, -0.2],
        [1., 1.2, -0.4]
    ],
    'cam_ur2': [
        [0.4, 0.3, 0.],
        [0.5, 1.4, -0.1]
    ],
    'cam_ur3': [
        [0.4, 0.3, 0.],
        [0.6, 1.3, 0.3]
    ],
    'cam_ur4': [   # x>0 to x<0
        [-1., 0.3, 0.],
        [1.3, 1.2, -0.1]
    ],
    'cam_ur5': [
        [-1., 0.3, 0.],
        [-0.05, 1.2, -1.5]
    ],
    'talos': [
        [0., 1.2, 0.],
        [1.5, 1.5, -0.3]
    ],
    'talos2': [
        [0., 1.1, 0.],
        [1.2, 1.5, -0.6]
    ]
}


def _set_cam_angle(viewer, i):
    """Set the camera angle and target, from a set of presets: """
    tar = CAM_PRESETS[i][0]
    viewer.set_cam_target(tar)
    pos = CAM_PRESETS[i][1]
    path2 = "/Cameras/default/rotated/<object>"
    viewer[path2].set_property("position", pos)


_set_cam_angle.__doc__ += str(list(CAM_PRESETS.keys()))


def display_trajectory(vizer: MeshcatVisualizer,
                       drawer: "ForceDraw", xs, us=None,
                       extra_pts=None, frame_ids=[], record=False,
                       wait: float = None, show_vel: bool = False,
                       progress_bar=True,
                       frame_sphere_size=0.03,
                       record_kwargs={}):
    import time
    import tqdm
    import warnings
    if len(frame_ids) == 0 and show_vel:
        warnings.warn("asked to show frame velocity, but frame_ids is empty!")

    if extra_pts is not None:
        extra_pts = np.asarray(extra_pts)
        if extra_pts.ndim == 2:
            extra_pts = extra_pts.reshape(1, *extra_pts.shape)
        n_p = extra_pts.shape[0]
    else:
        n_p = 0

    rmodel = vizer.model
    rdata = vizer.data
    drawer.rdata = rdata
    nq = rmodel.nq
    nv = rmodel.nv
    nsteps = len(xs) - 1
    images = []
    trange = tqdm.trange(nsteps + 1, disable=not progress_bar)
    drawer.clear_trajectories()
    for t in trange:
        q = xs[t][:nq]
        v = xs[t][nq:nq + nv]
        pin.forwardKinematics(rmodel, rdata, q, v)
        vizer.display()
        pin.updateFramePlacements(rmodel, rdata)

        # plot input frame ids
        for fid in frame_ids:
            if show_vel:
                drawer.draw_frame_vel(fid)
            pt = rdata.oMf[fid].translation.copy()
            drawer.update_trajectory(pt, prefix=f'traj_fid{fid}')
            drawer.draw_objective(pt, prefix=f'ee_{fid}', color=0xF0FF33, size=frame_sphere_size, opacity=0.3)

        # plot other points
        for j in range(n_p):
            drawer.update_trajectory(extra_pts[j, t], prefix=f'traj_extra{j}')
            drawer.draw_objective(extra_pts[j, t], prefix=f'extra_{j}', color=0x2DFB6F)

        if record:
            width = record_kwargs.get('w', None)
            height = record_kwargs.get('h', None)
            img = np.asarray(vizer.viewer.get_image(width, height))
            images.append(img)
        if wait is not None:
            time.sleep(wait)
    return images


class ForceDraw:
    """Utility class extending base capability of pinocchio's MeshcatVisualizer."""

    def __init__(self, viewer: meshcat.Visualizer, rmodel: pin.Model, rdata: pin.Data):
        from collections import defaultdict
        self.rmodel = rmodel
        self.rdata = rdata
        self.viewer = viewer
        self._traj = defaultdict(list)  # trajectory buffer

    def set_bg(self):
        set_bg(self.viewer)

    def set_cam(self, i):
        _set_cam_angle(self.viewer, i)

    def draw_objective(self, target, prefix='target', color=None, size=0.05, opacity=0.5):
        sp = g.Sphere(radius=size)
        color = color or 0xc9002f
        self.viewer[prefix].set_object(sp, g.MeshLambertMaterial(color=color, opacity=opacity))
        self.viewer[prefix].set_transform(tf.translation_matrix(direction=target))

    def draw_cylinder(self, base_point, radius, height=1., prefix='cyl'):
        """Input base_point should be in xyz."""
        color = 0xFF33CA
        opacity = 0.5
        sp = g.Cylinder(height, radius)
        base_point = np.asarray(base_point) + (0, 0, height / 2)
        self.viewer[prefix].set_object(sp, g.MeshLambertMaterial(color=color, opacity=opacity))
        Tr = tf.translation_matrix(direction=base_point)
        R1 = tf.euler_matrix(np.pi / 2, 0., 0.)
        self.viewer[prefix].set_transform(Tr @ R1)

    def draw_bbox(self, lb, ub):
        color = 0x7fcb85
        opacity = 0.18
        prefix = 'bbox'
        lengths = ub - lb
        center = (ub + lb) * .5
        box = g.Box(lengths)
        wf_length = 1.2
        self.viewer[prefix].set_object(box, g.MeshLambertMaterial(color=color, opacity=opacity, wireframe=False, wireframeLinewidth=wf_length))
        tr = tf.translation_matrix(center)
        self.viewer[prefix].set_transform(tr)

    def draw_objectives(self, targets: Union[list, dict], prefix='', color=None, size=0.08, opacity=0.5):
        """Draw multiple objective waypoints."""
        if not isinstance(targets, dict):
            targets = {i: t for i, t in enumerate(targets)}
        for i, tgt in targets.items():
            self.draw_objective(tgt, f'{prefix}/#{i}', color, size, opacity=opacity)

    def draw_point_cloud(self, points, colors, prefix="points"):
        """Point coordinates are given batch-first."""
        points = np.asarray(points).T
        colors = np.asarray(colors).T
        self.viewer[prefix].set_object(g.PointCloud(points, colors, size=0.008))

    def draw_frame_vel(self, frame_id, color=0x00FF00):
        """Call `forwardKinematics` beforehand with velocities."""
        pl = self.rdata.oMf[frame_id].translation.copy()
        vFr = pin.getFrameVelocity(self.rmodel, self.rdata, frame_id, pin.LOCAL_WORLD_ALIGNED)
        v_scale = 0.2
        vtx = np.array([pl, pl + v_scale * vFr.linear]).T
        geom = g.PointsGeometry(vtx)
        go = g.LineSegments(geom, g.LineBasicMaterial(color=color))
        prefix = f'lines/ee_v/{frame_id}'
        self.viewer[prefix].set_object(go)

    def clear_trajectories(self):
        """Empty the trajectory buffer."""
        self._traj.clear()

    def update_trajectory(self, pt: np.ndarray, prefix='ee_traj'):
        self._traj[prefix].append(pt)
        self.draw_trajectory(self._traj[prefix], prefix=prefix)

    def draw_trajectory(self, pts: np.ndarray, prefix='ee_traj'):
        """Add the input `p` to a drawn trajectory."""
        pts = np.asarray(pts)
        segs = np.asarray(pts).T
        color = 0xEE1100
        mat = g.MeshLambertMaterial(color=color)
        # gobj = g.Points(geom, mat)
        gobj = g.LineSegments(g.PointsGeometry(segs), mat)
        self.viewer[prefix].set_object(gobj)

    def draw_contact_forces(self, q, forces: List[np.ndarray], frame_ids: List[int]):
        SCALE = 0.06
        pin.updateFramePlacements(self.rmodel, self.rdata)
        COLOR = 0x00FF11
        anchors = []
        for i, fid in enumerate(frame_ids):
            p = self.rdata.oMf[fid].translation.copy()
            anchors.append(p)
            fi = forces[i]
            seg = np.array([p, p + SCALE * fi]).T
            geom = g.PointsGeometry(seg)
            go = g.LineSegments(geom, g.LineBasicMaterial(color=COLOR))

            prefix = f'lines/contact/force{i}'
            self.viewer[prefix].set_object(go)

            sp = g.Sphere(0.01)
            matrl = g.MeshLambertMaterial(color=COLOR, opacity=0.8)
            self.viewer[f'{prefix}/sphere'].set_object(sp, matrl)
            self.viewer[f'{prefix}/sphere'].set_transform(tf.translation_matrix(p))

    @staticmethod
    def to_hex(col):
        from matplotlib.colors import to_hex
        return int(to_hex(col)[1:], 16)
