import numpy as np
import pinocchio as pin

import meshcat.transformations as mtf
import meshcat.geometry as mgeom

from pinocchio.visualize import MeshcatVisualizer
from typing import List, Union, Callable
from .presets import CAMERA_PRESETS, COLOR_PROFILES
from .video import VideoRecorder


def set_background_color(viewer, profile: str):
    """Set the background."""
    assert profile in COLOR_PROFILES.keys()
    col_top, col_bot = COLOR_PROFILES[profile]
    viewer["/Background"].set_property("top_color", col_top)
    viewer["/Background"].set_property("bottom_color", col_bot)


class VizUtil:
    """Utility class extending Pinocchio's MeshcatVisualizer.

    Implements:
    * drawing objectives
    * drawing cylinders
    * drawing spheres
    * drawing lines for velocities or forces
    """

    FORCE_SCALE = 0.06

    def __init__(self, vizer: pin.visualize.MeshcatVisualizer):
        from collections import defaultdict

        self.vizer = vizer
        self.rmodel = self.vizer.model
        self.rdata = self.vizer.data
        self.viewer = self.vizer.viewer
        self._traj = defaultdict(list)  # trajectory buffer

    def set_bg_color(self, profile: str = "gray"):
        set_background_color(self.viewer, profile)

    def set_cam_angle_preset(self, key: str):
        """Set the camera angle and target, from a set of presets."""
        assert key in CAMERA_PRESETS.keys()
        self.set_cam_target(CAMERA_PRESETS[key][0])
        self.set_cam_pos(CAMERA_PRESETS[key][1])

    def set_cam_target(self, tar):
        self.viewer.set_cam_target(tar)

    def set_cam_pos(self, pos):
        """Set camera position."""
        self.viewer.set_cam_pos(pos)

    def draw_objective(
        self, target, prefix="target", color=None, size=0.05, opacity=0.5
    ):
        sp = mgeom.Sphere(radius=size)
        color = color or 0xC9002F
        self.viewer[prefix].set_object(
            sp, mgeom.MeshLambertMaterial(color=color, opacity=opacity)
        )
        self.viewer[prefix].set_transform(mtf.translation_matrix(direction=target))

    def draw_cylinder(self, base_point, radius, height=1.0, prefix="cyl"):
        """Input base_point should be in xyz."""
        color = 0xFF33CA
        opacity = 0.5
        sp = mgeom.Cylinder(height, radius)
        base_point = np.asarray(base_point) + (0, 0, height / 2)
        self.viewer[prefix].set_object(
            sp, mgeom.MeshLambertMaterial(color=color, opacity=opacity)
        )
        Tr = mtf.translation_matrix(direction=base_point)
        R1 = mtf.euler_matrix(np.pi / 2, 0.0, 0.0)
        self.viewer[prefix].set_transform(Tr @ R1)

    def draw_bbox(self, lb, ub):
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        color = 0x7FCB85
        opacity = 0.18
        prefix = "bbox"
        lengths = ub - lb
        center = (ub + lb) * 0.5
        box = mgeom.Box(lengths)
        wf_length = 1.2
        self.viewer[prefix].set_object(
            box,
            mgeom.MeshLambertMaterial(
                color=color,
                opacity=opacity,
                wireframe=False,
                wireframeLinewidth=wf_length,
            ),
        )
        tr = mtf.translation_matrix(center)
        self.viewer[prefix].set_transform(tr)

    def draw_plane(
        self, width=1, height=1, color=0xCFE0FF, opacity=0.4, transform=np.eye(4)
    ):
        self._plane = mgeom.Plane(width, height)
        material = mgeom.MeshLambertMaterial(color, opacity)
        self.viewer["/plane"].set_object(self._plane, material)
        self.viewer["/plane"].set_transform(transform)

    def draw_objectives(
        self, targets: Union[list, dict], prefix="", color=None, size=0.08, opacity=0.5
    ):
        """Draw multiple objective waypoints."""
        if not isinstance(targets, dict):
            targets = {i: t for i, t in enumerate(targets)}
        for i, tgt in targets.items():
            self.draw_objective(tgt, f"{prefix}/#{i}", color, size, opacity=opacity)

    def draw_point_cloud(self, points, colors, prefix="points"):
        """Point coordinates are given batch-first."""
        points = np.asarray(points).T
        colors = np.asarray(colors).T
        self.viewer[prefix].set_object(mgeom.PointCloud(points, colors, size=0.008))

    def draw_frame_vel(self, frame_id, color=0x00FF00):
        """Call `forwardKinematics` beforehand with velocities."""
        pl = self.rdata.oMf[frame_id].translation.copy()
        vFr = pin.getFrameVelocity(
            self.rmodel, self.rdata, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        v_scale = 0.2
        vtx = np.array([pl, pl + v_scale * vFr.linear]).T
        geom = mgeom.PointsGeometry(vtx)
        go = mgeom.LineSegments(geom, mgeom.LineBasicMaterial(color=color))
        prefix = f"lines/ee_v/{frame_id}"
        self.viewer[prefix].set_object(go)

    def clear_trajectories(self):
        """Empty the trajectory buffer."""
        self._traj.clear()

    def update_trajectory(self, pt: np.ndarray, prefix="ee_traj"):
        self._traj[prefix].append(pt)
        self.draw_trajectory(self._traj[prefix], prefix=prefix)

    def draw_trajectory(self, pts: np.ndarray, prefix="ee_traj"):
        """Add the input `p` to a drawn trajectory."""
        pts = np.asarray(pts)
        segs = np.asarray(pts).T
        color = 0xEE1100
        mat = mgeom.MeshLambertMaterial(color=color)
        # gobj = g.Points(geom, mat)
        gobj = mgeom.LineSegments(mgeom.PointsGeometry(segs), mat)
        self.viewer[prefix].set_object(gobj)

    def draw_contact_forces(self, q, forces: List[np.ndarray], frame_ids: List[int]):
        pin.updateFramePlacements(self.rmodel, self.rdata)
        COLOR = 0x00FF11
        anchors = []
        for i, fid in enumerate(frame_ids):
            p = self.rdata.oMf[fid].translation.copy()
            anchors.append(p)
            fi = forces[i]
            seg = np.array([p, p + self.FORCE_SCALE * fi]).T
            geom = mgeom.PointsGeometry(seg)
            go = mgeom.LineSegments(geom, mgeom.LineBasicMaterial(color=COLOR))

            prefix = f"lines/contact/force{i}"
            self.viewer[prefix].set_object(go)

            sp = mgeom.Sphere(0.01)
            matrl = mgeom.MeshLambertMaterial(color=COLOR, opacity=0.8)
            self.viewer[f"{prefix}/sphere"].set_object(sp, matrl)
            self.viewer[f"{prefix}/sphere"].set_transform(mtf.translation_matrix(p))

    @staticmethod
    def to_hex(col):
        from matplotlib.colors import to_hex

        return int(to_hex(col)[1:], 16)

    def play_trajectory(
        self,
        xs: List[np.ndarray],
        us: List[np.ndarray] = None,
        extra_pts=None,
        frame_ids: List[int] = [],
        timestep: float = None,
        show_vel=False,
        progress_bar=True,
        frame_sphere_size=0.03,
        frame_sphere_color=0xF0FF33,
        record=False,
        record_kwargs=None,
        recorder: VideoRecorder = None,
        post_callback: Callable = None,
    ):
        """Display a state trajectory.

        :param vizer: the underlying meshcat visualizer
        :param xs:  states
        :param us:  controls
        """
        import time
        import tqdm
        import warnings

        if recorder is not None:
            record = True
        if record and record_kwargs is None:
            if timestep:
                fps = 1.0 / timestep
            else:
                fps = 30
            record_kwargs = {"uri": "meshcat_video.mp4", "fps": fps}
        if len(frame_ids) == 0 and show_vel:
            warnings.warn(
                "Asked to show frame velocity, but no frame IDs were supplied."
            )

        if extra_pts is not None:
            extra_pts = np.asarray(extra_pts)
            if extra_pts.ndim == 2:
                extra_pts = extra_pts.reshape(1, *extra_pts.shape)
            num_pts = extra_pts.shape[0]
        else:
            num_pts = 0

        if record:
            if recorder is None:
                warnings.warn("Default-constructing VideoRecorder instance.")
                recorder = VideoRecorder(
                    uri=record_kwargs["uri"], fps=record_kwargs["fps"]
                )
        else:
            recorder = None

        rmodel = self.rmodel
        rdata = self.rdata
        nq = rmodel.nq
        nv = rmodel.nv
        nsteps = len(xs) - 1
        trange = tqdm.trange(nsteps + 1, disable=not progress_bar)

        self.clear_trajectories()
        for t in trange:
            q = xs[t][:nq]
            v = xs[t][nq : nq + nv]
            pin.forwardKinematics(rmodel, rdata, q, v)
            self.vizer.display()
            pin.updateFramePlacements(rmodel, rdata)

            # plot input frame ids
            for fid in frame_ids:
                if show_vel:
                    self.draw_frame_vel(fid)
                pt = rdata.oMf[fid].translation.copy()
                self.update_trajectory(pt, prefix=f"traj_fid{fid}")
                self.draw_objective(
                    pt,
                    prefix=f"ee_{fid}",
                    color=frame_sphere_color,
                    size=frame_sphere_size,
                    opacity=0.1,
                )

            # plot other points
            for j in range(num_pts):
                self.update_trajectory(extra_pts[j, t], prefix=f"traj_extra{j}")
                self.draw_objective(
                    extra_pts[j, t], prefix=f"extra_{j}", color=0x2DFB6F
                )

            if post_callback is not None:
                post_callback(t)

            if record:
                width = record_kwargs.get("w", None)
                height = record_kwargs.get("h", None)
                img = np.asarray(self.viewer.get_image(width, height))
                recorder(img)

            if timestep is not None:
                time.sleep(timestep)


def play_trajectory(
    vizer: MeshcatVisualizer,
    xs: List[np.ndarray],
    us: List[np.ndarray] = None,
    viz_util: VizUtil = None,
    extra_pts=None,
    frame_ids: List[int] = [],
    record: bool = False,
    timestep: float = None,
    show_vel: bool = False,
    progress_bar=True,
    frame_sphere_size=0.03,
    frame_sphere_color=0xF0FF33,
    record_kwargs=None,
    recorder: VideoRecorder = None,
    post_callback: Callable = None,
):
    if viz_util is None:
        viz_util = VizUtil(vizer)
    viz_util.play_trajectory(
        xs,
        us,
        extra_pts=extra_pts,
        frame_ids=frame_ids,
        record=record,
        timestep=timestep,
        show_vel=show_vel,
        progress_bar=progress_bar,
        frame_sphere_size=frame_sphere_size,
        frame_sphere_color=frame_sphere_color,
        record_kwargs=record_kwargs,
        recorder=recorder,
        post_callback=post_callback,
    )


__all__ = ["VizUtil", "play_trajectory", "set_background_color"]
