# Meshcat utils

Meshcat utils to use with Pinocchio.

## Dependencies

* meshcat (my [branch](https://github.com/ManifoldFR/meshcat-python/tree/feat/set-capture-resolution) before it's merged)
* Pinocchio

## Install

From source, with `--no-deps` to avoid modifying your python env:

```bash
pip install . --no-deps
```

## Features

* set background
* camera angle/target presets
* display state trajectory (position-velocity)
* record images
* draw end-effector trajectories in 3D
* draw end-effector 3D velocities
* wrappers to easily draw cylinders, bboxes
* draw objectives as spheres
* draw point clouds
* draw contact forces
