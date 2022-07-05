# Meshcat utils

Utils to use Meshcat with Pinocchio.

## Dependencies

* My [branch of Meshcat](https://github.com/ManifoldFR/meshcat-python/tree/feat/set-capture-resolution)
* tqdm | [GitHub](https://github.com/tqdm/tqdm)
* imageio with the ffmpeg plugin | [GitHub](https://github.com/imageio/imageio), [RTD](https://imageio.readthedocs.io/en/latest/)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio/)

## Install

From source, with `--no-deps` to avoid modifying your python env:

```bash
pip install . --no-deps
```

## Features

* set background color
* set camera target and position
* camera angle/target presets
* display state trajectory (position-velocity)
* record images and videos
* draw end-effector 3D trajectories & velocities
* wrappers to easily draw cylinders, bboxes
* draw objectives as spheres
* draw point clouds
* draw contact forces

## Copyright (c) 2022 LAAS-CNRS, INRIA
