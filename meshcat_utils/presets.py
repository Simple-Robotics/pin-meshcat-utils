import numpy as np


CAMERA_PRESETS = {
    "preset0": [
        np.zeros(3),  # target
        [3., 0., 1.]  # anchor point (x, z, -y) lhs coords
    ],
    "preset1": [
        np.zeros(3),
        [1., 1., 1.]
    ],
    "preset2": [
        [0., 0., 0.6],
        [0.8, 1., 1.2]
    ],
    'acrobot': [
        [0., 0.1, 0.],
        [.5, 0., 0.2]
    ],
    'cam_ur': [
        [0.4, 0.6, -0.2],
        [1., 0.4, 1.2]
    ],
    'cam_ur2': [
        [0.4, 0.3, 0.],
        [0.5, 0.1, 1.4]
    ],
    'cam_ur3': [
        [0.4, 0.3, 0.],
        [0.6, 1.3, 0.3]
    ],
    'cam_ur4': [   # x>0 to x<0
        [-1., 0.3, 0.],
        [1.3, 0.1, 1.2]
    ],
    'cam_ur5': [
        [-1., 0.3, 0.],
        [-0.05, 1.5, 1.2]
    ],
    'talos': [
        [0., 1.2, 0.],
        [1.5, 0.3, 1.5]
    ],
    'talos2': [
        [0., 1.1, 0.],
        [1.2, 0.6, 1.5]
    ]
}

VIDEO_CONFIG_DEFAULT = {
    "codec": "libx264",
    "macro_block_size": 8,
    "output_params": ["-crf", "17"]
}

VIDEO_CONFIGS = {
    "default": VIDEO_CONFIG_DEFAULT,
    "x265": {
        "codec": "libx265",
        "macro_block_size": 8,
        "output_params": ["-crf", "23"]
    }
}
