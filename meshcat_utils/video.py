import imageio.v2 as iio
from .presets import VIDEO_CONFIG_DEFAULT, VIDEO_CONFIGS


class VideoRecorder:

    def __init__(self, uri: str, fps: float, config=VIDEO_CONFIG_DEFAULT):
        self.writer = iio.get_writer(uri, fps=fps, **config)

    def __call__(self, img):
        self.writer.append_data(img)

    def __del__(self):
        self.writer.close()
        del self
