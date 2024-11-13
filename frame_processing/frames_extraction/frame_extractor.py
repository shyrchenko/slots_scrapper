from pathlib import Path
from typing import Iterator
import cv2
import numpy as np

from utils.data_models import ROI
from utils.image_processing import crop_image


class FramesExtractor:
    CORR_THRESH = 0.9

    def __init__(self, skip_frames: int = 5, similar_frames_needed: int = 3):
        self._skip_frames = skip_frames
        self._similar_frames_needed = similar_frames_needed

    def extract_frames(self, video_path: Path, roi: ROI) -> Iterator[np.ndarray]:
        frames = []
        video = cv2.VideoCapture(str(video_path))

        _, prev_frame = video.read()
        prev_frame = crop_image(prev_frame, roi)
        similar_frames = 0
        counter = 0

        while True:
            _, curr_frame = video.read()
            if curr_frame is None:
                break

            curr_frame = crop_image(curr_frame, roi)
            counter += 1
            if counter < self._skip_frames:
                continue
            else:
                counter = 0

            if not self._is_frames_similar(curr_frame, prev_frame):
                if similar_frames > self._similar_frames_needed:
                    yield prev_frame
                similar_frames = 0
            else:
                similar_frames += 1
            prev_frame = curr_frame

    def _is_frames_similar(self, frame_1: np.ndarray, frame_2: np.ndarray) -> bool:
        return np.corrcoef(frame_1.flatten(), frame_2.flatten())[0][1] > self.CORR_THRESH
