from .data_models import ROI
import numpy as np


def crop_image(img: np.ndarray, roi: ROI) -> np.ndarray:
    return img[roi.y_top : roi.y_bottom, roi.x_left : roi.x_right, :]
