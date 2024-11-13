import numpy as np
from skimage.util import view_as_windows


def _corr(img1, img2):
    r = np.corrcoef(img1, img2)
    return r[0][1]


_corr = np.vectorize(_corr, signature="(n), (n) ->()")


def window_correlation(
    img1: np.ndarray, img2: np.ndarray, windows_part: float = 0.1, windows_step: int = 3
) -> float:
    x_thresh = int(img2.shape[1] * windows_part)
    y_thresh = int(img2.shape[0] * windows_part)

    img2 = img2[y_thresh:-y_thresh, x_thresh:-x_thresh, 0]
    img1 = img1[:, :, 0]
    windows = view_as_windows(img1, img2.shape, step=windows_step)

    number_of_windows = windows.shape[0] * windows.shape[1]
    windows = windows.reshape((number_of_windows, -1))
    img2 = img2.flatten()

    cc = _corr(windows, img2)
    result = max(cc)
    return result


if __name__ == "__main__":
    img1 = np.random.randn(100, 100, 3)
    img2 = np.random.randn(100, 100, 3)
    print(window_correlation(img1, img2))
