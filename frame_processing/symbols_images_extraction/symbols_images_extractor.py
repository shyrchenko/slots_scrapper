import uuid
from pathlib import Path
from typing import List

import numpy as np

from utils.image_processing import crop_image
from utils.data_models import SymbolsGrid, CroppedSymbol
from utils.io import save_images_set
from utils.state import get_debug_dir, is_debug_mode_activated


class SymbolsImagesExtractor:
    @staticmethod
    def extract_symbols(
        frame: str, frame_image: np.ndarray, grid: SymbolsGrid
    ) -> List[CroppedSymbol]:
        symbols = []
        for grid_cell in grid.cells:
            symbols.append(
                CroppedSymbol(
                    index=grid_cell.index,
                    image=crop_image(img=frame_image, roi=grid_cell.roi),
                    frame=frame,
                )
            )
        if is_debug_mode_activated():
            debug_path = Path(f"{get_debug_dir()}/{frame}")
            debug_path.mkdir(exist_ok=True, parents=True)
            save_images_set(debug_path, [s.image for s in symbols])
        return symbols
