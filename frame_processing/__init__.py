from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from frame_processing.frames_extraction.frame_extractor import FramesExtractor
from frame_processing.symbols_identification.symbols_identifier import (
    SymbolsProcessor,
    CorrSymbolIdentifier,
)
from frame_processing.symbols_images_extraction.symbols_images_extractor import (
    SymbolsImagesExtractor,
)
from utils.data_models import ROI, SymbolsGrid, Reel


def process_video(
    video_path: Path,
    frame_extractor: FramesExtractor,
    roi: ROI,
    grid: SymbolsGrid,
    symbols_extractor: SymbolsImagesExtractor,
    reels_processor: SymbolsProcessor,
) -> Dict[str, Optional[List[Reel]]]:
    result = {}
    for index, frame in enumerate(frame_extractor.extract_frames(video_path, roi)):
        frame_name = f"frame_{index}"
        symbols = symbols_extractor.extract_symbols(
            frame=frame_name, frame_image=frame, grid=grid
        )
        processed = reels_processor.process_frames_symbols(symbols)
        if processed is not None:
            result[frame_name] = [reel.to_dict() for reel in processed]
        else:
            result[frame_name] = processed
    return result


def process_frame(
    frame_name: str,
    frame_image: np.ndarray,
    symbols_extractor: SymbolsImagesExtractor,
    reels_processor: SymbolsProcessor,
    grid: SymbolsGrid,
) -> Optional[List[dict]]:
    symbols = symbols_extractor.extract_symbols(
        frame=frame_name, frame_image=frame_image, grid=grid
    )
    processed = reels_processor.process_frames_symbols(symbols)
    if processed is not None:
        processed = [reel.to_dict() for reel in processed]
    return processed
