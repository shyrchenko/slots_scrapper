from pathlib import Path
from typing import List

import numpy as np
from skimage.io import imsave, imread

from utils.data_models import ExpectedSymbol, CroppedSymbol, Vector


def save_symbols(output_folder: Path, symbols: List[CroppedSymbol]):
    for symbol in symbols:
        filename = f'symbol_{symbol.index.x}_{symbol.index.y}.png'
        filepath = output_folder / filename
        imsave(filepath, symbol.image)
        print(f'Saved image {filepath}')


def read_all_symbols(symbols_folder: Path) -> List[np.ndarray]:
    symbols = []
    for frame_folder in symbols_folder.iterdir():
        for symbol_path in frame_folder.iterdir():
            symbols.append(imread(symbol_path))
    return symbols


def save_images_set(output_folder: Path, images: List[np.ndarray]):
    for index, image in enumerate(images):
        filename = f'image_{index}.png'
        filepath = output_folder / filename
        imsave(filepath, image)


def read_expected_symbols(symbols_folder: Path) -> List[ExpectedSymbol]:
    symbols = []
    for symbol_path in symbols_folder.iterdir():
        symbols.append(ExpectedSymbol(
            name=symbol_path.name.replace('.png', ''),
            image=imread(symbol_path)
        ))
    return symbols


def read_cropped_symbols(frame_symbols_folder: Path) -> List[CroppedSymbol]:
    symbols = []
    for symbol_path in frame_symbols_folder.iterdir():
        symbols.append(CroppedSymbol(
            frame=frame_symbols_folder.name,
            index=Vector(*[int(x) for x in symbol_path.name.replace('symbol_', '').replace('.png', '').split("_")]),
            image=imread(symbol_path)
        ))
    return symbols
