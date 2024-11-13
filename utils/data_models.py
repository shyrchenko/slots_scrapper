from typing import List, Tuple, Dict

import numpy as np


class ROI:
    def __init__(self, x_left: int, x_right: int, y_top: int, y_bottom: int):
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom

    def to_dict(self):
        return dict(
            x_left=self.x_left,
            x_right=self.x_right,
            y_top=self.y_top,
            y_bottom=self.y_bottom,
        )


class Vector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @property
    def coordinate(self) -> Tuple[int, int]:
        return self.x, self.y


class GridCell:
    def __init__(self, index: Vector, roi: ROI):
        self.index = index
        self.roi = roi


class CroppedSymbol:
    def __init__(self, frame: str, index: Vector, image: np.ndarray):
        self.frame = frame
        self.index = index
        self.image = image


class ExpectedSymbol:
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image

    def __repr__(self):
        return self.name


class SymbolsGrid:
    def __init__(
        self,
        start_point: Vector,
        symbol_size: Vector,
        number_of_elements: Vector,
        offset: Vector = Vector(0, 0),
    ):
        self.start_point = start_point
        self.symbol_size = symbol_size
        self.number_of_elements = number_of_elements
        self.offset = offset

    @property
    def cells(self) -> List[GridCell]:
        cells = []
        for row_index in range(self.number_of_elements.y):
            y_top = (
                self.start_point.y + (self.symbol_size.y + self.offset.y) * row_index
            )
            y_bottom = y_top + self.symbol_size.y

            for columns_index in range(self.number_of_elements.x):
                x_left = (
                    self.start_point.x
                    + (self.symbol_size.x + self.offset.x) * columns_index
                )
                x_right = x_left + self.symbol_size.x
                cells.append(
                    GridCell(
                        index=Vector(x=columns_index, y=row_index),
                        roi=ROI(
                            x_left=x_left,
                            x_right=x_right,
                            y_top=y_top,
                            y_bottom=y_bottom,
                        ),
                    )
                )
        return cells


class Reel:
    def __init__(self, frame: str, index: int, symbols: Dict[int, ExpectedSymbol]):
        self.frame = frame
        self.index = index
        self.symbols = symbols

    @classmethod
    def create_empty(cls, frame: str, index: int):
        return cls(frame=frame, index=index, symbols={})

    def add_symbol(self, index: int, symbol: ExpectedSymbol):
        self.symbols[index] = symbol

    def __repr__(self):
        return f"{self.symbols}"

    def to_dict(self):
        return {index: symbol.name for index, symbol in self.symbols.items()}
