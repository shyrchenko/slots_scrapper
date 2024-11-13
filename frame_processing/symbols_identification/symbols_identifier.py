import time
from typing import Optional, List

from utils.data_models import ExpectedSymbol, CroppedSymbol, Reel, Vector
from utils.custom_metrics import window_correlation
from utils.logger import logger
from abc import abstractmethod


class BaseSymbolIdentifier:
    @abstractmethod
    def identify_symbol(self, symbol: CroppedSymbol):
        pass


class CorrSymbolIdentifier(BaseSymbolIdentifier):
    METRIC_THRESHOLD = 0.7

    def __init__(self, expected_symbols: List[ExpectedSymbol]):
        self._expected_symbols = expected_symbols

    def identify_symbol(self, symbol: CroppedSymbol) -> Optional[ExpectedSymbol]:
        s = time.time()
        for index, expected_symbol in enumerate(self._expected_symbols):
            score = window_correlation(symbol.image, expected_symbol.image)
            if score > self.METRIC_THRESHOLD:
                logger.debug(
                    f"For element {symbol.index.coordinate} identified {expected_symbol.name}. "
                    f"Processed {index + 1} expected symbols. Time: {time.time() - s}"
                )
                return expected_symbol
        else:
            logger.debug(
                f"For element {symbol.index.coordinate} not identified expected_symbol. "
                f"Processed {len(self._expected_symbols)} expected symbols. Time: {time.time() - s}"
            )


class SymbolsProcessor:
    def __init__(self, symbol_identifier: BaseSymbolIdentifier):
        self._symbol_identifier = symbol_identifier

    def process_frames_symbols(
        self, symbols: List[CroppedSymbol]
    ) -> Optional[List[Reel]]:
        g_s = time.time()
        current_frame = symbols[0].frame

        if any(symbol.frame != current_frame for symbol in symbols):
            raise ValueError("All symbols should be from one frame")

        grid_size = Vector(
            x=max(symbol.index.x for symbol in symbols) + 1,
            y=max(symbol.index.y for symbol in symbols) + 1,
        )
        reels = [
            Reel.create_empty(frame=current_frame, index=index)
            for index in range(grid_size.x)
        ]
        for symbol in symbols:
            expected_symbol = self._symbol_identifier.identify_symbol(symbol)
            if expected_symbol is None:
                logger.info(
                    f"In frame {current_frame} symbol with index {symbol.index.coordinate} is not detected. "
                    f"Frame is invalid."
                )
                return None
            reels[symbol.index.x].add_symbol(symbol.index.y, expected_symbol)
        logger.debug(f"Time of processing one frame: {time.time() - g_s}")
        return reels


if __name__ == "__main__":
    from utils.io import read_expected_symbols, read_cropped_symbols
    from pathlib import Path

    identifier = CorrSymbolIdentifier(
        read_expected_symbols(
            Path(r"C:\Users\user\PycharmProjects\slots_scrapper\files\agg_symbols")
        )
    )
    processor = SymbolsProcessor(identifier)
    symbols = read_cropped_symbols(
        Path(r"C:\Users\user\PycharmProjects\slots_scrapper\files\symbols\frame_0")
    )

    r = processor.process_frames_symbols(symbols)
    print(r)
