import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from skimage.io import imread
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class SymbolsAggregator:
    @staticmethod
    def collect_symbols(
            symbol_images: List[np.ndarray], correlation_distance_threshold: float = 0.2
    ) -> List[np.ndarray]:
        symbols_shape = symbol_images[0].shape
        symbol_images = [image.reshape(1, -1) for image in symbol_images]

        print(f'Start computing correlation, number of items {len(symbol_images)}')
        s = time.time()
        cc = pairwise_distances(np.concatenate(symbol_images), metric='correlation')
        print(f'Finish computing correlation, time {time.time() - s} s')

        print(f'Start analyzing graph')
        s = time.time()
        graph = csr_matrix((cc < correlation_distance_threshold).astype(int))
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        comps = [comp for comp, n in Counter(labels).items() if n > 5]
        print(f'Finish analyzing graph, time {time.time() - s} s')

        output_symbols = []
        for symbol_index, comp in enumerate(comps):
            index = np.where(labels == comp)[0][0]
            output_symbols.append(symbol_images[index].reshape(symbols_shape))
        return output_symbols


if __name__ == '__main__':
    from utils.io import read_all_symbols, save_images_set

    symbols = read_all_symbols(Path(r'C:\Users\user\PycharmProjects\slots_scrapper\files\symbols'))
    aggregator = SymbolsAggregator()

    symbols = aggregator.collect_symbols(symbols)

    output_path = Path(r'C:\Users\user\PycharmProjects\slots_scrapper\files\agg_symbols')
    output_path.mkdir(parents=True, exist_ok=True)

    save_images_set(output_path, symbols)
