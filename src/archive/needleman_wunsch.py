import numpy as np


class SimilarityMatrix:
    """
    Similarity matrix between phonetic symbols.
    Can be indexed by two phonetic symbols to get the similarity score between them.
    """

    def __init__(self, similarity_data):
        all_symbols = list(self._data_to_symbols(similarity_data))
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(all_symbols)}
        size = len(all_symbols)
        self.matrix = np.full((size, size), 0, dtype=np.int8)

        for (symbol1, symbol2), score in similarity_data.items():
            i, j = self.symbol_to_index[symbol1], self.symbol_to_index[symbol2]
            self.matrix[i, j] = int(score)
            self.matrix[j, i] = int(score)  # symmetric

    def _data_to_symbols(self, data) -> set[str]:
        symbols = set()
        for (symbol1, symbol2), _score in data.items():
            symbols.add(symbol1)
            symbols.add(symbol2)
        return symbols

    def cost(self, symbol1, symbol2) -> float:
        """
        Returns the similarity cost between two phonetic symbols.
        """
        i, j = self.symbol_to_index[symbol1], self.symbol_to_index[symbol2]
        return self.matrix[i, j]


def calculate_score(
    a: list[str], b: list[str], similarity_matrix: SimilarityMatrix, gap_penalty: float
) -> int:
    """
    Calculates the alignment score between two sequences a and b.

    Uses the Needleman-Wunsch algorithm with the given similarity matrix
    and linear gap penalty.
    """
    matrix = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int8)
    for i in range(len(a) + 1):
        matrix[i, 0] = gap_penalty * i
    for j in range(len(b) + 1):
        matrix[0, j] = gap_penalty * j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = similarity_matrix.cost(a[i - 1], b[j - 1])
            match = matrix[i - 1, j - 1] + cost
            delete = matrix[i - 1, j] + gap_penalty
            insert = matrix[i, j - 1] + gap_penalty
            matrix[i, j] = max(match, delete, insert)
    return matrix[-1, -1]
