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
        self.matrix = np.full((size, size), np.nan)

        for (symbol1, symbol2), score in similarity_data.items():
            i, j = self.symbol_to_index[symbol1], self.symbol_to_index[symbol2]
            self.matrix[i, j] = score
            self.matrix[j, i] = score  # symmetric

        if np.isnan(self.matrix).any():
            raise ValueError(
                "Similarity matrix contains NaN values."
                + " Make sure to fill out all values."
            )

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


class NeedlemanWunsch:
    """
    Implementation of the Needleman-Wunsch algorithm for global sequence alignment
    score calculation. The algorithm uses a similarity matrix and a linear gap penalty.
    """

    def __init__(self, similarity_matrix: SimilarityMatrix, gap_penalty: float):
        self.similarity_matrix = similarity_matrix
        self.gap_penalty = gap_penalty

    def calculate_score(self, a: list[str], b: list[str]):
        """
        Calculates the alignment score between two sequences a and b.
        """
        matrix = np.zeros((len(a) + 1, len(b) + 1))
        for i in range(len(a) + 1):
            matrix[i, 0] = self.gap_penalty * i
        for j in range(len(b) + 1):
            matrix[0, j] = self.gap_penalty * j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = self.similarity_matrix.cost(a[i - 1], b[j - 1])
                match = matrix[i - 1, j - 1] + cost
                delete = matrix[i - 1, j] + self.gap_penalty
                insert = matrix[i, j - 1] + self.gap_penalty
                matrix[i, j] = max(match, delete, insert)
        return matrix[-1, -1]
