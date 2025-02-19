from dataclasses import dataclass


@dataclass
class WordData:
    """
    Data class for a word from the French language.

    For more information on the fields, see:
    https://github.com/frodonh/french-words#structure-of-the-file

    - The POS-tagging is using the Lexique.org nomenclature.
    - The frequencies are measured in number of occurrences per million of words.
      They come from Lexique.org and Google Ngrams. If a frequency is not provided,
      it is set to -1.0.
    """

    word: str
    pos: list[str]
    lemma: str
    frequency_lexique: float
    frequency_ngrams: float
    phonetics: list[str]
