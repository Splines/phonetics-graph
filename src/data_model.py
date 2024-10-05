from dataclasses import dataclass


@dataclass
class WordIpaChars:
    word: str
    ipa: str
    ipa_chars: list[str]
