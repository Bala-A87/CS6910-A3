from typing import Tuple, List
import torch

class Alphabet():
    def __init__(self) -> None:
        self.letter_to_index = {}
        self.index_to_letter = ['SOW', 'EOW', 'UNK']
        self.letter_count = 3
    
    def addLetter(self, letter: str) -> None:
        if letter not in self.letter_to_index:
            self.letter_to_index[letter] = self.letter_count
            self.index_to_letter.append(letter)
            self.letter_count += 1
    
    def getLetterIndex(self, letter: str) -> None:
        if letter in self.letter_to_index:
            return self.letter_to_index[letter]
        else:
            return 2

def get_data(language: str = 'tam', split: str = 'train') -> Tuple[List[str], List[str]]:
    with open(f'data/aksharantar_sampled/{language}/{language}_{split}.csv') as f:
        data_pairs = f.readlines()
    data_given = [pair.split(',')[0].strip().lower() for pair in data_pairs]
    data_target = [pair.split(',')[1].strip('\n').strip() for pair in data_pairs]
    return data_given, data_target

def make_alphabet(data: List[str]) -> Alphabet:
    alphabet = Alphabet()
    for word in data:
        for letter in word:
            alphabet.addLetter(letter)
    return alphabet

def word_to_tensor(alphabet: Alphabet, word: str) -> torch.Tensor:
    chars = list(alphabet.getLetterIndex(letter) for letter in word) + [1]
    return torch.tensor(chars, dtype=torch.long).reshape(-1, 1)
