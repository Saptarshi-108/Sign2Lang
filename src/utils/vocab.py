import json
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.idx = 4

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = self.idx
                    self.itos[self.idx] = word
                    self.idx += 1

    def numericalize(self, text):
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in text.lower().split()]

    def save_vocab(self, path):
        with open(path, "w") as f:
            json.dump({"itos": self.itos, "stoi": self.stoi}, f)

    def load_vocab(self, path):
        with open(path, "r") as f:
            vocab = json.load(f)
            self.itos = {int(k): v for k, v in vocab["itos"].items()}
            self.stoi = vocab["stoi"]
