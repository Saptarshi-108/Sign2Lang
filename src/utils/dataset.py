import torch
from torch.utils.data import Dataset
import numpy as np

class KeypointTextDataset(Dataset):
    def __init__(self, npz_dir, annotations, vocab, max_len):
        self.npz_dir = npz_dir
        self.annotations = annotations  # dict: video_id → sentence
        self.vocab = vocab
        self.max_len = max_len
        self.keys = list(annotations.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        vid = self.keys[idx]
        path = f"{self.npz_dir}/{vid}.npz"
        keypoints = np.load(path)["keypoints"]
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        tgt = self.vocab.encode(self.annotations[vid])
        tgt = tgt[:self.max_len] + [0] * (self.max_len - len(tgt))
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)

        return keypoints_tensor, tgt_tensor
