import torch
import yaml
from torch.utils.data import DataLoader
from models.encoder import Encoder
from models.decoder import Decoder
from utils.dataset import KeypointTextDataset
from utils.vocab import Vocabulary
import torch.nn as nn
import torch.optim as optim
import json
import os

with open("config/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if cfg["training"]["use_cuda"] and torch.cuda.is_available() else "cpu")

# Load data
with open("data/processed/annotations.json") as f:
    annotations = json.load(f)

sentences = list(annotations.values())
vocab = Vocabulary()
vocab.build(sentences)

dataset = KeypointTextDataset("data/processed", annotations, vocab, cfg["training"]["max_seq_len"])
loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=lambda x: x)

# Models
encoder = Encoder(**cfg["model"]).to(device)
decoder = Decoder(len(vocab.token2idx), cfg["model"]["hidden_size"], cfg["model"]["num_layers"], cfg["model"]["dropout"]).to(device)

# Optimizer, loss
enc_opt = optim.Adam(encoder.parameters(), lr=cfg["training"]["learning_rate"])
dec_opt = optim.Adam(decoder.parameters(), lr=cfg["training"]["learning_rate"])
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(cfg["training"]["epochs"]):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in loader:
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        for keypoints, tgt in batch:
            keypoints, tgt = keypoints.to(device), tgt.to(device)
            enc_out, (h, c) = encoder(keypoints.unsqueeze(0))
            loss = 0

            for t in range(1, tgt.size(0)):
                out, h, c = decoder(tgt[t - 1], h, c)
                loss += criterion(out, tgt[t].unsqueeze(0))

            loss.backward()
            enc_opt.step()
            dec_opt.step()
            total_loss += loss.item()

    print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {total_loss:.2f}")
