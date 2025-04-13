import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import yaml
from model.encoder import EncoderLSTM
from model.decoder import DecoderLSTM
from model.seq2seq import Seq2Seq
from utils.vocab import Vocabulary

class SignTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("600x300")
        self.root.configure(bg="#f0f0f0")

        self.label = tk.Label(root, text="Upload a .npz keypoint file", font=("Arial", 14), bg="#f0f0f0")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="Browse File", command=self.load_file, font=("Arial", 12))
        self.upload_button.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=550, bg="#f0f0f0")
        self.result_label.pack(pady=20)

        self.model, self.vocab, self.config = self.load_model_and_vocab()

    def load_model_and_vocab(self):
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        vocab = Vocabulary()
        vocab.load_vocab(config['data']['vocab_path'])

        device = torch.device("cpu")
        encoder = EncoderLSTM(config['model']['input_dim'], config['model']['hidden_dim'])
        decoder = DecoderLSTM(config['model']['output_dim'], config['model']['embed_dim'], config['model']['hidden_dim'])
        model = Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(torch.load("saved_model.pth", map_location=device))
        model.eval()
        return model, vocab, config

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if not file_path:
            return
        try:
            sentence = self.predict(file_path)
            self.result_label.config(text=f"Predicted Sentence:\n{sentence}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict(self, npz_path):
        data = np.load(npz_path)['keypoints']  # (T, 299)
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, T, 299)
        sos_idx = self.vocab.stoi["<SOS>"]
        eos_idx = self.vocab.stoi["<EOS>"]
        outputs = self.model.translate(input_tensor, max_len=self.config['model']['max_len'], sos_idx=sos_idx, eos_idx=eos_idx)
        sentence = self.vocab.decode(outputs)
        return sentence

if __name__ == "__main__":
    root = tk.Tk()
    app = SignTranslatorApp(root)
    root.mainloop()
