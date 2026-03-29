"""
RNN Explanation Generator Training Script

Trains a multi-layer GRU model that generates natural language
repair explanations given an issue type.

Training data: pairs of (issue_type, explanation_text)
"""

import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Training data: issue-type -> list of explanation texts
TRAINING_DATA = {
    "cracked_screen": [
        "The screen shows visible cracks. Screen replacement is recommended. Estimated time: 30 minutes.",
        "Display damage detected with crack patterns. A new screen assembly should be installed.",
        "Screen integrity compromised. Touch sensitivity may be affected. Replace the display module.",
    ],
    "battery_swelling": [
        "Battery swelling detected. Immediate replacement required for safety. Estimated time: 45 minutes.",
        "The battery has expanded beyond safe limits. Replace with certified battery immediately.",
        "Swollen battery poses fire risk. Remove and replace the battery module urgently.",
    ],
    "charging_port_damage": [
        "Charging port shows damage or debris. Clean or replace the port. Estimated time: 20 minutes.",
        "Connection issues detected at charging port. Inspect for debris and physical damage.",
        "Port module may need replacement. Clean contacts and test before full replacement.",
    ],
}


class SimpleVocab:
    """Character-level vocabulary for the prototype."""

    def __init__(self):
        self.char_to_idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.idx_to_char = {0: "<pad>", 1: "<sos>", 2: "<eos>"}
        self.size = 3

    def build(self, texts: list[str]):
        for text in texts:
            for ch in text:
                if ch not in self.char_to_idx:
                    self.char_to_idx[ch] = self.size
                    self.idx_to_char[self.size] = ch
                    self.size += 1

    def encode(self, text: str, max_len: int = 200) -> list[int]:
        tokens = [self.char_to_idx["<sos>"]]
        for ch in text[:max_len - 2]:
            tokens.append(self.char_to_idx.get(ch, 0))
        tokens.append(self.char_to_idx["<eos>"])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        chars = []
        for t in tokens:
            ch = self.idx_to_char.get(t, "")
            if ch == "<eos>":
                break
            if ch not in ("<pad>", "<sos>"):
                chars.append(ch)
        return "".join(chars)


class ExplanationDataset(Dataset):
    def __init__(self, data: dict, vocab: SimpleVocab, max_len: int = 200):
        self.samples = []
        self.issue_to_idx = {name: i for i, name in enumerate(data.keys())}
        self.max_len = max_len
        self.vocab = vocab

        for issue_type, texts in data.items():
            issue_idx = self.issue_to_idx[issue_type]
            for text in texts:
                encoded = vocab.encode(text, max_len)
                self.samples.append((issue_idx, encoded))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        issue_idx, tokens = self.samples[idx]
        # Pad to max_len
        padded = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(issue_idx), torch.tensor(padded[: self.max_len])


class ExplanationRNN(nn.Module):
    def __init__(self, num_issues, vocab_size, embed_dim=64, hidden_size=256, num_layers=3):
        super().__init__()
        self.issue_embedding = nn.Embedding(num_issues, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, issue_ids, target_tokens):
        batch_size, seq_len = target_tokens.shape
        issue_emb = self.issue_embedding(issue_ids).unsqueeze(1).expand(-1, seq_len, -1)
        token_emb = self.token_embedding(target_tokens)
        combined = torch.cat([issue_emb, token_emb], dim=-1)
        rnn_out, _ = self.rnn(combined)
        return self.output(rnn_out)


def train(
    epochs: int = 100,
    lr: float = 0.001,
    output_path: str = "../backend/weights/rnn_explainer.pt",
):
    # Build vocabulary
    all_texts = []
    for texts in TRAINING_DATA.values():
        all_texts.extend(texts)
    vocab = SimpleVocab()
    vocab.build(all_texts)

    dataset = ExplanationDataset(TRAINING_DATA, vocab)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    model = ExplanationRNN(
        num_issues=len(TRAINING_DATA),
        vocab_size=vocab.size,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for issue_ids, tokens in loader:
            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            optimizer.zero_grad()
            logits = model(issue_ids, input_tokens)
            loss = criterion(logits.reshape(-1, vocab.size), target_tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)

    # Save vocabulary
    vocab_path = output_path.replace(".pt", "_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"char_to_idx": vocab.char_to_idx, "size": vocab.size}, f)

    print(f"Model saved to {output_path}")
    print(f"Vocab saved to {vocab_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RNN explanation generator")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="../backend/weights/rnn_explainer.pt")
    args = parser.parse_args()

    train(args.epochs, args.lr, args.output)
