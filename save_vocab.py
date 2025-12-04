"""
Script to save the vocabulary as a pickle file for deployment.
This eliminates the need to have the dataset available during inference.

Run this script once after training to generate vocab.pkl
"""

import torch
import torchvision.transforms as transforms
from dataset import get_loader
import pickle


def main():
    # Same transform as used in training
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load dataset to build vocabulary
    print("Loading dataset to build vocabulary...")
    _, dataset = get_loader(
        root_folder="data/Images",
        annotation_file="data/captions.txt",
        transform=transform,
        num_workers=0,  # Set to 0 for simplicity
    )

    # Save the vocabulary object
    vocab_path = "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(dataset.vocab, f)

    print(f"Vocabulary saved to {vocab_path}")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Sample mappings:")
    print(f"  '<PAD>': {dataset.vocab.stoi['<PAD>']}")
    print(f"  '<SOS>': {dataset.vocab.stoi['<SOS>']}")
    print(f"  '<EOS>': {dataset.vocab.stoi['<EOS>']}")
    print(f"  '<UNK>': {dataset.vocab.stoi['<UNK>']}")


if __name__ == "__main__":
    main()
