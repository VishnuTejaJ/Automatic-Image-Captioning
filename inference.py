import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
import pickle
import os

# We need to load the vocabulary to convert indices back to words
# Since we built the vocab in dataset.py, we should probably save it during training.
# But for now, let's assume we can rebuild it or load it if we saved it.
# To make this robust, let's add a small check in dataset.py or just rebuild it here quickly
# (rebuilding is fast for Flickr8k).
from dataset import FlickrDataset, Vocabulary


def load_model(checkpoint_path, embed_size, hidden_size, vocab_size, attention_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, attention_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def generate_caption(image_path, model, dataset, device, max_length=50):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    feature = model.encoderCNN(img_tensor)
    # We need to reshape feature to (1, 49, 2048) if it's not already handled in forward
    # In our model.py, encoderCNN returns (batch, 49, embed_size) directly?
    # Wait, let's check model.py.
    # EncoderCNN returns (batch, 49, embed_size). Correct.

    # Actually, let's just use the caption_image method we added to CNNtoRNN!
    # It handles everything.

    caption = model.caption_image(img_tensor.squeeze(0), dataset.vocab, max_length)
    return " ".join(caption)


if __name__ == "__main__":
    # Example usage
    # We need to initialize the dataset to get the vocab
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    # Initialize dataset to load vocabulary
    dataset = FlickrDataset(
        root_dir="data/Images",
        captions_file="data/captions.txt",
        transform=transform,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These must match training!
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    attention_dim = 256

    model = load_model(
        "my_checkpoint.pth.tar", embed_size, hidden_size, vocab_size, attention_dim
    )
    print(generate_caption("test.jpg", model, dataset, device))
    # print("Inference script ready. Load model and call generate_caption.")
