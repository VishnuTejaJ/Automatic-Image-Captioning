import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
import pickle
import os
import urllib.request
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Image Caption Generator", page_icon="📷", layout="centered"
)

st.title("📷 Image Caption Generator")
st.markdown("Upload an image and the AI will describe it for you!")

# Hugging Face model URL
CHECKPOINT_URL = "https://huggingface.co/VishnuTejaJ/my_checkpoint.pth.tar/resolve/main/my_checkpoint.pth.tar"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
VOCAB_PATH = "vocab.pkl"


def download_checkpoint(url, destination):
    """Download checkpoint from Hugging Face if not present locally."""
    if not os.path.exists(destination):
        st.info(f"Downloading model checkpoint from Hugging Face... (~320MB)")
        with st.spinner("Downloading checkpoint... This may take a few minutes."):
            try:
                urllib.request.urlretrieve(url, destination)
                st.success("Checkpoint downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download checkpoint: {e}")
                st.stop()
    return destination


@st.cache_data
def load_vocabulary(vocab_path):
    """Load the pickled vocabulary."""
    if not os.path.exists(vocab_path):
        st.error(
            f"Vocabulary file not found at {vocab_path}. Please run save_vocab.py first."
        )
        st.stop()

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


@st.cache_resource
def load_model_and_vocab():
    """Load vocabulary and model."""
    # Load vocabulary
    vocab = load_vocabulary(VOCAB_PATH)

    # Download checkpoint if needed
    checkpoint_path = download_checkpoint(CHECKPOINT_URL, CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model params (must match training)
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    attention_dim = 256

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, attention_dim).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()

    return model, vocab, device


# Load resources
model, vocab, device = load_model_and_vocab()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Transform image
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            img_tensor = transform(image).unsqueeze(0).to(device)

            # Generate
            caption = model.caption_image(img_tensor.squeeze(0), vocab)

            # Clean up caption (remove <SOS>, <EOS>)
            cleaned_caption = []
            for word in caption:
                if word == "<SOS>":
                    continue
                if word == "<EOS>":
                    break
                cleaned_caption.append(word)

            final_caption = " ".join(cleaned_caption)

            st.success(f"**Caption:** {final_caption}")
