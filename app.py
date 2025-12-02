import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from dataset import FlickrDataset
import os

# Page config
st.set_page_config(
    page_title="Image Caption Generator", page_icon="📷", layout="centered"
)

st.title("📷 Image Caption Generator")
st.markdown("Upload an image and the AI will describe it for you!")

# Sidebar for settings
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Checkpoint Path", "my_checkpoint.pth.tar")


@st.cache_resource
def load_resources(checkpoint_path):
    # Load dataset to get vocab
    # We assume the data is in the same relative path as training
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Note: This might be slow if it rebuilds vocab every time.
    # In a real app, we'd pickle the vocab.
    # For now, it's fine as Flickr8k is small.
    dataset = FlickrDataset(
        root_dir="data/Images",
        captions_file="data/captions.txt",
        transform=transform,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model params (must match training)
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    attention_dim = 256

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, attention_dim).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        st.sidebar.success("Model loaded successfully!")
    else:
        st.sidebar.warning(
            f"Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    return model, dataset, device


# Load resources
if os.path.exists("data/captions.txt"):
    model, dataset, device = load_resources(model_path)
else:
    st.error("Dataset not found! Please check the path.")
    st.stop()

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
            caption = model.caption_image(img_tensor.squeeze(0), dataset.vocab)

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
