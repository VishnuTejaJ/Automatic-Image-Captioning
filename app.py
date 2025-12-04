import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from dataset import FlickrDataset
import os
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(
    page_title="Image Caption Generator", page_icon="📷", layout="centered"
)

st.title("📷 Image Caption Generator")
st.markdown("Upload an image and the AI will describe it for you!")

# HuggingFace model details
HF_REPO = "VishnuTejaJ/my_checkpoint.pth.tar/tree/main"
HF_FILENAME = "my_checkpoint.pth.tar"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"


def download_checkpoint():
    """Download checkpoint from HuggingFace if not present locally."""
    if not os.path.exists(CHECKPOINT_PATH):
        st.info(f"Downloading model checkpoint from HuggingFace... (~320MB)")
        with st.spinner("Downloading checkpoint... This may take a few minutes."):
            try:
                # Use huggingface_hub for better retry logic and rate limit handling
                downloaded_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=HF_FILENAME,
                    cache_dir=".",
                    local_dir=".",
                    local_dir_use_symlinks=False,
                )
                st.success("Checkpoint downloaded successfully!")
                return downloaded_path
            except Exception as e:
                st.error(f"Failed to download checkpoint: {e}")
                st.info(
                    "💡 Tip: If you see a rate limit error, please wait a few minutes and refresh the page."
                )
                st.stop()
    return CHECKPOINT_PATH


@st.cache_resource
def load_resources(checkpoint_path):
    # Load dataset to get vocab
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Note: This rebuilds vocab from the dataset
    # For Streamlit Cloud, we need the data folder with captions.txt
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


# Download checkpoint from HuggingFace
checkpoint_path = download_checkpoint()

# Load resources
if os.path.exists("data/captions.txt"):
    model, dataset, device = load_resources(checkpoint_path)
else:
    st.error(
        "Dataset not found! Please ensure data/captions.txt exists in the repository."
    )
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
