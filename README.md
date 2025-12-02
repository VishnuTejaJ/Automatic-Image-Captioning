# Image Caption Generator

A deep learning model that generates descriptive captions for images using a ResNet50 encoder and an LSTM decoder with Attention mechanism.

## Features

- **Encoder-Decoder Architecture**: Uses ResNet50 for image feature extraction and LSTM for caption generation.
- **Attention Mechanism**: Implements Bahdanau Attention to focus on relevant image parts during caption generation.
- **Web Application**: Includes a Streamlit-based web interface for easy interaction.
- **Custom Dataset Support**: Designed to work with the Flickr8k dataset but adaptable to others.

## Prerequisites

- Python 3.8, 3.9, or 3.10 (Recommended)
- CUDA-capable GPU (Recommended for training)

## Installation

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (optional but recommended)**:

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Spacy English Model**:
    This project uses Spacy for tokenization. You need to download the English language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Dataset Setup

1.  Download the **Flickr8k** dataset (Images and Captions).
2.  Create a `data` folder in the project root.
3.  Inside `data`, create an `Images` folder and place all image files there.
4.  Place the `captions.txt` file directly inside the `data` folder.

Structure:

```
project_root/
  ├── data/
  │   ├── Images/
  │   │   ├── 1000268201_693b08cb0e.jpg
  │   │   ├── ...
  │   └── captions.txt
  ├── train.py
  ├── app.py
  ├── ...
```

## Usage

### Training

To train the model from scratch:

```bash
python train.py
```

- **Configuration**: You can adjust hyperparameters (epochs, batch size, learning rate) directly in `train.py`.
- **Checkpoints**: The model saves checkpoints to `my_checkpoint.pth.tar`.
- **Logs**: Training progress is logged to `runs/` and can be viewed with TensorBoard:
  ```bash
  tensorboard --logdir runs
  ```

### Web Application

To run the interactive web app:

```bash
streamlit run app.py
```

1.  Upload an image.
2.  The app will load the trained model (`my_checkpoint.pth.tar`) and generate a caption.

### Inference Script

You can also generate captions via the command line:

```bash
python inference.py
```

(Note: You may need to modify `inference.py` to point to your specific test image).

## Project Structure

- `model.py`: Defines the EncoderCNN, DecoderRNN, and Attention architecture.
- `dataset.py`: Handles data loading, vocabulary building, and preprocessing.
- `train.py`: Main training loop.
- `app.py`: Streamlit web application.
- `inference.py`: Script for generating captions on single images.
- `utils.py`: Helper functions for saving/loading checkpoints.

## Acknowledgements

- Based on the "Show, Attend and Tell" paper.
- Uses pre-trained ResNet50 from Torchvision.
