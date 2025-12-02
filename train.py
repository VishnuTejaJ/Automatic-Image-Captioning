import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from dataset import get_loader
from model import CNNtoRNN
from tqdm import tqdm


def train():
    # Hyperparameters
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="data/Images",
        annotation_file="data/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Model hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    attention_dim = 256
    learning_rate = 3e-4
    num_epochs = 25

    # Initialize model, loss, optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, attention_dim, train_CNN).to(
        device
    )
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter("runs/flickr8k")
    step = 0

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # tqdm for a nice progress bar
        loop = tqdm(train_loader, leave=True)

        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs, alphas = model(imgs, captions)

            # Reshape for loss calculation
            # outputs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # targets: (batch_size, seq_len) -> (batch_size * seq_len)

            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = captions[:, 1:].reshape(-1)

            loss = criterion(outputs, targets)
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())


if __name__ == "__main__":
    train()
