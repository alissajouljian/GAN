# __init__.py
from training import train_model
import torch
if __name__ == "__main__":
    epochs = 400
    batch_size = 128
    latent_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(epochs, batch_size, latent_size, device)
