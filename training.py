import torch
import os
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image, make_grid
from models import Generator, Discriminator
from dataset import get_data_loaders


# def show_images(images, nmax=64):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xticks([]);
#     ax.set_yticks([])
#     ax.imshow(make_grid((images.detach()[:nmax]) * 0.5 + 0.5, nrow=8).permute(1, 2, 0))


def train_discriminator(real_images, generator, discriminator, opt_d, batch_size, latent_size, device):
    opt_d.zero_grad()

    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    fake_preds = discriminator(fake_images.detach())
    fake_targets = torch.zeros(batch_size, 1, device=device)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)

    loss_d = real_loss + fake_loss
    loss_d.backward()
    opt_d.step()

    return loss_d.item()


def train_generator(generator, discriminator, opt_g, batch_size, latent_size, device):
    opt_g.zero_grad()

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss_g = F.binary_cross_entropy(preds, targets)

    loss_g.backward()
    opt_g.step()

    return loss_g.item()


def train_model(epochs, batch_size, latent_size, device, lr=0.0002):
    data_dir = "images"
    csv_path = "classes.csv"
    train_loader, _, _ = get_data_loaders(csv_path, data_dir, batch_size=batch_size)

    generator = Generator(latent_size=latent_size).to(device)
    discriminator = Discriminator().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in train_loader:
            real_images = real_images.to(device)

            loss_d = train_discriminator(real_images, generator, discriminator, opt_d, batch_size, latent_size, device)

            loss_g = train_generator(generator, discriminator, opt_g, batch_size, latent_size, device)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss Discriminator: {loss_d:.4f}, Loss Generator: {loss_g:.4f}")
        save_sample(generator, epoch, latent_size, device)


    model_dir ='model_saved'
    torch.save(generator.state_dict(), os.path.join(model_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator.pth"))

    print("Training Done")


def save_sample(generator, epoch, latent_size, device, sample_dir="output"):
    latent = torch.randn(1, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    save_image((fake_images) * 0.5 + 0.5, f"{sample_dir}/fake_{epoch}.png", nrow=10)


if __name__ == "__main__":
    epochs = 400
    batch_size = 128
    latent_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(epochs, batch_size, latent_size, device)
