import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as tt


class WikiArtDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.artist_map = {name: idx for idx, name in enumerate(df['artist'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.artist_map[self.df.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label




def get_data_loaders(csv_path, root_dir, batch_size=128, image_size=64):
    df = pd.read_csv(csv_path)
    train_data = df[df['subset'] == 'train']
    val_data = df[df['subset'] == 'val']
    test_data = df[df['subset'] == 'test']

    transform = tt.Compose([
        tt.Resize(image_size),
        tt.CenterCrop(image_size),
        tt.ToTensor(),
        tt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = WikiArtDataset(train_data, root_dir, transform=transform)
    val_dataset = WikiArtDataset(val_data, root_dir, transform=transform)
    test_dataset = WikiArtDataset(test_data, root_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    return train_loader, val_loader, test_loader
