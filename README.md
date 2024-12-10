
# GAN Training on WikiArt Dataset

This project implements a Generative Adversarial Network (GAN) trained on the **WikiArt dataset**, which includes images of artwork across various genres. The GAN architecture consists of a **Generator** for creating synthetic images and a **Discriminator** for differentiating between real and fake images. The model is implemented in PyTorch.



## Project Structure
  ```
├── dataset.py                  # Script for handling data loading and preprocessing
├── __init__.py                 # Initialize the model
├── models.py                   # Contains Generator and Discriminator models
├── classes.csv                 # CSV file mapping images to their metadata (e.g., artist, genre)
├── training.py                 # Script for training the GAN
├── output/                     # Directory to save generated images during training
├── images/                     # Directory containing artwork images (subdirectories for each genre)
└── README.md                   # This README file

 ```
---

## Setup Instructions

### 1. Prerequisites
Ensure you have the following installed:
- Python 3.8 or above
- PyTorch (GPU support recommended for faster training)
- Required Python packages (install via `pip`):
  ```bash
  pip install torch torchvision matplotlib tqdm pandas pillow
  ```

### 2. Dataset Preparation
- The `images/` directory must contain subdirectories for each genre, such as:
  ```
  images/
  ├── Abstract_Expressionism/
  ├── Impressionism/
  ├── Realism/
  ...
  ```


## How to Use

### 1. Update File Paths
In `training.py`, ensure the paths for the CSV and image directory are correctly set:
```python
csv_path = "new_classes.csv"  # Path to the CSV file
data_dir = "images"           # Path to the directory containing images
```

### 2. Running the Training Script
Run the training script using:
```bash
python training.py
```
or 
```bash
python __init__.py
```


### 3. Generated Images
The script will save generated images in the `output/` directory after each epoch. Images are named as `fake_{epoch_number}.png`.



## Code Workflow

### 1. Data Loading
- The `ImageFolder` class from `torchvision` is used to load and preprocess images.
- Images are resized, cropped, and normalized to ensure consistent input dimensions.

### 2. GAN Architecture
- **Generator**: A deep convolutional transpose network that generates fake images from random noise.
- **Discriminator**: A deep convolutional network that classifies images as real or fake.

### 3. Training Loop
- Each epoch includes:
  1. **Discriminator Training**: The Discriminator learns to classify real and fake images.
  2. **Generator Training**: The Generator learns to create realistic images to fool the Discriminator.
- Losses for both networks are logged for monitoring.




## Key Details

### WikiArt Dataset
The WikiArt dataset contains thousands of artwork images from various genres and styles. For this project:
- Images are resized to 64x64 pixels.
- Data is split into training, validation, and testing subsets as specified in `new_classes.csv`.

### Loss Functions
- **Binary Cross-Entropy Loss (BCE)** is used for both the Generator and Discriminator to optimize their outputs.

### Normalization
- Input images are normalized using mean and standard deviation:
  ```python
  stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  ```
- The inverse transformation (`denorm`) is applied before saving generated images.





