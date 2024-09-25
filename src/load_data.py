import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import yaml

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        # grabbing all the jpg files from the image directory
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') and f != '.DS_Store'])
        # same thing for the label files, but they are txt files
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt') and f != '.DS_Store'])
        self.transform = transform

        # print how many images and labels we found
        num_images = len(self.image_filenames)
        num_labels = len(self.label_filenames)
        print(f"Loading images from {image_dir}...")
        print(f"Found {num_images} images and {num_labels} labels.")

        # checking if any images are missing their labels
        missing_labels = [img for img in self.image_filenames if img.replace('.jpg', '.txt') not in self.label_filenames]
        missing_images = [lbl for lbl in self.label_filenames if lbl.replace('.txt', '.jpg') not in self.image_filenames]

        if missing_labels:
            print(f"These images are missing labels: {missing_labels}")
        if missing_images:
            print(f"These labels are missing images: {missing_images}")

        # if the number of images and labels don't match
        if num_images != num_labels:
            print("Mismatch detected!")
            print(f"Image filenames: {self.image_filenames}")
            print(f"Label filenames: {self.label_filenames}")
            raise ValueError("Number of images and labels do not match.")

    def __len__(self):
        # returning how many images we have
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        # loading the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        # loading the label
        try:
            with open(label_path, 'r') as f:
                label_data = np.loadtxt(f)  # getting all the label data
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            raise

        # making sure the label is what we expect
        if label_data.ndim != 1 or len(label_data) != 5:
            print(f"Label shape error for {label_path}: {label_data.shape}")
            raise ValueError(f"Label for {label_path} must have exactly 5 values.")

        # basic image preprocessing
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # resize to 224x224
                transforms.ToTensor(),  # convert to tensor
            ])
            image = transform(image)

        return image, label_data

def load_all_data(base_dir, batch_size=32):
    all_loaders = {}
    planes = ['axial', 'coronal', 'sagittal']
    
    for plane in planes:
        yaml_file_path = os.path.join(base_dir, plane, f"{plane}.yaml")
        if os.path.exists(yaml_file_path):
            with open(yaml_file_path, 'r') as yaml_file:
                config = yaml.safe_load(yaml_file)  # loading the config file
        else:
            print(f"YAML file {yaml_file_path} does not exist. Skipping plane {plane}.")
            continue

        for data_type in ['training', 'validation', 'test']:
            image_dir = os.path.join(base_dir, plane, 'images', data_type)
            label_dir = os.path.join(base_dir, plane, 'labels', data_type)

            # checking if image and label directories exist
            if os.path.exists(image_dir) and os.path.exists(label_dir):
                dataset = MedicalImageDataset(image_dir, label_dir)
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # making batches
                all_loaders[f"{plane}_{data_type}"] = data_loader
            else:
                print(f"Skipping {plane} {data_type}: directory does not exist.")

    return all_loaders

if __name__ == "__main__":
    base_directory = 'data'  # replace w/ your own directory
    loaders = load_all_data(base_directory)  # loading all the data
    print("Data loading complete.")