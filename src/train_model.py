import os
import pickle
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        # getting all the image files from the directory
        self.images = [
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg') or f.endswith('.png') # only accepting jpg or png atm
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        
        try:
            image = Image.open(img_name).convert("RGB")  # load the image
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None  # if there's an error, just skip this one; must be an error from preprocessing

        # make the label file name from the image file name
        label_name = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        try:
            label = np.loadtxt(label_name)  # load the label
            if len(label) != 5:
                print(f"Invalid label shape for {label_name}: expected 5 values, got {len(label)}.")
                return None, None
        except Exception as e:
            print(f"Error loading label {label_name}: {e}")
            return None, None  # skip this one too if there's a problem

        # apply any transformations we want to the image
        if self.transform:
            image = self.transform(image)

        return image, label

def load_data_loaders(pickle_file):
    # check if we have a saved loaders file
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data_loaders = pickle.load(f)  # load the data loaders from the file
            return data_loaders
    else:
        raise FileNotFoundError(f"No saved DataLoaders found at {pickle_file}.")

def train_model(data_loaders, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check if we can use a GPU, but use CPU on my device

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # get the resnet model
    model.fc = nn.Linear(model.fc.in_features, 5)  # adjust the output to match our needs
    model.to(device)

    # define a loss function that combines two losses
    def combined_loss(outputs, targets):
        classification_loss = nn.BCEWithLogitsLoss()(outputs[:, 0], targets[:, 0])  # loss for the class
        box_loss = nn.MSELoss()(outputs[:, 1:], targets[:, 1:])  # loss for the bounding boxes
        return classification_loss + box_loss

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # using Adam optimizer

    for epoch in range(num_epochs):
        model.train()  # set the model to training mode
        for phase in ['axial_training', 'coronal_training', 'sagittal_training']:
            loader = data_loaders[phase]
            running_loss = 0.0

            # loop through the data in batches
            for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}, {phase}"):
                if images is None or labels is None:
                    continue  # skip if there's an issue with the image or label

                images, labels = images.to(device), labels.float().to(device)  # make sure we have floats for labels

                optimizer.zero_grad()  # reset gradients

                outputs = model(images)  # get predictions from the model

                # make sure outputs and labels are the right shape
                if outputs.dim() != 2 or outputs.size(1) != 5:
                    print(f"Shape mismatch: outputs {outputs.shape}, expected (batch_size, 5).")
                    continue

                loss = combined_loss(outputs, labels)  # calculate the loss
                loss.backward()  # backpropagation
                optimizer.step()  # update the model weights

                running_loss += loss.item() 

            avg_loss = running_loss / len(loader)  # average loss for this phase
            print(f"Avg loss for {phase}: {avg_loss:.4f}")

    model_save_path = 'resnet_model.pth'  # where to save the model
    torch.save(model.state_dict(), model_save_path) 
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    pickle_file = 'data_loaders.pkl' 
    data_loaders = load_data_loaders(pickle_file) 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to fit ResNet input
        transforms.ToTensor(),
    ])

    #transformations to datasets if needed
    for phase in data_loaders.keys():
        dataset = data_loaders[phase].dataset
        dataset.transform = transform

    train_model(data_loaders)  # start training the model