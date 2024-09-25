import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# load the trained model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # load the resnet model
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # adjust the last layer for our output
    model.load_state_dict(torch.load(model_path, map_location=device))  # load the model weights
    model.eval()  # switch the model to evaluation mode
    return model.to(device)  # send the model to the right device

# prepare image for inference
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  # convert image to tensor
    ])
    image = Image.open(image_path).convert("RGB") 
    original_size = image.size  # save the original size for later
    image = transform(image).unsqueeze(0)  # add a batch dimension
    return image, original_size  

# run inference on a single image
def predict(model, image):
    with torch.no_grad(): 
        output = model(image)  # get the model's output
        print(f"Model Output (Raw): {output.cpu().numpy()}")  # print raw output for debugging
        return output.cpu().numpy()  

# process the model output to interpret predictions
def process_output(output):
    tumor_score = output[0, 0]  # get the tumor score
    tumor_prediction = 1 if tumor_score >= 0.5 else 0  # decide if we have a tumor or not
    
    # print the tumor score and prediction for debugging
    print(f"Tumor Score: {tumor_score}, Prediction: {'Tumor' if tumor_prediction == 1 else 'No Tumor'}")
    
    # get the bounding box coordinates
    x_min, y_min, width, height = output[0, 1:5]
    
    # convert to standard floats
    x_min = float(x_min)
    y_min = float(y_min)
    width = float(width)
    height = float(height)
    
    x_max = x_min + width  # calculate the max x
    y_max = y_min + height  # calculate the max y
    
    return tumor_prediction, (x_min, y_min, x_max, y_max), tumor_score  # return prediction and box

# draw bounding box on image
def draw_bounding_box(original_image, bounding_box, score, prediction):
    draw = ImageDraw.Draw(original_image)  # create a drawing object
    x_min, y_min, x_max, y_max = map(int, bounding_box) 
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=5)  # draw
    draw.text((x_min, y_min), f"Score: {score:.2f}, Prediction: {'Tumor' if prediction == 1 else 'No Tumor'}", fill="red")  # add text
    return original_image  

# run inference on images from a directory
def run_inference(image_dir, model_path):
    model = load_model(model_path)  # load the model
    
    for filename in os.listdir(image_dir):  # loop through all files in the directory
        if filename.endswith('.jpg') or filename.endswith('.png'):  # check for image files
            image_path = os.path.join(image_dir, filename) 
            print(f"Processing {image_path}...")  # print what's being processed
            image, original_size = prepare_image(image_path) 
            output = predict(model, image)  # get the model's predictions
            tumor_prediction, bounding_box, score = process_output(output) 

            print(f"Predicted output for {filename}: Tumor Prediction: {'Tumor' if tumor_prediction == 1 else 'No Tumor'}, Bounding Box: {bounding_box}, Confidence Score: {score:.2f}")
            
            # load the original image to draw the bounding box
            original_image = Image.open(image_path).convert("RGB")

            # scale the bounding box to the original image size
            width_scale = original_size[0] / 224
            height_scale = original_size[1] / 224
            scaled_bounding_box = (
                bounding_box[0] * width_scale,
                bounding_box[1] * height_scale,
                bounding_box[2] * width_scale,
                bounding_box[3] * height_scale,
            )

            print(f"Original Bounding Box: {bounding_box}")
            print(f"Scaled Bounding Box: {scaled_bounding_box}")

            # draw the bounding box on the original image
            original_image_with_box = draw_bounding_box(original_image, scaled_bounding_box, score, tumor_prediction)
            
            # display the image
            plt.imshow(original_image_with_box) 
            plt.axis('off')  # hide axes
            plt.title(f"{filename} - Tumor: {'Tumor' if tumor_prediction == 1 else 'No Tumor'}, Score: {score:.2f}") 
            plt.show()  # show the plot

if __name__ == "__main__":
    image_directory = '/Users/adityaranjan/Documents/cura3d/test image'  # update this to your image directory
    model_file_path = 'resnet_model.pth'  # path to saved model in main directory
    run_inference(image_directory, model_file_path)  