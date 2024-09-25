import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

# function to load the model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # changing the last layer for our output
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # setting model to evaluation mode
    return model.to(device)

# prepare the image for the model
def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resizing the image
        transforms.ToTensor(),  # converting to tensor
    ])
    image = image.convert("RGB")  # making sure image is RGB
    original_size = image.size  # save original size for later
    image = transform(image).unsqueeze(0) 
    return image, original_size

# function to make a prediction
def predict(model, image):
    with torch.no_grad(): 
        output = model(image)  # get model output
        return output.cpu().numpy()  # convert output to numpy array

# function to understand the output from the model
def process_output(output):
    tumor_score = output[0, 0]  # score for tumor presence
    tumor_prediction = 1 if tumor_score >= 0.5 else 0  # decide if it's a tumor; need to make this binary (cancerous or non-cancerous: 1 or 0)
    x_min, y_min, width, height = output[0, 1:5]  # get bounding box coordinates
    
    # changing to standard float
    x_min, y_min, width, height = map(float, [x_min, y_min, width, height])
    x_max, y_max = x_min + width, y_min + height 
    
    return tumor_prediction, (x_min, y_min, x_max, y_max), tumor_score

# function to draw a box around the tumor
def draw_bounding_box(original_image, bounding_box, score, prediction):
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
        
    draw = ImageDraw.Draw(original_image)  # set up to draw on the image
    x_min, y_min, x_max, y_max = map(int, bounding_box) 
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=5)  # draw red box
    draw.text((x_min, y_min), f"Score: {score:.2f}, Prediction: {'Tumor' if prediction == 1 else 'No Tumor'}", fill="red")  # add text
    return original_image  # return the image with box

# main route for the app
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')  # try to get the uploaded image
        if file:
            model = load_model('resnet_model.pth')  # load our model
            image = Image.open(file)  # open the uploaded image
            prepared_image, original_size = prepare_image(image)
            output = predict(model, prepared_image)  # get prediction
            tumor_prediction, bounding_box, score = process_output(output) 

            # adjust bounding box to original scale
            width_scale = original_size[0] / 224
            height_scale = original_size[1] / 224
            scaled_bounding_box = (
                bounding_box[0] * width_scale,
                bounding_box[1] * height_scale,
                bounding_box[2] * width_scale,
                bounding_box[3] * height_scale,
            )

            original_image_with_box = draw_bounding_box(image.copy(), scaled_bounding_box, score, tumor_prediction)

            # save the image with the box to a BytesIO object
            img_byte_arr = io.BytesIO()
            original_image_with_box.save(img_byte_arr, format='PNG')  # save image as PNG
            img_byte_arr.seek(0) 

            # return results to the web page
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')  # convert image for HTML
            return render_template('index.html', prediction=tumor_prediction, score=score, image=img_base64)
    
    return render_template('index.html', prediction=None)  # if no image, just render the page

if __name__ == "__main__":
    app.run(debug=True)  # start the app in debug mode