import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import os

# Load the pretrained model
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

# Function to preprocess the image
@st.cache_data()
def preprocess_image(_image):
    image = _image.resize((224, 224))  # Resize the image to the model's input size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to perform FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    perturbation = epsilon * data_grad.sign()
    adv_image = image + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image

# Function to generate and save adversarial images for all class folders
def generate_adversarial_images(base_dir, class_folders, device):
    for data_dir in class_folders:
        full_data_dir = os.path.join(base_dir, data_dir)

        # Create "Adv" folder for saving adversarial images
        adv_folder_name = data_dir + "_Adv"
        adv_folder_path = os.path.join(base_dir, adv_folder_name)
        if not os.path.exists(adv_folder_path):
            os.makedirs(adv_folder_path)

        # Get the list of image files in the selected folder
        image_files = [file for file in os.listdir(full_data_dir) if file.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Loop through all images in the folder
        for selected_image in image_files:
            # Load and preprocess the image
            image_path = os.path.join(full_data_dir, selected_image)
            image = Image.open(image_path)
            input_image = preprocess_image(image)

            # Move the input image to the appropriate device
            input_image = input_image.to(device)

            # Get the model's prediction for the input image
            with torch.no_grad():
                output = model(input_image)
            _, predicted_class = torch.max(output.logits, 1)

            # Perform the FGSM attack
            epsilon = 0.01  # You can change the epsilon value here
            input_image.requires_grad = True

            # Get the model's prediction for the adversarial image
            with torch.enable_grad():
                output = model(input_image)
                loss = nn.CrossEntropyLoss()(output.logits, predicted_class)
            model.zero_grad()
            loss.backward()

            # Generate the adversarial image using FGSM
            adv_image = fgsm_attack(input_image, epsilon, input_image.grad.data)

            # Save the adversarial image in the "Adv" folder
            adv_image_path = os.path.join(adv_folder_path, selected_image)
            adv_image = adv_image.cpu().squeeze(0)
            transform = transforms.Compose([
                transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                transforms.ToPILImage()
            ])
            adv_image = transform(adv_image)
            adv_image.save(adv_image_path)

# Streamlit app
def main():
    st.title("Adversarial Attack with FGSM")

    # Define the base directory where the image folders are located
    base_dir = "/Users/kkhan/Desktop/AAI1001/Project Folder/diabetes-data-engineering-project/main/images/evaluation2"  # Update this with the path to the "evaluation/" folder

    # List of all class folders
    class_folders = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

    # Convert the model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the model and generate/save adversarial images only once at the beginning
    generate_adversarial_images(base_dir, class_folders, device)

    # Show dropdown images
    st.header("Select an image to view:")
    selected_data_dir = st.selectbox("Select a folder:", class_folders)
    full_data_dir = os.path.join(base_dir, selected_data_dir)
    image_files = [file for file in os.listdir(full_data_dir) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    selected_image = st.selectbox("Select an image:", image_files)
    if selected_image:
        image_path = os.path.join(full_data_dir, selected_image)
        image = Image.open(image_path)
        st.image(image, caption="Selected Image", width=224)

        # Load and preprocess the selected image using caching
        input_image = preprocess_image(image)

        # Convert the input image to the appropriate device
        input_image = input_image.to(device)
        
        # Get the original image label from the folder name
        original_label = selected_data_dir

        # Get the model's prediction for the input image
        with torch.no_grad():
            output = model(input_image)
        _, predicted_class = torch.max(output.logits, 1)
        st.write(f"Original Image Prediction: {original_label}")

        # Perform the FGSM attack
        epsilon = 0.01  # You can change the epsilon value here
        input_image.requires_grad = True

        # Get the model's prediction for the adversarial image
        with torch.enable_grad():
            output = model(input_image)
            loss = nn.CrossEntropyLoss()(output.logits, predicted_class)
        model.zero_grad()
        loss.backward()

        # Generate the adversarial image using FGSM
        adv_image = fgsm_attack(input_image, epsilon, input_image.grad.data)

        # Convert the adversarial image back to CPU for visualization
        adv_image = adv_image.cpu().squeeze(0)
        transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])
        adv_image = transform(adv_image)

        # Display original and adversarial images
        st.image([image, adv_image], caption=["Original Image", "Adversarial Image"], width=224)

    # Calculate and display the accuracy of the model using adversarial images
    st.header("Model Accuracy")

    # Variables to keep track of correct and total predictions
    correct_predictions = 0
    total_predictions = 0

    # Loop over all class folders and images
    for data_dir in class_folders:
        full_data_dir = os.path.join(base_dir, data_dir)
        all_image_files = [file for file in os.listdir(full_data_dir) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
        total_predictions += len(all_image_files)

        for image_file in all_image_files:
            image_path = os.path.join(full_data_dir, image_file)
            image = Image.open(image_path)
            input_image = preprocess_image(image)
            input_image = input_image.to(device)

            # Get the original image label from the folder name
            original_label = data_dir

            # Get the model's prediction for the input image
            with torch.no_grad():
                output = model(input_image)
            _, predicted_class = torch.max(output.logits, 1)

            # Check if the prediction is correct
            if class_folders[predicted_class.item()] == original_label:
                correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    st.write(f"Model Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
