import tensorflow as tf
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import streamlit as st
import os
import io
import torch
import torch.nn as nn
import torch.optim as optim

def fgsm_attack(model, image_path, epsilon=0.03):
    # Load the image and preprocess for the model
    image = Image.open(image_path)
    preprocess = AutoFeatureExtractor.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")
    input_tensor = preprocess(image, return_tensors="pt")["pixel_values"]

    # Set the model to evaluation mode
    model.eval()

    # Clone the image tensor to avoid modifying the original image
    perturbed_image = input_tensor.clone().requires_grad_()

    # Forward pass through the model to get the predictions
    output = model(perturbed_image)

    # Get the true label (class with the highest probability)
    _, target_class = output.max(1)

    # Calculate the loss (cross-entropy) between the predicted label and the true label
    loss = nn.CrossEntropyLoss()
    model.zero_grad()
    loss_value = loss(output, target_class)

    # Calculate gradients of the loss with respect to the input image
    loss_value.backward()

    # Get the data gradient
    data_grad = perturbed_image.grad.data

    # Create a perturbed image by adding the sign of the data gradient
    perturbed_image = perturbed_image + epsilon * data_grad.sign()

    # Clip the perturbed image to ensure it stays within valid pixel range (0 to 1 for normalized images)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Set the model back to training mode
    model.train()

    return perturbed_image

# Model Evaluation Function with FGSM Attack
def evaluate_model(data_dir, epsilon=0.03):
    st.title("Model Evaluation")
    
    # Load the model and feature extractor
    extractor = AutoFeatureExtractor.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")
    model = AutoModelForImageClassification.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")

    # Get the list of folders in the evaluation directory
    eval_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    # Create a dropdown to select the class
    selected_class = st.selectbox("Select a class:", eval_folders)

    # Get the list of image files in the selected folder
    image_files = [file for file in os.listdir(os.path.join(data_dir, selected_class)) if file.endswith(".png")]

    # Create a dropdown to select the image
    selected_image = st.selectbox("Select an image:", image_files, index=0)  # Set index to 0 to display the first image

    # If no image is selected, set the image path to None
    image_path = os.path.join(data_dir, selected_class, selected_image) if selected_image else None

    # If an image is selected, display it
    if image_path and os.path.exists(image_path):
        # Convert the string to a file object
        image_file = io.BytesIO(open(image_path, "rb").read())

        # Load the image
        image = Image.open(image_file)

        # Display the original image
        st.image(image, caption=f"{selected_class} - Original Image")

        # Generate the adversarial image using FGSM
        perturbed_image = fgsm_attack(model, image_path, epsilon=epsilon)

        # Convert the adversarial image tensor to a PIL Image
        adversarial_image_pil = transforms.ToPILImage()(perturbed_image[0].cpu())

        # Display the adversarial image
        st.image(adversarial_image_pil, caption=f"{selected_class} - Adversarial Image")

# Main App
def main():
    data_dir = "main/images/evaluation2"
    evaluate_model(data_dir)

if __name__ == "__main__":
    main()
