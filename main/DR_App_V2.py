import tensorflow as tf
from PIL import Image
from transformers import ViTFeatureExtractor, ViTImageProcessor
import streamlit as st 
import os
import io
import torch
import numpy as np

def create_adversarial_pattern(image, label, model, epsilon):
    """Generates an adversarial pattern for the input image.

    Args:
        image: The input image tensor.
        label: The label of the image.
        model: The pretrained ViT model.
        epsilon: The epsilon value to use for the adversarial attack.

    Returns:
        The adversarial pattern tensor.
    """

    # Create a copy of the image tensor.
    processed_image = image.clone().detach()

    # Calculate the loss for the image.
    loss = model(processed_image).logits[label]

    # Calculate the gradients of the loss with respect to the image.
    gradients = torch.autograd.grad(loss, processed_image, create_graph=True)[0]

    # Create the adversarial pattern.
    adversarial_pattern = epsilon * gradients.sign()

    return adversarial_pattern

def evaluate_model(data_dir):
    st.title("Model Evaluation")

    # Load the pretrained ViT model for image classification
    feature_extractor = ViTFeatureExtractor.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")
    model = ViTImageProcessor.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")

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

        if image is not None:
            processed_image = feature_extractor(images=image, return_tensors="pt")

            # Generate adversarial attack using FGSM
            adversarial_image = create_adversarial_pattern(processed_image, selected_class, model, epsilon=0.1)

            # Convert the adversarial attack to an image and display it
            adv_image = Image.fromarray(np.uint8(np.squeeze(adversarial_image) * 255.0))
            st.subheader(f"{selected_class} - Adversarial Image (FGSM)")
            st.image(adv_image, caption=f"{selected_class} - Adversarial (FGSM)", use_column_width=True)

        # Display the original image
        st.image(image, caption=f"{selected_class} - Original Image")

# Main App
def main():
    data_dir = "/Users/kkhan/Desktop/AAI1001/Project Folder/diabetes-data-engineering-project/main/images/evaluation2"
    evaluate_model(data_dir)

if __name__ == "__main__":
    main()