import tensorflow as tf
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import streamlit as st
import os
import io
import torch

# Function to generate adversarial attack using FGSM
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
    # Processed image
    processed_image = image.clone().detach().requires_grad_(True)

    # Calculate the loss for the image.
    loss = model(processed_image).logits[0, label]

    # Calculate the gradients of the loss with respect to the image.
    gradients = torch.autograd.grad(loss, processed_image)[0]

    # Create the adversarial pattern.
    adversarial_pattern = epsilon * gradients.sign()

    return adversarial_pattern

def evaluate_model(data_dir):
    st.title("Model Evaluation")

    # Load the pretrained ViT model for image classification
    feature_extractor = ViTImageProcessor.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")
    model = ViTForImageClassification.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")

    # Get the list of folders in the evaluation directory
    eval_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    # Create a mapping between class names and integer labels
    class_to_label = {folder: idx for idx, folder in enumerate(eval_folders)}

    # Create a dropdown to select the class
    selected_class = st.selectbox("Select a class:", eval_folders)

    # Get the list of image files in the selected folder
    image_files = [file for file in os.listdir(os.path.join(data_dir, selected_class)) if file.endswith(".png")]

    if not image_files:
        st.write(f"No image files found in the selected class: {selected_class}")
        return

    # Create a dropdown to select the image
    selected_image = st.selectbox("Select an image:", image_files)

    # Load the image
    image_path = os.path.join(data_dir, selected_class, selected_image)
    image = Image.open(image_path)

    # Preprocess the image and convert it to PyTorch tensor
    processed_image = feature_extractor(images=image, return_tensors="pt")

    # Get the input label of the image
    label = class_to_label[selected_class]

    # Generate adversarial attack using FGSM
    epsilon = 0.1  # You can adjust the value of epsilon as needed
    perturbations = create_adversarial_pattern(processed_image.input_values, label, model, epsilon)

    # Create the adversarial image
    adversarial_image = processed_image.input_values + epsilon * torch.unsqueeze(perturbations, 0)

    # Convert the adversarial image to a numpy array and display it
    adv_image = adversarial_image.detach().numpy().squeeze()
    st.subheader(f"{selected_class} - Adversarial Image (FGSM)")
    st.image(adv_image, caption=f"{selected_class} - Adversarial (FGSM)", use_column_width=True)

    # Display the original image
    st.image(image, caption=f"{selected_class} - Original Image")
    
# Main App
def main():
    data_dir = r"C:\Users\serih\Desktop\diabetes-data-engineering-project\main\images\evaluation2"
    evaluate_model(data_dir)

if __name__ == "__main__":
    main()
