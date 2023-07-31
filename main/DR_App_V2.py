import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import tensorflow as tf
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os


# Define the loss function
loss_object = tf.keras.losses.CategoricalCrossentropy()

# Function to create the adversarial pattern using FGSM
def create_adversarial_pattern(input_image, input_label, model):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

# Function to generate adversarial attack using FGSM
def generate_adversarial_attack(image, dr_class_index, model):
    num_classes = model.config.num_labels

    label = tf.one_hot(dr_class_index, num_classes)
    label = tf.reshape(label, (1, num_classes))

    perturbations = create_adversarial_pattern(image, label, model)

    epsilons = [0, 0.01, 0.1, 0.15]

    adversarial_images = []
    for eps in epsilons:
        adv_x = image + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        adversarial_images.append(adv_x)

    return adversarial_images

# Function to evaluate the model's accuracy and robustness
def evaluate_model(data_dir):
    st.title("Model Evaluation")

    # Load the pretrained ViT model for image classification
    feature_extractor = ViTFeatureExtractor.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")
    model = ViTForImageClassification.from_pretrained("rafalosa/diabetic-retinopathy-224-procnorm-vit")

    # Get the list of folders in the evaluation directory
    eval_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    # Create a dropdown to select the class
    selected_class = st.selectbox("Select a class:", eval_folders)

    # Get the list of image files in the selected folder
    image_files = [file for file in os.listdir(os.path.join(data_dir, selected_class)) if file.endswith(".jpeg")]

    # Create a dropdown to select the image
    selected_image = st.selectbox("Select an image:", image_files)

    # Load the image
    image_path = os.path.join(data_dir, selected_class, selected_image)
    image = Image.open(image_path)

    # Preprocess the image and convert it to PyTorch tensor
    processed_image = feature_extractor(images=image, return_tensors="pt")

    # Get the input label of the image (use the folder name directly as the label)
    label = selected_class

    # Generate adversarial image
    perturbations = create_adversarial_pattern(processed_image, label, model)
    adversarial_image = processed_image + 0.1 * perturbations
    adversarial_image = torch.clip(adversarial_image, 0.0, 1.0)

    # Generate adversarial attack using FGSM
    for adversarial_image_np in generate_adversarial_attack(processed_image, label, model):
        # Convert the adversarial attack to an image and display it
        adv_image = Image.fromarray(np.uint8(np.squeeze(adversarial_image_np) * 255.0))
        st.subheader(f"{selected_class} - Adversarial Image (FGSM)")
        st.image(adv_image, caption=f"{selected_class} - Adversarial (FGSM)", use_column_width=True)

# Main App
def main():
    data_dir = "./images/evaluation2"
    evaluate_model(data_dir)

if __name__ == "__main__":
    main()
