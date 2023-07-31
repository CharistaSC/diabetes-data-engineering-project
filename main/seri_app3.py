import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import io
import torch
import numpy as np
import os
import tensorflow as tf

# Load Pre-trained model from Huggingface and feature extractor
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

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

# Create the Streamlit App

# Page functions
def welcome_page():
    st.title("Diabetic Retinopathy Detection App")
    st.write("Welcome to the Diabetic Retinopathy Detection App! This app uses a pre-trained Vision Transformer model to classify eye fundus images for diabetic retinopathy.")
    st.write("Please use the navigation on the left to upload eye fundus images for prediction or to view the model evaluations.")

def upload_image_page():
    st.title("Upload Eye Fundus Images")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image_data = uploaded_file.read()  # Read the image data as bytes
            image = Image.open(io.BytesIO(image_data))  # Create the Image object from bytes

            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Process the image and get predictions
            features = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**features)
            logits = outputs.logits
            probabilities = logits.softmax(dim=1)
            class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            prediction_probabilities = probabilities[0].tolist()

            # Display the predicted class and probabilities
            st.subheader("Prediction")
            predicted_class_idx = prediction_probabilities.index(max(prediction_probabilities))
            predicted_class_name = class_names[predicted_class_idx]
            st.write(f"Predicted Class: {predicted_class_name}")
            st.write("Prediction Probabilities:")
            for class_name, prob in zip(class_names, prediction_probabilities):
                st.write(f"{class_name}: {prob*100:.2f}%")

def evaluations_page():
    st.title("Model Evaluations")
    st.write("In this section, you can find the evaluations of the model, including adversarial image generation.")
    st.write("Select an evaluation type from the dropdown to view details:")
    evaluation_type = st.selectbox("Select Evaluation Type", options=["Adversarial Image (FGSM)"])

    if evaluation_type == "Adversarial Image (FGSM)":
        adversarial_attack_page()

def adversarial_attack_page():
    st.title("Adversarial Image (FGSM) Evaluation")
    st.write("Generating adversarial attacks using the FGSM method.")
    st.write("Select an evaluation class from the dropdown:")

    data_dir = os.path.abspath('./images/evaluation2')
    # Get the list of folders in the evaluation directory
    eval_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    # Create a dropdown to select the class
    selected_class = st.selectbox("Select a class:", eval_folders)

    # Get the list of image files in the selected folder
    image_files = [file for file in os.listdir(os.path.join(data_dir, selected_class)) if file.endswith(".jpeg")]

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

    # Get the input label of the image (use the folder name directly as the label)
    label = selected_class

    # Generate adversarial image
    perturbations = create_adversarial_pattern(processed_image, label, model)
    adversarial_image = processed_image + 0.1 * perturbations
    adversarial_image = torch.clip(adversarial_image, 0.0, 1.0)

    # Convert the adversarial attack to an image and display it
    adv_image = Image.fromarray(np.uint8(np.squeeze(adversarial_image) * 255.0))
    st.subheader(f"{selected_class} - Adversarial Image (FGSM)")
    st.image(adv_image, caption=f"{selected_class} - Adversarial (FGSM)", use_column_width=True)

# Main App
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Set the style for the navigation bar header and subheader
    st.sidebar.markdown("<h1 style='font-weight: bold; font-size: 30px;'>🔬👁️ Diabetic Retinopathy Detection App</h1>", unsafe_allow_html=True)
    st.sidebar.subheader("Navigation")

    page = st.sidebar.radio("Go to", options=["💫 Welcome", "💻 Upload Image", "📊 Evaluations"])

    if page == "💫 Welcome":
        welcome_page()
    elif page == "💻 Upload Image":
        upload_image_page()
    elif page == "📊 Evaluations":
        evaluations_page()

if __name__ == "__main__":
    main()
