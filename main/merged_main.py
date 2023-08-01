import os
import io
import torch
import numpy as np
import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from transformers import ViTFeatureExtractor, ViTImageProcessor

# Set up tkinter
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

# Function to show the folder picker dialog
def pick_folder():
    dirname = filedialog.askdirectory(master=root)
    return dirname

# Load Pre-trained model from Huggingface and image processor
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)


# Page functions
def welcome_page():
    st.title("Diabetic Retinopathy Detection App")
    st.write("Welcome to the Diabetic Retinopathy Detection App! This app uses a pre-trained Vision Transformer model to classify eye fundus images for diabetic retinopathy.")
    st.write("Please use the navigation on the left to upload an eye fundus image for prediction or to view the model evaluations.")

def upload_image_page():
    st.title("Upload Eye Fundus Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image and get predictions
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=1)
        class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        prediction_probabilities = probabilities[0].tolist()
        st.bar_chart({class_names[i]: prediction_probabilities[i] for i in range(5)})
        st.write("Prediction Probabilities:")
        for i in range(5):
            st.write(f"{class_names[i]}: {prediction_probabilities[i]*100:.2f}%")

# Evaluations page with folder picker
def evaluations_page():
    st.title("Model Evaluations")
    st.write("In this section, you can find the evaluations of the model, including its robustness to adversarial attacks, out-of-distribution detection, computational efficiency, model calibration, transfer learning evaluation, and ethical considerations.")
    
    # Use st.button for folder picker
    clicked = st.button('Select Evaluation Folder')
    
    if clicked:
        # Get the selected folder path from the folder picker dialog
        data_dir_path = pick_folder()

        # Check if the selected path is a valid directory
        if os.path.isdir(data_dir_path):
            # Pass the selected folder path to the evaluate_model function
            evaluate_model(data_dir_path)
        else:
            st.error("Please select a valid folder.")

def create_adversarial_pattern(image, label, model, epsilon):
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
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", options=["Welcome", "Upload Image", "Evaluations"])
    
    if page == "Welcome":
        welcome_page()
    elif page == "Upload Image":
        upload_image_page()
    elif page == "Evaluations":
        evaluations_page()

if __name__ == "__main__":
    main()
