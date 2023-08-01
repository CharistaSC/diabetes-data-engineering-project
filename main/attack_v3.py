import streamlit as st
import torch
import torch.nn as nn
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
def preprocess_image(image):
    image = image.resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Function to perform FGSM attack
def fgsm_attack(model, image, epsilon, data_grad):
    perturbation = epsilon * data_grad.sign()
    adv_image = image + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image

# Function to process a single image and perform adversarial attack
def process_image(image_path, key=None):
    if key is None:
        key = image_path  # Use image_path as key if no key is provided

    image = Image.open(image_path)
    input_image = preprocess_image(image)
    orig_image = input_image.clone()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_image = input_image.to(device)

    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output.logits, 1)
    st.write(f"Original Image Prediction: {predicted_class.item()}")

    # Create a slider for epsilon value with a unique key based on the image path
    if key is None:
        key = image_path  # You can customize this key generation based on your preference

    # Button to trigger the adversarial attack with the current epsilon value
    if st.button(f"Generate Adversarial Image {key}"):
        input_image.requires_grad = True

        # Get the model's prediction for the adversarial image
        with torch.enable_grad():
            output = model(input_image)
            loss = nn.CrossEntropyLoss()(output.logits, predicted_class)
        model.zero_grad()
        loss.backward()

        # Generate the adversarial image using FGSM
        adv_image = fgsm_attack(model, input_image, epsilon, input_image.grad.data)

        with torch.no_grad():
            adv_output = model(adv_image)
        _, adv_predicted_class = torch.max(adv_output.logits, 1)
        st.write(f"Adversarial Image Prediction: {adv_predicted_class.item()}")

        adv_image = adv_image.cpu().squeeze(0)

        transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])
        adv_image = transform(adv_image)

        st.image([image, adv_image], caption=["Original Image", "Adversarial Image"], width=224)

        inputs = extractor(images=adv_image, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=1)
        class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        prediction_probabilities = probabilities[0].tolist()
        st.bar_chart({class_names[i]: prediction_probabilities[i] for i in range(5)})
        st.write("Prediction Probabilities:")
        for i in range(5):
            st.write(f"{class_names[i]}: {prediction_probabilities[i]*100:.2f}%")

def main():
    st.title("Adversarial Attack with FGSM and Pretrained Model Evaluation")

    # File uploader for single image
    st.write("Upload an image:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="fileUploader")

    # Text input for folder path
    st.sidebar.write("Enter a folder path:")
    folder_path = st.sidebar.text_input("Folder Path", key="folderInput")
    epsilon = st.slider("Epsilon", 0.0, 0.1, 0.01, 0.001)

    if uploaded_file is not None:
        process_image(uploaded_file)

    if folder_path and os.path.exists(folder_path):
        # Process all images in the folder and subfolders
        for root, _, files in os.walk(folder_path):
            for filename in files:
                image_path = os.path.join(root, filename)

                # Check if the file is an image
                if not any(image_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                    continue

                # Perform the adversarial attack and evaluate the pretrained model for the image
                process_image(image_path)

# Run the app
if __name__ == "__main__":
    main()
