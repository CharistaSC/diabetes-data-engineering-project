import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Load the pretrained model
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to the model's input size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to perform FGSM attack
def fgsm_attack(model, image, epsilon, data_grad):
    perturbation = epsilon * data_grad.sign()
    adv_image = image + perturbation
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image

# Streamlit app
def main():
    st.title("Adversarial Attack with FGSM")

    # File uploader
    st.write("Upload an image:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        input_image = preprocess_image(image)
        orig_image = input_image.clone()  # Make a copy of the original input image tensor

        # Convert the model to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_image = input_image.to(device)

        # Get the model's prediction for the input image
        with torch.no_grad():
            output = model(input_image)
        _, predicted_class = torch.max(output.logits, 1)
        st.write(f"Original Image Prediction: {predicted_class.item()}")

        # Create a slider for epsilon value
        epsilon = st.slider("Epsilon", 0.0, 0.1, 0.01, 0.001)

        # Button to trigger the adversarial attack with the current epsilon value
        if st.button("Generate Adversarial Image"):
            input_image.requires_grad = True

            # Get the model's prediction for the adversarial image
            with torch.enable_grad():
                output = model(input_image)
                loss = nn.CrossEntropyLoss()(output.logits, predicted_class)
            model.zero_grad()
            loss.backward()

            # Generate the adversarial image using FGSM
            adv_image = fgsm_attack(model, input_image, epsilon, input_image.grad.data)

            # Get the model's prediction for the adversarial image
            with torch.no_grad():
                adv_output = model(adv_image)
            _, adv_predicted_class = torch.max(adv_output.logits, 1)
            st.write(f"Adversarial Image Prediction: {adv_predicted_class.item()}")

            # Convert the adversarial image back to CPU for visualization
            adv_image = adv_image.cpu().squeeze(0)

            # Denormalize the adversarial image for display
            transform = transforms.Compose([
                transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                transforms.ToPILImage()
            ])
            adv_image = transform(adv_image)

            # Display original and adversarial images
            st.image([image, adv_image], caption=["Original Image", "Adversarial Image"], width=224)

            # Process the adversarial image and get predictions
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

if __name__ == "__main__":
    main()
