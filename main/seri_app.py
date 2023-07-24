import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import io

# Load Pre-trained model from Huggingface and feature extractor
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Create the Streamlit App

# Page functions
def welcome_page():
    st.title("Diabetic Retinopathy Detection App")
    st.image("4 stages of diabetic retinopathy.png", caption="Image Source: https://www.dolmaneyecare.com/", use_column_width=True)
    
     # Set the font size for the welcome message
    st.markdown("<p style='font-size: 30px;'>Welcome to the Diabetic Retinopathy Detection App!</p>", unsafe_allow_html=True)
    
    st.write("This app uses a pre-trained Vision Transformer model to classify eye fundus images for diabetic retinopathy.")

    # Style the key features with bold and green ticks
    st.markdown("### Key Features:")
    st.write("✓ **Upload Eye Fundus Images for Prediction**")
    st.write("✓ **View Model Evaluations and Insights**")
    
    # Add left finger emoji
    st.write("👈 Please use the navigation on the left to get started.")




def upload_image_page():
    st.title("Upload Eye Fundus Images")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if "uploaded_images" not in st.session_state:
            st.session_state.uploaded_images = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Process the image and get predictions
            features = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**features)
            logits = outputs.logits
            probabilities = logits.softmax(dim=1)
            class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            prediction_probabilities = probabilities[0].tolist()

            # Store the image and evaluation result in the session state
            st.session_state.uploaded_images.append(
                {
                    "image": image,
                    "predictions": prediction_probabilities,
                    "class_names": class_names,
                }
            )

def display_evaluation_results():
    st.title("Evaluation Results")
    if "uploaded_images" in st.session_state:
        for idx, uploaded_image_data in enumerate(st.session_state.uploaded_images):
            image = uploaded_image_data["image"]
            predictions = uploaded_image_data["predictions"]
            class_names = uploaded_image_data["class_names"]

            st.image(image, caption=f"Uploaded Image {idx+1}", use_column_width=True)

            st.bar_chart({class_names[i]: predictions[i] for i in range(5)})

            st.subheader("Prediction Probabilities")
            col1, col2 = st.columns(2)
            space = st.empty()
            for i in range(5):
                table_content = f"<table style='border-collapse: collapse; width: 100%;'><tr><th style='width: 50%; padding: 12px; text-align: center; border: 1px solid black; background-color: #f0f0f0;'>{class_names[i]}</th><td style='width: 50%; padding: 12px; text-align: center; border: 1px solid black; background-color: #ffffff;'>{predictions[i]*100:.2f}%</td></tr></table>"
                col1.markdown(table_content, unsafe_allow_html=True)
                space.markdown("<br>", unsafe_allow_html=True)  # Add a space between rows

def evaluations_page():
    st.title("Model Evaluations")
    st.write("In this section, you can find the evaluations of the model related to the uploaded images.")
    st.write("This section provides a visual representation of the model's predictions and their probabilities for each uploaded image.")
    st.write("The bar chart shows the probability distribution of different classes for each image.")

    # Check if there are uploaded images in the session state
    if "uploaded_images" in st.session_state and len(st.session_state.uploaded_images) > 0:
        display_evaluation_results()
    else:
        st.write("Please upload eye fundus images for evaluation.")

# Main App
def main():
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
