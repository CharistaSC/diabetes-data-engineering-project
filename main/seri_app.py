import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Load Pre-trained model from Huggingface and feature extractor
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Create the Streamlit App

# Page functions
def welcome_page():
    st.title("Diabetic Retinopathy Detection App")
    st.write("Welcome to the Diabetic Retinopathy Detection App! This app uses a pre-trained Vision Transformer model to classify eye fundus images for diabetic retinopathy.")
    st.write("Please use the navigation on the left to upload eye fundus images for prediction or to view the model evaluations.")

def upload_image_page():
    st.title("Upload Eye Fundus Images")
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    clear_button = st.button("Clear All")
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files

    for idx, uploaded_file in enumerate(st.session_state.uploaded_images):
        # Using cache here to prevent caching of images after they are removed
        image = Image.open(uploaded_file).cache()
        st.image(image, caption=f"Uploaded Image {idx+1}: {uploaded_file.name}", use_column_width=True)

        # Process the image and get predictions
        features = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**features)
        logits = outputs.logits
        probabilities = logits.softmax(dim=1)
        class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        prediction_probabilities = probabilities[0].tolist()
        st.bar_chart({class_names[i]: prediction_probabilities[i] for i in range(5)})
        st.write("Prediction Probabilities:")
        for i in range(5):
            st.write(f"{class_names[i]}: {prediction_probabilities[i]*100:.2f}%")

    if clear_button:
        st.session_state.uploaded_images = []
        st.experimental_rerun()

def evaluations_page():
    st.title("Model Evaluations")
    st.write("In this section, you can find the evaluations of the model, including its robustness to adversarial attacks, out-of-distribution detection, computational efficiency, model calibration, transfer learning evaluation, and ethical considerations.")

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
