import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# Load Pre-trained model from Huggingface and image processor
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Create the Streamlit App

# Page functions
def welcome_page():
    st.title("Diabetic Retinopathy Detection App")
    st.image("4 stages of diabetic retinopathy.png", caption="Image Source: https://www.dolmaneyecare.com/", use_column_width=True)
    
    #done by
    st.markdown("<p style='font-size: 35px;'>Done By:</p>", unsafe_allow_html=True)
    st.write("🧑‍💼Osama Rasheed Khan, 2203385")
    st.write("🧕Seri Hanzalah Bte Haniffah, 2201601")
    st.write("👨‍💼Tian Yue Xiao Bryon, 2201615")
    
     # Set the font size for the welcome message
    st.markdown("<p style='font-size: 30px;'>Welcome to the Diabetic Retinopathy Detection App!</p>", unsafe_allow_html=True)
    
    st.write("This app uses a pre-trained Vision Transformer model to classify eye fundus images for diabetic retinopathy.")

    # Style the key features with bold and green ticks
    st.markdown("### Key Features:")
    st.write("✓ Upload Eye Fundus Images for Prediction")
    st.write("✓ View Model Evaluations and Insights")
    
    # Add left finger emoji
    st.write("👈 Please use the navigation on the left to get started.")
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

def evaluations_page():
    st.title("Model Evaluations")
    st.write("In this section, you can find the evaluations of the model, including its robustness to adversarial attacks, out-of-distribution detection, computational efficiency, model calibration, transfer learning evaluation, and ethical considerations.")

# Main App
def main():
    # Set the style for the navigation bar header and subheader
    st.sidebar.markdown("<h1 style='font-weight: bold; font-size: 30px;'>🔬👁 Diabetic Retinopathy Detection App</h1>", unsafe_allow_html=True)
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
    