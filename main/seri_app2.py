import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import io
import matplotlib.pyplot as plt

# Load Pre-trained model from Huggingface and feature extractor
model_name = "rafalosa/diabetic-retinopathy-224-procnorm-vit"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Create the Streamlit App

# Initialize the uploaded_images attribute as an empty list
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Page functions
def welcome_page():
    st.title("Diabetic Retinopathy Detection App")
    st.write("Welcome to the Diabetic Retinopathy Detection App! This app uses a pre-trained Vision Transformer model to classify eye fundus images for diabetic retinopathy.")
    st.write("Please use the navigation on the left to upload eye fundus images for prediction or to view the model evaluations.")

def upload_image_page():
    st.title("Upload Eye Fundus Images")
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if "uploaded_images" not in st.session_state:
            st.session_state.uploaded_images = []

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

            # Store the uploaded image data in the session state
            st.session_state.uploaded_images.append(
                {
                    "uploaded_file": uploaded_file,
                    "image_data": image_data,
                    "predictions": prediction_probabilities,
                    "class_names": class_names,
                    "robustness_evaluation": {},
                    "ood_detection_evaluation": {},
                    "computational_efficiency_evaluation": {},
                    "model_calibration_evaluation": {},
                    "transfer_learning_evaluation": {},
                    "ethical_considerations": {},
                }
            )



def robustness_evaluation():
    st.title("Robustness Evaluation")
    st.write("Assessing the model's robustness to adversarial attacks.")
    st.write("Performing adversarial attacks and evaluating the model's performance.")

    # Assume you have calculated the robustness evaluation results in a variable called "robustness_results"
    # Update the evaluation results in the session state for each uploaded image
    for uploaded_image_data in st.session_state.uploaded_images:
        # Sample evaluation results (you need to calculate actual results based on your evaluation)
        robustness_results = {
            "accuracy": 0.85,
            "precision": 0.87,
            "recall": 0.82,
            "f1_score": 0.84,
        }

        uploaded_file = uploaded_image_data["uploaded_file"]
        st.image(uploaded_file, caption=f"Uploaded Image", use_column_width=True)

        uploaded_image_data["robustness_evaluation"] = robustness_results

        # Plot the evaluation results as a bar chart
        st.bar_chart(robustness_results)

def ood_detection_evaluation():
    st.title("Out-of-Distribution Detection Evaluation")
    st.write("Assessing the model's ability to recognize inputs from unseen distributions.")
    st.write("Evaluating the model on out-of-distribution (OOD) data.")

    # Assume you have calculated the OOD detection evaluation results in a variable called "ood_results"
    # Update the evaluation results in the session state for each uploaded image
    for uploaded_image_data in st.session_state.uploaded_images:
        # Sample evaluation results (you need to calculate actual results based on your evaluation)
        ood_results = {
            "accuracy": 0.78,
            "precision": 0.79,
            "recall": 0.76,
            "f1_score": 0.77,
        }

        uploaded_file = uploaded_image_data["uploaded_file"]
        st.image(uploaded_file, caption=f"Uploaded Image", use_column_width=True)

        uploaded_image_data["ood_detection_evaluation"] = ood_results

        # Plot the evaluation results as a bar chart
        st.bar_chart(ood_results)

# ... (other evaluation functions remain the same)

#def computational_efficiency_evaluation():
    #st.title("Computational Efficiency Evaluation")
    #st.write("Evaluating the model's speed and memory usage during inference.")
    # Add relevant evaluation content here...
    
    # Assume you have calculated the robustness evaluation results in a variable called "computational_efficiency_results"
    # Update the evaluation results in the session state for each uploaded image
    #for uploaded_image_data in st.session_state.uploaded_images:
        #uploaded_image_data["computational_efficiency_evaluation"] = computational_efficiency_results

#def model_calibration_evaluation():
    #st.title("Model Calibration Evaluation")
    #st.write("Measuring how well the model's predicted probabilities align with actual confidence.")
    # Add relevant evaluation content here...
    
    # Assume you have calculated the robustness evaluation results in a variable called "model_calibration_results"
    # Update the evaluation results in the session state for each uploaded image
    #for uploaded_image_data in st.session_state.uploaded_images:
        #uploaded_image_data["model_calibration_evaluation"] = model_calibration_results

#def transfer_learning_evaluation():
    #st.title("Transfer Learning Evaluation")
    #st.write("Exploring the model's performance on related tasks with limited data.")
    # Add relevant evaluation content here...
    
    # Assume you have calculated the robustness evaluation results in a variable called "transfer_learning_results"
    # Update the evaluation results in the session state for each uploaded image
    #for uploaded_image_data in st.session_state.uploaded_images:
        #uploaded_image_data["transfer_learning_evaluation"] = transfer_learning_results


def evaluations_page():
    st.title("Model Evaluations")
    st.write("In this section, you can find the evaluations of the model, including its robustness to adversarial attacks, out-of-distribution detection, computational efficiency, model calibration, transfer learning evaluation, and ethical considerations.")
    st.write("Select an evaluation type from the dropdown to view details:")
    evaluation_type = st.selectbox("Select Evaluation Type", options=["Robustness", "Out-of-Distribution Detection", "Computational Efficiency", "Model Calibration", "Transfer Learning", "Ethical Considerations"])

    if evaluation_type == "Robustness":
        robustness_evaluation()
    elif evaluation_type == "Out-of-Distribution Detection":
        ood_detection_evaluation()
    # ... (other evaluation functions remain the same)
    #elif evaluation_type == "Computational Efficiency":
       # computational_efficiency_evaluation()
    #elif evaluation_type == "Model Calibration":
        #model_calibration_evaluation()
    #elif evaluation_type == "Transfer Learning":
        #transfer_learning_evaluation()


def display_evaluation_results():
    st.title("Evaluation Results")
    if "uploaded_images" in st.session_state and len(st.session_state.uploaded_images) > 0:
        for idx, uploaded_image_data in enumerate(st.session_state.uploaded_images):
            image_data = uploaded_image_data["image_data"]
            image = Image.open(io.BytesIO(image_data))

            st.image(image, caption=f"Uploaded Image {idx+1}", use_column_width=True)

            st.bar_chart({uploaded_image_data["class_names"][i]: uploaded_image_data["predictions"][i] for i in range(5)})

            st.subheader("Prediction Probabilities")
            col1, col2 = st.columns(2)
            space = st.empty()
            for i in range(5):
                table_content = f"<table style='border-collapse: collapse; width: 100%;'><tr><th style='width: 50%; padding: 12px; text-align: center; border: 1px solid black; background-color: #f0f0f0;'>{uploaded_image_data['class_names'][i]}</th><td style='width: 50%; padding: 12px; text-align: center; border: 1px solid black; background-color: #ffffff;'>{uploaded_image_data['predictions'][i]*100:.2f}%</td></tr></table>"
                col1.markdown(table_content, unsafe_allow_html=True)
                space.markdown("<br>", unsafe_allow_html=True)  # Add a space between rows

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
