import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import base64
import io
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def resize_with_aspect_ratio(image, max_size):
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return image.resize((new_width, new_height), Image.ANTIALIAS)

def display_image(image_path):
    image = Image.open(image_path)

    # Resize the image for display, preserving aspect ratio
    image = resize_with_aspect_ratio(image, 300)

    # Convert the image to a base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    return image_base64  # Return the base64 string of the resized image

def evaluate_model(image_base64):
    # Load the pipeline with the fine-tuned model for image classification
    pipe = pipeline("image-classification", model="rafalosa/diabetic-retinopathy-224-procnorm-vit")

    # Perform the prediction using the base64 string as input
    results = pipe(image_base64)

    # Update class labels to full names and capitalize
    for result in results:
        result['label'] = result['label'].capitalize().replace("dr", "Diabetic Retinopathy")

    # Create a DataFrame to swap columns and align text to the left
    df = pd.DataFrame(results)
    df = df[['label', 'score']]
    df['label'] = df['label'].apply(lambda x: f"{x:<25}")

    # Display the Predicted Class and No Diabetic Retinopathy with their scores
    predicted_class = df.loc[0, 'label']
    predicted_score = df.loc[0, 'score']
    st.write(f"#### Predicted Class: {predicted_class} (Score: {predicted_score:.2f})")
    
    # Display the rest of the table without the index column and remove null cells
    st.write("#### Confidence Scores:")
    st.dataframe(df.iloc[1:].reset_index(drop=True).dropna())  # Reset index, drop null cells

def fgsm_attack(model, image, epsilon):
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    image /= 255.0

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)

    # Get the gradient of the loss with respect to the input image
    gradient = tape.gradient(prediction, image)

    # Create perturbation in the direction that maximizes the loss
    perturbed_image = image + epsilon * tf.sign(gradient)
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

    return perturbed_image[0]

# Create the main Streamlit app
def main():
    st.title("Diabetic Retinopathy Detection")

    # Upload multiple images
    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            st.header(f"Uploaded Image: {uploaded_file.name}")
            image_path = f"uploaded_image_{uploaded_file.name}.{uploaded_file.type.split('/')[-1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Convert the image to a base64 string and get the base64 string of the resized image
            image_base64 = display_image(image_path)

            # Show the model predictions for the uploaded image
            st.write("### Model Predictions")
            evaluate_model(image_base64)
            
              # Perform FGSM attack
            epsilon = 0.03  # Adjust the epsilon value to control the strength of the attack
            image = Image.open(image_path)

            # Load the model using TensorFlow Hub
            model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
            model = hub.load(model_url)

            # Convert the image to a NumPy array
            image_np = np.array(image)

            # Generate perturbed image using FGSM
            perturbed_image_np = fgsm_attack(model, image_np, epsilon)
            perturbed_image_pil = Image.fromarray(np.uint8(perturbed_image_np * 255))

            # Display the original and perturbed images
            st.image([image, perturbed_image_pil], caption=["Original Image", "Perturbed Image"], use_column_width=True)

            # Show the model predictions for the original and perturbed images
            st.write("### Model Predictions for Original Image")
            evaluate_model(image_base64)

            st.write("### Model Predictions for Perturbed Image")
            perturbed_image_path = "perturbed_image.jpg"
            perturbed_image_pil.save(perturbed_image_path)
            evaluate_model(perturbed_image_path)

if __name__ == "__main__":
    main()
