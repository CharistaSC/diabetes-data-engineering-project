import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

# Load the pipeline with the fine-tuned model for image classification
pipe = pipeline("image-classification", model="rafalosa/diabetic-retinopathy-224-procnorm-vit")

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
    st.image(image, caption="Uploaded Image", use_column_width=True)

def evaluate_model(image_path):
    results = pipe(image_path)

    # Update class labels to full names and capitalize
    for result in results:
        result['label'] = result['label'].capitalize().replace("dr", "Diabetic Retinopathy")

    # Create a DataFrame to swap columns and align text to the left
    df = pd.DataFrame(results)
    df = df[['score', 'label']]
    df['label'] = df['label'].apply(lambda x: f"{x:<25}")
    st.write("Predicted Class and Confidence:")
    df['Index'] = range(1, len(df) + 1)  # Add a new column 'Index' to avoid the KeyError
    st.dataframe(df.set_index('Index'), height=300)  # Modify the height here and set index to 'Index'

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
            display_image(image_path)

            # Show the model predictions for each uploaded image
            st.header("Model Predictions")
            evaluate_model(image_path)

if __name__ == "__main__":
    main()
