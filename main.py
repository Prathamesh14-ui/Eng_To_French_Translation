import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from transformers import pipeline

# Point pytesseract to the tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the translation model
translation_pipeline = pipeline("translation_en_to_fr")

def extract_text(image):
    """Extracts text from an image using Tesseract OCR."""
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(thresh)
    return extracted_text.strip()

def translate_text(text):
    """Translates text from English to French using the translation model."""
    if text:
        # Split the text into smaller chunks for better translation (if needed)
        sentences = text.split('\n')
        translated_sentences = []
        for sentence in sentences:
            if sentence.strip():
                # Perform the translation for each sentence
                translation = translation_pipeline(sentence, max_length=512)  # Increased max_length
                translated_sentences.append(translation[0]['translation_text'])
        # Join all the translated sentences
        return "\n".join(translated_sentences)
    return ""

# Streamlit UI
st.title("Image Text Extraction and Translation to French")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to OpenCV format
    image_cv = np.array(image)

    # Extract text from the image
    extracted_text = extract_text(image_cv)
    st.subheader("Extracted Text:")
    st.text_area("Extracted Text", extracted_text, height=200)

    # Translate the extracted text
    if extracted_text:
        translated_text = translate_text(extracted_text)
        st.subheader("Translated Text in French:")
        st.text_area("Translated Text", translated_text, height=200)
    else:
        st.warning("No text was extracted from the image to translate.")
