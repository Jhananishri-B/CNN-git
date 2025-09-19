import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

model = load_model("model.h5")
le_dict = np.load("le.npy", allow_pickle=True).item()
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Butterfly Classifier 🦋", page_icon="🦋", layout="centered")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #FFDAB9 0%, #FFE4E1 100%); /* Peach to light pink */
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #FF6F61; /* Coral pink */
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF7F50, #FFB347); /* Coral to light orange */
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #FFB347, #FF7F50);
        transform: scale(1.05);
    }
    .prediction-box {
        background: linear-gradient(135deg, #FFDAB9, #FFB6C1); /* Peach to pink */
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #8B008B; /* Dark Magenta */
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦋 Butterfly Species Classifier")
st.markdown("<p style='text-align:center;font-size:18px;'>Upload an image of a butterfly and the app will predict its species.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    class_label = le_dict[class_idx]

    st.markdown(f'<div class="prediction-box">✨ Predicted Class: {class_label} ✨</div>', unsafe_allow_html=True)
