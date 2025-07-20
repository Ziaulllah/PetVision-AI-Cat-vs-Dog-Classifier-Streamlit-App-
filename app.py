import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Cat vs Dog Classifier", page_icon="🐾", layout="wide")

# ---- FULL DARK THEME CSS ----
st.markdown("""
    <style>
    /* App Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    /* --- Option A: Hide Toolbar (Uncomment this to hide) ---
    header[data-testid="stHeader"] {
        display: none;
    }
    */

    /* --- Option B: Style Toolbar (Keep visible but dark) --- */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #fff;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f2027, #1e3c52, #2c5364);
        color: #fff;
    }
    section[data-testid="stSidebar"] * {
        color: #ddd;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 8px;
        border-radius: 12px;
    }
    button[data-baseweb="tab"] {
        color: #eee;
        font-weight: bold;
    }
    button[data-baseweb="tab"]:hover {
        color: #00e6e6;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(0, 230, 230, 0.1);
        border-radius: 8px;
    }

    /* Titles */
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #00e6e6;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0px 0px 15px #00ffff;
    }
    .subtitle {
        text-align: center;
        color: #ddd;
        font-size: 20px;
        margin-bottom: 30px;
    }

    /* Upload and Prediction Cards */
    .upload-box {
        text-align: center;
        background: rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(0, 230, 230, 0.4);
        margin-bottom: 20px;
        color: #ddd;
        transition: transform 0.2s;
    }
    .upload-box:hover {
        transform: scale(1.02);
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 0 40px rgba(0, 230, 230, 0.6);
        text-align: center;
        margin-top: 30px;
        color: #fff;
        animation: popIn 0.6s ease-in-out;
    }
    @keyframes popIn {
        0% {transform: scale(0.8); opacity: 0;}
        100% {transform: scale(1); opacity: 1;}
    }
    .confidence {
        font-size: 24px;
        font-weight: bold;
        color: #00e6e6;
        text-shadow: 0px 0px 8px #00ffff;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.markdown("## ⚡ About This App")
st.sidebar.info("""
                
This is a **Deep Learning App** built with:

- TensorFlow (CNN Model)
- Streamlit for Web UI
- Supports JPG, PNG, WEBP

**Created by:** Zia Ullah  
[LinkedIn](https://www.linkedin.com/in/engr-ziaullah-7672ab260)
""")

# ---- LOAD MODEL ----
@st.cache_resource
def load_cnn_model():
    return load_model("cat_dog_model.keras")

model = load_cnn_model()

# ---- PAGE HEADER ----
st.markdown("<div class='title'>🐶 Cat vs Dog Classifier 🐱</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered classification with a polished dark dashboard for social demos.</div>", unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 About Model", "ℹ️ Instructions"])

with tab1:
    st.markdown("<div class='upload-box'>Upload your image (JPG, PNG, WEBP)</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Analyzing with AI..."):
            prediction = model.predict(img_array)[0][0]

        confidence = round(float(prediction) * 100, 2) if prediction > 0.5 else round((1 - float(prediction)) * 100, 2)
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

        # Prediction card
        st.markdown(f"""
            <div class='prediction-card'>
                <h2 style="color:#00ffff; text-shadow:0px 0px 10px #00e6e6;">{label}</h2>
                <p class='confidence'>Confidence: {confidence}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))

    else:
        st.info("👋 Upload a Cat, Dog, or WEBP image above to start.")

with tab2:
    st.subheader("Model Details")
    st.write("""
    - **Architecture**: Convolutional Neural Network (3 Conv layers, 2 Dense layers)  
    - **Trained on**: Dogs vs Cats dataset  
    - **Input Size**: 256x256 RGB images  
    - **Output**: Binary classification (Dog or Cat)
    """)

with tab3:
    st.subheader("How to Use")
    st.write("""
    1. **Upload an Image**  
       Drag & drop or click "Browse Files" to upload a picture of a **cat or dog** 
       (Supports JPG, PNG, and WEBP formats)

    2. **AI Analyzes the Image**  
       Our **deep learning model (CNN)** instantly processes your image to detect if it’s a **cat or dog**.

    3. **View the Results**  
       The app shows a **clear prediction** (Cat 🐱 or Dog 🐶) along with a **confidence score** (how certain the AI is).  
       A **visual confidence gauge** helps you understand the result at a glance.

    """)


# ---- FOOTER ----
st.markdown("<div class='footer'>Powered by TensorFlow & Streamlit | Dark Gradient Edition</div>", unsafe_allow_html=True)
