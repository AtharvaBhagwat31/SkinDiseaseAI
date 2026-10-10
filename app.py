
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "skin_image_model_v2.keras"
CLASSES_PATH = ROOT / "models" / "skin_classes.json"
IMAGE_SIZE = (160, 160)

CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

st.set_page_config(
    page_title="DermaVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PREMIUM LIGHT UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
      radial-gradient(ellipse at 90% 5%, #d9f5f7 0%, transparent 35%),
      radial-gradient(ellipse at 5% 60%, #e8f1ff 0%, transparent 38%),
      linear-gradient(135deg, #f8fbff 0%, #eef7fa 55%, #f7fbff 100%);
    color: #17324d;
}
[data-testid="stHeader"] {
    background: rgba(248,251,255,0.82);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#e7f7fa,#f3f8ff);
    border-right: 1px solid #d5e8ef;
}
.hero {
    background: linear-gradient(115deg,rgba(255,255,255,.96),
                rgba(222,247,249,.88),rgba(233,241,255,.92));
    border: 1px solid rgba(255,255,255,.95);
    border-radius: 26px;
    padding: 35px 38px;
    box-shadow: 0 15px 45px rgba(38,91,120,.10);
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: "⚕  ◉  ✣";
    position: absolute;
    right: 25px;
    bottom: 8px;
    font-size: 65px;
    color: rgba(40,153,169,.10);
    letter-spacing: 10px;
}
.eyebrow {
    color: #128b98;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero h1 {
    color: #12304b;
    font-size: clamp(30px,4vw,48px);
    line-height: 1.12;
    margin: 12px 0;
    font-weight: 800;
}
.hero p {
    color: #58738a;
    font-size: 16px;
    max-width: 700px;
    line-height: 1.7;
}
.panel {
    background: rgba(255,255,255,.83);
    border: 1px solid #deedf2;
    border-radius: 21px;
    padding: 24px;
    box-shadow: 0 9px 28px rgba(39,80,105,.07);
    margin-bottom: 16px;
}
.metric {
    background: rgba(255,255,255,.90);
    border: 1px solid #dcebf1;
    border-radius: 18px;
    padding: 19px;
    min-height: 112px;
    box-shadow: 0 6px 20px rgba(39,80,105,.05);
}
.metric-label {
    font-size: 12px;
    color: #6c8496;
    font-weight: 600;
}
.metric-value {
    font-size: 22px;
    font-weight: 800;
    color: #153a55;
    margin-top: 9px;
}
h2, h3 {
    color: #173c56 !important;
}
.stButton > button {
    background: linear-gradient(100deg,#0e9aa7,#2879c7);
    color: white;
    border: 0;
    border-radius: 12px;
    min-height: 47px;
    font-weight: 700;
    box-shadow: 0 7px 17px rgba(21,133,165,.19);
}
.stButton > button:hover {
    color: white;
    border: 1px solid #0e9aa7;
    filter: brightness(1.04);
}
.stProgress > div > div > div {
    background: linear-gradient(90deg,#15a9ae,#458ce0);
}
div[data-testid="stFileUploader"] {
    background: #f7fcff;
    border: 1px dashed #90cbd6;
    border-radius: 15px;
    padding: 12px;
}
.small-muted {
    color: #698196;
    font-size: 13px;
    line-height: 1.6;
}
.footer {
    text-align: center;
    color: #7b91a2;
    padding: 20px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔬 DermaVision")
    st.caption("SKIN AI RESEARCH LAB")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Image Analysis", "Skin History", "About the Model"],
    )

    st.divider()
    st.markdown("### System status")
    st.success("Interface ready")

    if MODEL_PATH.exists():
        st.success("Model file detected")
    else:
        st.warning("Model file not found")

    st.markdown(
        '<p class="small-muted">Educational research prototype. '
        'Not a substitute for a dermatologist.</p>',
        unsafe_allow_html=True,
    )

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)

def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ AI-powered dermatology research</div>
  <h1>Understand skin.<br>Explore intelligent insights.</h1>
  <p>
    An educational skin-lesion image analysis platform built with deep learning.
    Upload a dermoscopic image to explore the model's predicted categories
    and learn about its limitations.
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown("### Your research workspace")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric">
          <div class="metric-label">AI ENGINE</div>
          <div class="metric-value">MobileNetV2</div>
          <p class="small-muted">Transfer-learning model</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric">
          <div class="metric-label">IMAGE INPUT</div>
          <div class="metric-value">160 × 160</div>
          <p class="small-muted">Resized model input</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric">
          <div class="metric-label">DATASET</div>
          <div class="metric-value">HAM10000</div>
          <p class="small-muted">Seven lesion categories</p>
        </div>""", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### ✨ Start an analysis")
        st.write(
            "Explore the model's response to a dermoscopic image. "
            "The output is experimental and must not be treated as a diagnosis."
        )
        if st.button("Open Image Analysis →", use_container_width=True):
            st.session_state["page"] = "Image Analysis"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🧬 Research workflow")
        st.write("01  Upload a dermoscopic image")
        st.write("02  Run the trained image model")
        st.write("03  Review the highest model scores")
        st.write("04  Seek clinical assessment when concerned")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "Safety note: this model has limited performance, especially for "
        "melanoma. A low score cannot rule out skin cancer."
    )

# ---------------- IMAGE ANALYSIS ----------------
elif page == "Image Analysis":
    st.markdown("### 🖼️ Dermoscopic image analysis")
    st.markdown(
        '<p class="small-muted">For research use, upload a dermoscopic image '
        'similar to those in the HAM10000 dataset.</p>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        st.error(
            "Required model files are missing. Check that "
            "skin_image_model_v2.keras and skin_classes.json are in the models folder."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        key="derma_upload",
    )

    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            preview_col, detail_col = st.columns([1, 1], gap="large")

            with preview_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Image preview")
                st.image(image, use_container_width=True)
                st.caption(f"Image dimensions: {image.width} × {image.height}")
                st.markdown("</div>", unsafe_allow_html=True)

            with detail_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Analysis")
                st.write("Run the trained model to view its top three outputs.")
                analyze = st.button(
                    "✦ Analyze Image",
                    type="primary",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            if analyze:
                with st.spinner("Running image model..."):
                    model = load_model()
                    class_codes = load_classes()
                    scores = model.predict(preprocess_image(image), verbose=0)[0]

                top = np.argsort(scores)[::-1][:3]
                st.markdown("### Model output")
                st.caption(
                    "These are raw model scores, not calibrated medical "
                    "probabilities or confirmed diagnoses."
                )

                for rank, idx in enumerate(top, start=1):
                    code = class_codes[int(idx)]
                    name = CLASS_NAMES.get(code, code)
                    st.markdown(f"**{rank}. {name}**")
                    st.progress(float(np.clip(scores[idx], 0, 1)))
                    st.caption(f"Model score: {scores[idx] * 100:.1f}%")

                st.warning(
                    "This research model has limited sensitivity for melanoma. "
                    "Its output cannot determine whether a lesion is safe. "
                    "Have new, changing, bleeding, or non-healing lesions "
                    "assessed by a qualified healthcare professional."
                )
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Ready when you are")
        st.write(
            "Choose a JPG or PNG file above. Dermoscopic images are the "
            "intended input; ordinary phone photos may perform poorly."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- SKIN HISTORY ----------------
elif page == "Skin History":
    st.markdown("### 📝 Skin history questionnaire")
    st.write(
        "This questionnaire offers general safety guidance only. "
        "It does not feed into the AI model or establish a diagnosis."
    )

    with st.form("skin_history_form"):
        duration = st.selectbox(
            "How long have you noticed the spot?",
            ["Choose...", "Less than 2 weeks", "2 weeks to 3 months",
             "More than 3 months", "Not sure"],
        )
        changing = st.radio(
            "Has it changed in size, shape, or colour?",
            ["Not sure", "No", "Yes"],
        )
        bleeding = st.radio(
            "Does it bleed without injury or form a persistent sore?",
            ["Not sure", "No", "Yes"],
        )
        symptoms = st.multiselect(
            "Other symptoms",
            ["Itching", "Pain or tenderness", "Crusting", "None", "Not sure"],
        )
        submitted = st.form_submit_button(
            "Get general guidance",
            use_container_width=True,
        )

    if submitted:
        if changing == "Yes" or bleeding == "Yes":
            st.error(
                "Please arrange prompt assessment by a qualified healthcare "
                "professional. The AI model cannot determine the seriousness "
                "of a lesion."
            )
        elif (
            changing == "Not sure"
            or bleeding == "Not sure"
            or duration in ["Choose...", "Not sure"]
        ):
            st.info(
                "If you are uncertain, consider professional assessment, "
                "especially if the spot persists or changes."
            )
        else:
            st.info(
                "These answers cannot establish that the spot is harmless. "
                "Seek medical assessment if it persists, changes, worries you, "
                "or develops new symptoms."
            )

        st.caption(
            "Avoid picking the spot, protect your skin from excessive sun, "
            "and do not start treatment based on this questionnaire."
        )

# ---------------- ABOUT ----------------
elif page == "About the Model":
    st.markdown("### 🧠 About this research prototype")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Model architecture")
    st.write(
        "MobileNetV2 with transfer learning, trained on dermoscopic images "
        "from the HAM10000 dataset."
    )
    st.markdown("#### Categories")
    for code, name in CLASS_NAMES.items():
        st.write(f"**{code}** — {name}")

    st.markdown("#### Important limitations")
    st.write(
        "- The model is not clinically validated."
    )
    st.write(
        "- The dataset contains seven lesion categories, not every skin disease."
    )
    st.write(
        "- Performance varies across categories; melanoma detection is a "
        "particular weakness in this model."
    )
    st.write(
        "- It may perform poorly on ordinary phone photographs, different "
        "skin tones, or images unlike the training data."
    )
    st.write(
        "- Model scores should not be interpreted as an individual's cancer risk."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
DERMAVISION AI · EDUCATIONAL RESEARCH PROTOTYPE · NOT FOR CLINICAL DECISIONS
</div>
""", unsafe_allow_html=True)
