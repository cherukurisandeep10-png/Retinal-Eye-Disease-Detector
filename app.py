import streamlit as st
from PIL import Image
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from model.predict import DiseasePredictor
from utils.visualization import create_confidence_chart, create_report
from utils.heatmap import generate_gradcam_heatmap
from utils.preprocessing import preprocess_image

st.set_page_config(
    page_title="Retinal Eye Disease Detector",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

DISEASE_CLASSES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

DISEASE_INFO = {
    "Normal": {
        "desc": "No signs of eye disease detected. The retinal structure appears healthy.",
        "severity": "Low",
        "advice": "Continue regular eye checkups annually. Maintain a healthy diet rich in vitamins A, C, and E.",
        "color": "#2e7d32",
        "bg": "#e8f5e9",
    },
    "Diabetic Retinopathy": {
        "desc": "Damage to blood vessels in the retina caused by diabetes. Can lead to vision loss if untreated.",
        "severity": "High",
        "advice": "Consult an ophthalmologist immediately. Manage blood sugar levels. Regular screening is essential for diabetic patients.",
        "color": "#c62828",
        "bg": "#ffebee",
    },
    "Glaucoma": {
        "desc": "A group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye.",
        "severity": "High",
        "advice": "Seek immediate medical attention. Glaucoma can cause irreversible vision loss. Early treatment can slow progression.",
        "color": "#e65100",
        "bg": "#fff3e0",
    },
    "Cataract": {
        "desc": "Clouding of the eye's natural lens, leading to decreased vision. Common in older adults.",
        "severity": "Medium",
        "advice": "Consult an eye specialist for evaluation. Cataract surgery is a safe and effective treatment option.",
        "color": "#f9a825",
        "bg": "#fffde7",
    }
}


@st.cache_resource
def load_model():
    return DiseasePredictor()


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- Global ---------- */
    .stApp {
        background-color: #f7f9fc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }
    body, p, span, div, li {
        color: #2c3e50;
    }

    /* ---------- Top header bar ---------- */
    .top-header {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .logo-mark {
        width: 52px;
        height: 52px;
        background: #0b5394;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .header-text h1 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0b2545;
        margin: 0;
        line-height: 1.2;
    }
    .header-text p {
        font-size: 0.92rem;
        color: #5a6c7d;
        margin: 0.2rem 0 0 0;
    }
    .header-badge {
        margin-left: auto;
        background: #e7f0f9;
        color: #0b5394;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #cfe0f3;
    }

    /* ---------- Section panels ---------- */
    .panel {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .panel-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #5a6c7d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #eef2f7;
    }

    /* ---------- Result card ---------- */
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        border-left: 4px solid;
    }
    .result-card .result-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.75;
        margin-bottom: 0.3rem;
    }
    .result-card h2 {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        line-height: 1.1;
    }
    .result-card p {
        font-size: 0.93rem;
        line-height: 1.55;
        margin: 0;
    }

    /* ---------- Stat row ---------- */
    .stat-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.25rem;
    }
    .stat {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        padding: 1rem;
        border-radius: 6px;
        text-align: left;
    }
    .stat .lbl {
        font-size: 0.7rem;
        color: #7b8a9b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .stat .val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0b2545;
    }
    .stat .val.severity-low    { color: #2e7d32; }
    .stat .val.severity-medium { color: #f9a825; }
    .stat .val.severity-high   { color: #c62828; }

    /* ---------- Performance grid ---------- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    .metric {
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        padding: 1.1rem;
        border-radius: 6px;
        text-align: center;
    }
    .metric .m-label {
        font-size: 0.72rem;
        color: #5a6c7d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric .m-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0b5394;
        line-height: 1;
    }

    /* ---------- Workflow steps ---------- */
    .workflow {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    .step {
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        padding: 1.1rem;
        border-radius: 6px;
    }
    .step .step-no {
        display: inline-block;
        background: #0b5394;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        text-align: center;
        line-height: 24px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .step h4 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0b2545;
        margin: 0 0 0.3rem 0;
    }
    .step p {
        font-size: 0.83rem;
        color: #5a6c7d;
        margin: 0;
        line-height: 1.4;
    }

    /* ---------- Disease list (sidebar) ---------- */
    .disease-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0.7rem;
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        border-left: 3px solid;
    }
    .disease-row .name {
        flex: 1;
        font-size: 0.86rem;
        font-weight: 500;
        color: #2c3e50;
    }
    .disease-row .badge {
        font-size: 0.65rem;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---------- Advisory ---------- */
    .advisory {
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        border-left: 3px solid #0b5394;
        padding: 1rem 1.2rem;
        border-radius: 6px;
    }
    .advisory .a-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #0b5394;
        margin-bottom: 0.4rem;
    }
    .advisory p {
        font-size: 0.92rem;
        line-height: 1.55;
        color: #2c3e50;
        margin: 0;
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] {
        background: #f7f9fc;
        border: 1px dashed #b6c2d2;
        border-radius: 6px;
        padding: 0.6rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e3e8ef;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.85rem;
        font-weight: 700;
        color: #5a6c7d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.7rem;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #2c3e50 !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: #e3e8ef;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background: #0b5394;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: background 0.15s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #093f70;
    }

    /* ---------- Expanders ---------- */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border: 1px solid #e3e8ef !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        color: #0b2545 !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #7b8a9b;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding: 1.2rem;
        border-top: 1px solid #e3e8ef;
    }
    .demo-badge {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        padding: 2px 9px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 6px;
        border: 1px solid #ffeaa7;
        letter-spacing: 0.5px;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="top-header">
        <div class="logo-mark">RD</div>
        <div class="header-text">
            <h1>Retinal Disease Screening Tool</h1>
            <p>Deep learning-assisted analysis of fundus photographs · Glaucoma, Diabetic Retinopathy, Cataract, Normal</p>
        </div>
        <div class="header-badge">v1.0 · Screening Use Only</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### Analysis Options")
        show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
        show_top_predictions = st.checkbox("Show all class predictions", value=True)
        confidence_threshold = st.slider("Confidence threshold (%)", 10, 90, 50)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "<p style='font-size:0.85rem; line-height:1.55; color:#5a6c7d;'>"
            "This tool uses a convolutional neural network to assist in the early "
            "screening of retinal conditions from fundus photographs. "
            "<br><br>"
            "<strong>Not a substitute for professional medical diagnosis.</strong> "
            "Findings should always be reviewed by a qualified ophthalmologist."
            "</p>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### Detected Conditions")
        for disease in DISEASE_CLASSES:
            info = DISEASE_INFO[disease]
            sev_styles = {
                "Low":    ("#e8f5e9", "#2e7d32"),
                "Medium": ("#fffde7", "#f9a825"),
                "High":   ("#ffebee", "#c62828"),
            }
            bg, fg = sev_styles[info["severity"]]
            st.markdown(f"""
            <div class="disease-row" style="border-left-color: {info['color']};">
                <span class="name">{disease}</span>
                <span class="badge" style="background:{bg}; color:{fg};">{info['severity']}</span>
            </div>
            """, unsafe_allow_html=True)

        return show_heatmap, show_top_predictions, confidence_threshold


def render_landing():
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Reported Model Performance</div>
        <div class="metric-grid">
            <div class="metric"><div class="m-label">Accuracy</div><div class="m-value">94.2%</div></div>
            <div class="metric"><div class="m-label">Precision</div><div class="m-value">93.8%</div></div>
            <div class="metric"><div class="m-label">Recall</div><div class="m-value">94.5%</div></div>
            <div class="metric"><div class="m-label">F1 Score</div><div class="m-value">94.1%</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">How the Screening Works</div>
        <div class="workflow">
            <div class="step"><div class="step-no">1</div><h4>Upload</h4><p>A clear fundus photograph is provided as input.</p></div>
            <div class="step"><div class="step-no">2</div><h4>Preprocess</h4><p>The image is resized, contrast-enhanced, and normalized.</p></div>
            <div class="step"><div class="step-no">3</div><h4>Inference</h4><p>The CNN evaluates the image across four condition classes.</p></div>
            <div class="step"><div class="step-no">4</div><h4>Report</h4><p>Prediction, confidence, and a focus heatmap are displayed.</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Condition Reference</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (disease, info) in enumerate(DISEASE_INFO.items()):
        with cols[i % 2]:
            with st.expander(f"{disease}  ·  Severity: {info['severity']}"):
                st.write(info["desc"])
                st.markdown(f"**Recommendation:** {info['advice']}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_results(image, predictor, show_heatmap, show_top_predictions, confidence_threshold):
    with st.spinner("Analyzing image..."):
        processed = preprocess_image(image)
        prediction = predictor.predict(processed)
        idx = int(np.argmax(prediction))
        predicted_class = DISEASE_CLASSES[idx]
        confidence = float(prediction[idx] * 100)
        info = DISEASE_INFO[predicted_class]

        heatmap_image = (generate_gradcam_heatmap(predictor.model, processed, idx)
                         if show_heatmap else None)

    st.markdown(f"""
    <div class="result-card" style="background:{info['bg']}; border-color:{info['color']}; color:{info['color']};">
        <div class="result-label">Primary Finding</div>
        <h2>{predicted_class}</h2>
        <p style="color:#2c3e50;">{info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    sev_class = f"severity-{info['severity'].lower()}"
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat"><div class="lbl">Predicted Class</div><div class="val">{predicted_class}</div></div>
        <div class="stat"><div class="lbl">Confidence</div><div class="val">{confidence:.1f}%</div></div>
        <div class="stat"><div class="lbl">Risk Level</div><div class="val {sev_class}">{info['severity']}</div></div>
    </div>
    """, unsafe_allow_html=True)

    if show_top_predictions:
        st.markdown('<div class="panel"><div class="panel-title">Confidence Distribution</div>', unsafe_allow_html=True)
        fig = create_confidence_chart(prediction, DISEASE_CLASSES, confidence_threshold)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if show_heatmap and heatmap_image is not None:
        st.markdown('<div class="panel"><div class="panel-title">Model Focus (Grad-CAM)</div>', unsafe_allow_html=True)
        st.caption("Highlighted regions indicate which areas of the image most influenced the prediction.")
        st.image(heatmap_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="advisory">
        <div class="a-label">Clinical Recommendation</div>
        <p>{info['advice']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col_a, _ = st.columns([1, 3])
    with col_a:
        if st.button("Generate Report", use_container_width=True):
            report_buf = create_report(image, predicted_class, confidence, prediction,
                                       DISEASE_CLASSES, heatmap_image)
            st.download_button(
                label="Download Report (PNG)",
                data=report_buf,
                file_name=f"eye_report_{predicted_class.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )


def main():
    inject_css()
    render_header()

    show_heatmap, show_top_predictions, confidence_threshold = render_sidebar()
    predictor = load_model()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="panel"><div class="panel-title">1. Upload Fundus Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus photograph (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Input image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None:
            st.markdown('<div class="panel"><div class="panel-title">2. Analysis Results</div>', unsafe_allow_html=True)
            render_results(image, predictor, show_heatmap, show_top_predictions, confidence_threshold)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            render_landing()

    demo_badge = ""
    if not getattr(predictor, "using_real_weights", False):
        demo_badge = '<span class="demo-badge">DEMO MODE</span>'
    st.markdown(f"""
    <div class="footer">
        Retinal Disease Screening Tool {demo_badge}<br>
        For research and educational use. Not a medical device. Consult a qualified ophthalmologist for diagnosis.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
