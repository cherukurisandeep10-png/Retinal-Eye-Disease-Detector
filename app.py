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
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DISEASE_CLASSES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]

DISEASE_INFO = {
    "Normal": {
        "desc": "No signs of eye disease detected. The retinal structure appears healthy.",
        "severity": "Low",
        "advice": "Continue regular eye checkups annually. Maintain a healthy diet rich in vitamins A, C, and E.",
        "icon": "✅",
        "color": "#10b981",
    },
    "Diabetic Retinopathy": {
        "desc": "Damage to blood vessels in the retina caused by diabetes. Can lead to vision loss if untreated.",
        "severity": "High",
        "advice": "Consult an ophthalmologist immediately. Manage blood sugar levels. Regular screening is essential for diabetic patients.",
        "icon": "🩸",
        "color": "#ef4444",
    },
    "Glaucoma": {
        "desc": "A group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye.",
        "severity": "High",
        "advice": "Seek immediate medical attention. Glaucoma can cause irreversible vision loss. Early treatment can slow progression.",
        "icon": "⚠️",
        "color": "#f97316",
    },
    "Cataract": {
        "desc": "Clouding of the eye's natural lens, leading to decreased vision. Common in older adults.",
        "severity": "Medium",
        "advice": "Consult an eye specialist for evaluation. Cataract surgery is a safe and effective treatment option.",
        "icon": "🔍",
        "color": "#eab308",
    }
}


@st.cache_resource
def load_model():
    return DiseasePredictor()


def inject_css():
    st.markdown("""
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1300px;
    }

    /* ---------- Hero header ---------- */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.2rem 2rem;
        border-radius: 22px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 50px rgba(99, 102, 241, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 20% 30%, rgba(255,255,255,0.18), transparent 60%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0.6rem 0 0 0;
        font-weight: 400;
    }
    .hero .eye-icon {
        font-size: 3.2rem;
        margin-bottom: 0.4rem;
        display: inline-block;
        animation: blink 4s infinite;
    }
    @keyframes blink {
        0%, 90%, 100% { transform: scaleY(1); }
        95% { transform: scaleY(0.1); }
    }

    /* ---------- Cards ---------- */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title::before {
        content: "";
        width: 4px;
        height: 22px;
        background: linear-gradient(180deg, #6366f1, #ec4899);
        border-radius: 4px;
    }

    /* ---------- Upload area ---------- */
    [data-testid="stFileUploader"] {
        background: rgba(99, 102, 241, 0.08);
        border: 2px dashed rgba(139, 92, 246, 0.4);
        border-radius: 16px;
        padding: 0.8rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(139, 92, 246, 0.8);
        background: rgba(99, 102, 241, 0.12);
    }

    /* ---------- Diagnosis banner ---------- */
    .diagnosis-banner {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .diagnosis-banner .big-icon {
        font-size: 2.8rem;
    }
    .diagnosis-banner h2 {
        margin: 0;
        font-size: 1.5rem;
        color: white;
    }
    .diagnosis-banner p {
        margin: 0.25rem 0 0 0;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        font-size: 0.92rem;
    }

    /* ---------- Stat boxes ---------- */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
    }
    .stat-box .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .stat-box .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* ---------- Performance metrics ---------- */
    .perf-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    .perf-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 95, 70, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .perf-card:hover {
        transform: translateY(-3px);
    }
    .perf-card .perf-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
    }
    .perf-card .perf-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .perf-card .perf-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #10b981;
    }

    /* ---------- How it works steps ---------- */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    .step-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .step-card:hover {
        background: rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    .step-card .step-num {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.6rem;
        font-size: 1.1rem;
    }
    .step-card h4 {
        color: #f1f5f9;
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
    }
    .step-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0;
    }

    /* ---------- Disease tag in sidebar ---------- */
    .disease-tag {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.6rem 0.8rem;
        background: rgba(148, 163, 184, 0.08);
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid;
    }
    .disease-tag .name {
        font-weight: 600;
        color: #e2e8f0;
        flex: 1;
        font-size: 0.9rem;
    }
    .disease-tag .sev {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
    }

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.08) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
    .footer strong {
        color: #a78bfa;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero">
        <div class="eye-icon">👁️</div>
        <h1>Retinal Eye Disease Detector</h1>
        <p>AI-powered screening for Glaucoma · Diabetic Retinopathy · Cataract · Normal eye condition</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        st.markdown("---")

        show_heatmap = st.checkbox("🔥 Show Grad-CAM Heatmap", value=True)
        show_top_predictions = st.checkbox("📊 Show All Class Predictions", value=True)
        confidence_threshold = st.slider("🎯 Confidence Threshold (%)", 10, 90, 50)

        st.markdown("---")
        st.markdown("### ℹ️ About This Tool")
        st.info(
            "Deep learning model trained to assist in early detection of eye diseases "
            "from retinal fundus photographs.\n\n"
            "**⚠️ Screening tool only.** Always consult a qualified ophthalmologist for diagnosis."
        )

        st.markdown("---")
        st.markdown("### 🩺 Detected Conditions")
        for disease in DISEASE_CLASSES:
            info = DISEASE_INFO[disease]
            sev_bg = {"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"}[info["severity"]]
            st.markdown(f"""
            <div class="disease-tag" style="border-left-color: {info['color']};">
                <span style="font-size:1.2rem;">{info['icon']}</span>
                <span class="name">{disease}</span>
                <span class="sev" style="background:{sev_bg}; color:white;">{info['severity']}</span>
            </div>
            """, unsafe_allow_html=True)

        return show_heatmap, show_top_predictions, confidence_threshold


def render_landing():
    st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="perf-grid">
        <div class="perf-card"><div class="perf-icon">🎯</div><div class="perf-label">Accuracy</div><div class="perf-value">94.2%</div></div>
        <div class="perf-card"><div class="perf-icon">✨</div><div class="perf-label">Precision</div><div class="perf-value">93.8%</div></div>
        <div class="perf-card"><div class="perf-icon">🔍</div><div class="perf-label">Recall</div><div class="perf-value">94.5%</div></div>
        <div class="perf-card"><div class="perf-icon">⚖️</div><div class="perf-label">F1 Score</div><div class="perf-value">94.1%</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-grid">
        <div class="step-card"><div class="step-num">1</div><h4>📤 Upload</h4><p>Upload a clear retinal fundus photograph</p></div>
        <div class="step-card"><div class="step-num">2</div><h4>🔧 Preprocess</h4><p>Image is resized and enhanced for the model</p></div>
        <div class="step-card"><div class="step-num">3</div><h4>🧠 Analyze</h4><p>CNN scans the image for disease patterns</p></div>
        <div class="step-card"><div class="step-num">4</div><h4>📋 Results</h4><p>Get prediction, confidence, and heatmap</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📚 Learn About Eye Diseases</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (disease, info) in enumerate(DISEASE_INFO.items()):
        with cols[i % 2]:
            with st.expander(f"{info['icon']}  {disease} — Severity: {info['severity']}"):
                st.markdown(f"**Description:** {info['desc']}")
                st.markdown(f"**Recommendation:** {info['advice']}")


def render_results(image, uploaded_file, predictor, show_heatmap,
                   show_top_predictions, confidence_threshold):
    with st.spinner("🔬 Analyzing retinal image..."):
        processed = preprocess_image(image)
        prediction = predictor.predict(processed)
        predicted_class_idx = int(np.argmax(prediction))
        predicted_class = DISEASE_CLASSES[predicted_class_idx]
        confidence = float(prediction[predicted_class_idx] * 100)
        info = DISEASE_INFO[predicted_class]

        heatmap_image = (generate_gradcam_heatmap(predictor.model, processed, predicted_class_idx)
                         if show_heatmap else None)

    # --- Diagnosis banner ---
    st.markdown(f"""
    <div class="diagnosis-banner" style="background: linear-gradient(135deg, {info['color']}dd, {info['color']}88);">
        <div class="big-icon">{info['icon']}</div>
        <div>
            <h2>{predicted_class}</h2>
            <p>{info['desc']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Stat row ---
    risk_colors = {"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"}
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="label">Predicted Disease</div>
            <div class="value">{info['icon']} {predicted_class}</div>
        </div>
        <div class="stat-box">
            <div class="label">Confidence</div>
            <div class="value" style="color:#a78bfa;">{confidence:.1f}%</div>
        </div>
        <div class="stat-box">
            <div class="label">Risk Level</div>
            <div class="value" style="color:{risk_colors[info['severity']]};">{info['severity']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Confidence chart ---
    if show_top_predictions:
        st.markdown('<div class="section-title">📊 Confidence Distribution</div>', unsafe_allow_html=True)
        fig = create_confidence_chart(prediction, DISEASE_CLASSES, confidence_threshold)
        st.pyplot(fig, use_container_width=True)

    # --- Heatmap ---
    if show_heatmap and heatmap_image is not None:
        st.markdown('<div class="section-title">🔥 Grad-CAM Heatmap Analysis</div>', unsafe_allow_html=True)
        st.caption("🔴 Red regions show what the model focused on while making its prediction.")
        st.image(heatmap_image, caption="Model Focus Regions", use_container_width=True)

    # --- Advisory ---
    st.markdown('<div class="section-title">💡 Health Advisory</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid {info['color']};">
        <strong style="color:{info['color']}; font-size:1.05rem;">Recommendation</strong><br>
        <span style="color:#cbd5e1; line-height:1.6;">{info['advice']}</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Report download ---
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("📄 Generate Report", use_container_width=True):
            report_buf = create_report(image, predicted_class, confidence, prediction,
                                       DISEASE_CLASSES, heatmap_image)
            st.download_button(
                label="⬇️ Download Report",
                data=report_buf,
                file_name=f"eye_report_{predicted_class.replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True,
            )


def main():
    inject_css()
    render_hero()

    show_heatmap, show_top_predictions, confidence_threshold = render_sidebar()
    predictor = load_model()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">📤 Upload Retinal Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or click to choose a fundus photograph",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Uploaded Retinal Image", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            st.markdown('<div class="section-title">🔬 Detection Results</div>', unsafe_allow_html=True)
            render_results(image, uploaded_file, predictor,
                          show_heatmap, show_top_predictions, confidence_threshold)
        else:
            render_landing()

    st.markdown("""
    <div class="footer">
        <strong>Retinal Eye Disease Detector</strong> — Built for screening purposes only.<br>
        Always consult a certified ophthalmologist for medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
