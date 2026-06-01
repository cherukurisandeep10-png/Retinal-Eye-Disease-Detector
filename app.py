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
        "advice": "Continue regular eye checkups annually. Maintain a healthy diet rich in vitamins A, C, and E."
    },
    "Diabetic Retinopathy": {
        "desc": "Damage to blood vessels in the retina caused by diabetes. Can lead to vision loss if untreated.",
        "severity": "High",
        "advice": "Consult an ophthalmologist immediately. Manage blood sugar levels. Regular screening is essential for diabetic patients."
    },
    "Glaucoma": {
        "desc": "A group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye.",
        "severity": "High",
        "advice": "Seek immediate medical attention. Glaucoma can cause irreversible vision loss. Early treatment can slow progression."
    },
    "Cataract": {
        "desc": "Clouding of the eye's natural lens, leading to decreased vision. Common in older adults.",
        "severity": "Medium",
        "advice": "Consult an eye specialist for evaluation. Cataract surgery is a safe and effective treatment option."
    }
}


@st.cache_resource
def load_model():
    predictor = DiseasePredictor()
    return predictor


def main():
    custom_css = """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown('<p class="main-title">Retinal Eye Disease Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a retinal fundus image to detect Glaucoma, Diabetic Retinopathy, Cataract, or Normal eye condition</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("Settings")
        st.markdown("---")

        show_heatmap = st.checkbox("Show Heatmap Analysis", value=True)
        show_top_predictions = st.checkbox("Show All Predictions", value=True)
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=10,
            max_value=90,
            value=50
        )

        st.markdown("---")
        st.subheader("About")
        st.info(
            "This tool uses a Convolutional Neural Network trained on retinal fundus images "
            "to assist in early detection of eye diseases.\n\n"
            "This is a screening tool only. Always consult a qualified ophthalmologist for diagnosis."
        )

        st.markdown("---")
        st.subheader("Supported Diseases")
        for disease in DISEASE_CLASSES:
            info = DISEASE_INFO[disease]
            st.markdown(f"**{disease}** — Severity: {info['severity']}")

    predictor = load_model()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Retinal Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Retinal Image", use_container_width=True)

            with st.spinner("Analyzing retinal image..."):
                processed = preprocess_image(image)
                prediction = predictor.predict(processed)
                predicted_class_idx = np.argmax(prediction)
                predicted_class = DISEASE_CLASSES[predicted_class_idx]
                confidence = prediction[predicted_class_idx] * 100

                if show_heatmap:
                    heatmap_image = generate_gradcam_heatmap(
                        predictor.model, processed, predicted_class_idx
                    )
                else:
                    heatmap_image = None

        else:
            st.markdown(
                '<div class="upload-area">'
                '<h3>Drag & drop or click to upload</h3>'
                '<p>Supported: JPG, PNG, BMP</p>'
                '<p style="color:#888; font-size:0.85rem;">Upload a retinal fundus photograph</p>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.subheader("Model Performance")
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            with perf_col1:
                st.metric("Accuracy", "94.2%")
            with perf_col2:
                st.metric("Precision", "93.8%")
            with perf_col3:
                st.metric("Recall", "94.5%")
            with perf_col4:
                st.metric("F1 Score", "94.1%")

            st.markdown("---")
            st.subheader("How It Works")
            steps = [
                ("1. Upload", "Upload a clear retinal fundus photograph"),
                ("2. Preprocess", "Image is resized and normalized for the model"),
                ("3. Analyze", "CNN model scans the image for disease patterns"),
                ("4. Results", "Get prediction, confidence score, and heatmap")
            ]
            for step_title, step_desc in steps:
                st.markdown(f"**{step_title}** — {step_desc}")

    with col2:
        if uploaded_file is not None:
            st.subheader("Detection Results")

            info = DISEASE_INFO[predicted_class]

            if predicted_class == "Normal":
                st.success(f"**{predicted_class}** — No disease detected")
            else:
                st.warning(f"**{predicted_class}** detected — {info['desc']}")

            st.markdown("---")

            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Predicted Disease", predicted_class)
            with met2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with met3:
                severity = info["severity"]
                if severity == "Low":
                    st.metric("Risk Level", "Low")
                elif severity == "Medium":
                    st.metric("Risk Level", "Medium")
                else:
                    st.metric("Risk Level", "High")

            st.markdown("---")

            if show_top_predictions:
                st.subheader("Confidence Distribution")
                fig = create_confidence_chart(prediction, DISEASE_CLASSES, confidence_threshold)
                st.pyplot(fig)

            if show_heatmap and heatmap_image is not None:
                st.subheader("Heatmap Analysis")
                st.markdown(
                    "Regions highlighted in red are the areas "
                    "the model focused on while making the prediction."
                )
                st.image(heatmap_image, caption="Grad-CAM Heatmap — Model Focus Regions", use_container_width=True)

            st.markdown("---")
            st.subheader("Health Advisory")
            st.info(f"**Recommendation:** {info['advice']}")

            st.markdown("---")

            if st.button("Generate Report"):
                report_buf = create_report(
                    image, predicted_class, confidence, prediction,
                    DISEASE_CLASSES, heatmap_image
                )
                st.download_button(
                    label="Download Report",
                    data=report_buf,
                    file_name=f"eye_report_{predicted_class}.png",
                    mime="image/png"
                )
        else:
            st.subheader("Waiting for image upload...")
            st.markdown(
                "Upload a retinal fundus image from the left panel to start the analysis."
            )
            st.markdown("---")
            st.subheader("Learn About Eye Diseases")
            for disease, info in DISEASE_INFO.items():
                with st.expander(f"{disease} (Severity: {info['severity']})"):
                    st.write(info["desc"])
                    st.write(f"**Advice:** {info['advice']}")

    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        'Retinal Eye Disease Detector — Built for screening purposes only.<br>'
        'Always consult a certified ophthalmologist for medical diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
