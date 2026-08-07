"""
Frontend Streamlit Application for Brain Tumor Detection System.
Provides a professional, responsive dashboard for medical image analysis.
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

# Add parent directory to path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.inference import BrainTumorDetector
from src.config import BEST_MODEL_PATH

# Configure the Streamlit Page
st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit Default Formatting for a Premium Look
HIDE_ST_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
"""
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

@st.cache_resource
def load_inference_engine():
    """
    Initializes the backend AI engine. 
    Cached to prevent reloading the model into memory on every UI interaction.
    """
    return BrainTumorDetector()

def generate_json_report(results: dict) -> str:
    """Formats prediction results into a downloadable JSON string."""
    report_data = {k: v for k, v in results.items() if k not in ["processed_image", "isolated_mask"]}
    return json.dumps(report_data, indent=4)

def generate_csv_report(results: dict) -> str:
    """Formats prediction results into a downloadable CSV string."""
    report_data = {k: [v] for k, v in results.items() if k not in ["processed_image", "isolated_mask"]}
    df = pd.DataFrame(report_data)
    return df.to_csv(index=False)

def generate_txt_report(results: dict) -> str:
    """Formats prediction results into a downloadable TXT string."""
    txt_content = "Brain Tumor Detection Diagnostic Report\n"
    txt_content += "=" * 40 + "\n"
    txt_content += f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    txt_content += "-" * 40 + "\n"
    for key, value in results.items():
        if key not in ["processed_image", "isolated_mask"]:
            formatted_key = key.replace("_", " ").title()
            txt_content += f"{formatted_key}: {value}\n"
    txt_content += "=" * 40 + "\n"
    return txt_content

def main():
    """Main application execution logic."""
    
    # Initialize Backend
    try:
        engine = load_inference_engine()
    except Exception as e:
        st.error(f"Failed to load AI Engine: {str(e)}")
        st.stop()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    menu_selection = st.sidebar.radio(
        "Menu",
        ["Dashboard", "System Information", "Help", "About"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Version:** YOLOv8 Nano Seg")
    st.sidebar.markdown(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}")

    if menu_selection == "Dashboard":
        st.title("Brain Tumor Detection Dashboard")
        st.markdown("Upload a Brain MRI image to perform instance segmentation and detect anomalies.")

        # Drag and Drop File Uploader
        uploaded_file = st.file_uploader(
            "Choose a Medical Image (JPG, JPEG, PNG)", 
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display Loading UI
            with st.spinner("Processing MRI scan through Medical AI Pipeline..."):
                try:
                    # Save uploaded file temporarily for cv2 processing
                    temp_dir = Path("data/processed/temp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / uploaded_file.name
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Execute Inference
                    original_image = Image.open(uploaded_file)
                    results = engine.process_image(str(temp_path))

                    # Clean up temporary file
                    if temp_path.exists():
                        os.remove(temp_path)

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    return

            st.success("Analysis Complete.")

            # Diagnostic Status Banner
            if results["tumor_detected"]:
                formatted_class = results['tumor_class'].replace('_', ' ').title()
                st.error(f"DIAGNOSTIC RESULT: Tumor Detected ({formatted_class})")
            else:
                st.success("DIAGNOSTIC RESULT: No Brain Tumor Detected")

            # Image Visualization Tabs
            st.markdown("### Medical Image Visualization")
            tab1, tab2, tab3 = st.tabs(["Side-by-Side Comparison", "Isolated Tumor Mask", "Raw MRI View"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Pre-Analysis Raw MRI Scan")
                    st.image(original_image, use_container_width=True)

                with col2:
                    st.caption("Post-Analysis Segmented MRI (Medical Outline)")
                    st.image(results["processed_image"], use_container_width=True)

            with tab2:
                col_mask1, col_mask2 = st.columns([1, 2])
                with col_mask1:
                    st.markdown("#### Lesion Contour Extraction")
                    st.markdown("""
                    This view isolates the detected abnormal tissue contour from the rest of the brain parenchyma.
                    """)
                    if results["tumor_detected"]:
                        st.info(f"Tumor Area: {results['tumor_area']}")
                        st.info(f"Confidence: {results['confidence_score']}")
                    else:
                        st.write("No lesion contour extracted.")
                with col_mask2:
                    st.image(results["isolated_mask"], use_container_width=True)

            with tab3:
                st.image(original_image, caption=f"Original File: {uploaded_file.name}", use_container_width=True)

            st.markdown("---")

            # Prediction Statistics Section
            st.markdown("### Diagnostic Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Confidence Score", results["confidence_score"])
            m_col2.metric("Probability", results["probability"])
            m_col3.metric("Tumor Lesion Area", results["tumor_area"])
            m_col4.metric("Inference Speed", results["inference_speed"])

            st.markdown("---")

            # Detailed Output Table
            st.markdown("### Comprehensive Technical Audit")
            report_dict = {
                "Audit Parameter": [
                    "Tumor Classification",
                    "Image Resolution",
                    "Prediction Timestamp",
                    "Memory Footprint",
                    "Model Framework",
                    "File Name"
                ],
                "Value": [
                    results["tumor_class"].replace("_", " ").title(),
                    results["image_resolution"],
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    results["memory_usage"],
                    "YOLOv8-Seg (CPU Optimized)",
                    uploaded_file.name
                ]
            }
            detailed_df = pd.DataFrame(report_dict)
            st.table(detailed_df)

            st.markdown("---")

            # Download Features Section
            st.markdown("### Export Reports")
            d_col1, d_col2, d_col3 = st.columns(3)

            json_report = generate_json_report(results)
            csv_report = generate_csv_report(results)
            txt_report = generate_txt_report(results)

            d_col1.download_button(
                label="Download JSON Report",
                data=json_report,
                file_name=f"report_{uploaded_file.name}.json",
                mime="application/json"
            )

            d_col2.download_button(
                label="Download CSV Report",
                data=csv_report,
                file_name=f"report_{uploaded_file.name}.csv",
                mime="text/csv"
            )

            d_col3.download_button(
                label="Download TXT Report",
                data=txt_report,
                file_name=f"report_{uploaded_file.name}.txt",
                mime="text/plain"
            )

    elif menu_selection == "System Information":
        st.title("System Information")
        st.markdown("Live statistics of the execution environment.")
        
        sys_data = {
            "Component": ["Operating System", "Processor Architecture", "Python Version", "PyTorch Version", "Execution Device"],
            "Details": [
                os.name.upper(),
                "64-bit",
                sys.version.split(" ")[0],
                "2.13.0+cpu",
                "CPU (Optimized)"
            ]
        }
        st.table(pd.DataFrame(sys_data))

    elif menu_selection == "Help":
        st.title("User Guide")
        st.markdown("""
        **How to use the Brain Tumor Detection System:**
        1. Navigate to the **Dashboard** using the sidebar.
        2. Drag and drop a Brain MRI image into the upload box.
        3. Wait for the AI to process the image.
        4. Review the Side-by-Side Comparison or Isolated Tumor Mask tabs.
        5. Analyze the extracted metrics and confidence scores.
        6. Use the download buttons to export the diagnostic data.
        
        **Supported Formats:** JPG, JPEG, PNG.
        """)

    elif menu_selection == "About":
        st.title("About")
        st.markdown("""
        **Brain Tumor Detection System**
        
        This is an industry-grade medical imaging application. 
        It utilizes a YOLOv8 Instance Segmentation architecture strictly optimized for CPU computation.
        
        Unlike standard bounding boxes, this system traces the exact polygon contours of detected anomalies, adhering to professional medical software standards.
        """)

if __name__ == "__main__":
    main()