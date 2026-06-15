import streamlit as st
import os
from src.pipeline import run_sync_india_pipeline

st.set_page_config(page_title="SyncIndia Dashboard", page_icon="🎬", layout="centered")

st.title("🎬 SynkIndia: your go to video Translator to any indian language")
st.markdown("Automated Deep Learning English-to-Indian-Language Video Translation & Viseme Alignment System.")
st.write("---")

uploaded_file = st.file_uploader("📂 Drop an English source MP4 video file below:", type=["mp4"])
target_lang = st.selectbox("🌐 Select Target Regional Indian Language:", [
    ("hi", "Hindi"), ("te", "Telugu"), ("ta", "Tamil"), ("bn", "Bengali"), ("mr", "Marathi")
], format_func=lambda x: x[1])

epochs = st.slider("🏋️ Localized Fine-Tuning Optimization Loops (Epochs):", min_value=1, max_value=10, value=3)

if st.button("🚀 Execute Neural Lip-Sync Translation Pipeline", use_container_width=True):
    if uploaded_file is not None:
        input_path = os.path.join("data", "raw", "input_source.mp4")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())
            
        status_box = st.info("Initializing system routines...")
        
        def update_status(text):
            status_box.info(text)
            
        final_result = run_sync_india_pipeline(input_path, target_lang=target_lang[0], epochs=epochs, status_callback=update_status)
        
        status_box.success("🎉 Cross-Lingual Viseme Alignment Generation Complete!")
        st.video(final_result)
    else:
        st.error("Please insert a valid source MP4 video clip to begin processing.")