import os

# 1. Enforce single-threading BEFORE torch loads to prevent RAM crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import subprocess
import sys
import tempfile
import torch

torch.set_num_threads(1)

st.set_page_config(page_title="Techno Stem Extractor", page_icon="🎛️", layout="centered")

st.title("🎛️ Techno Stem Extractor")
st.write("Extract Drums, Bass, Vocals, and Other stems from a **30-second preview clip** using Meta's Demucs (`htdemucs`).")

uploaded_file = st.file_uploader("Upload a techno track (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Extract Stems (30s Preview)", type="primary"):
        with st.spinner("Trimming track & separating stems... (Takes ~15 seconds)"):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save original uploaded file
                input_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # -------------------------------------------------------------
                # ADDED HERE: Trim uploaded track to a 30-second snippet via FFmpeg
                # -------------------------------------------------------------
                preview_path = os.path.join(temp_dir, "preview.wav")
                trim_cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-t", "30",
                    "-c:a", "pcm_s16le",
                    preview_path
                ]
                subprocess.run(trim_cmd, capture_output=True, check=True)
                
                # Run Demucs on the 30s preview file using fast base model 'htdemucs'
                cmd = [
                    sys.executable, "-m", "demucs.separate",
                    "-n", "htdemucs",
                    "-j", "1",
                    "--device", "cpu",
                    "-o", temp_dir,
                    preview_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    stem_dir = os.path.join(temp_dir, "htdemucs", "preview")
                    st.success("30-Second Preview Extraction Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    drums_path = os.path.join(stem_dir, "drums.wav")
                    bass_path = os.path.join(stem_dir, "bass.wav")
                    vocals_path = os.path.join(stem_dir, "vocals.wav")
                    other_path = os.path.join(stem_dir, "other.wav")
                    
                    if os.path.exists(drums_path):
                        with col1:
                            st.subheader("🥁 Drums (Kick & Perc)")
                            st.audio(drums_path)
                            with open(drums_path, "rb") as f:
                                st.download_button("Download Drums", f, file_name="drums_preview.wav")
                                
                    if os.path.exists(bass_path):
                        with col2:
                            st.subheader("🔊 Bass & Sub")
                            st.audio(bass_path)
                            with open(bass_path, "rb") as f:
                                st.download_button("Download Bass", f, file_name="bass_preview.wav")

                    col3, col4 = st.columns(2)
                    
                    if os.path.exists(vocals_path):
                        with col3:
                            st.subheader("🎤 Vocals")
                            st.audio(vocals_path)
                            with open(vocals_path, "rb") as f:
                                st.download_button("Download Vocals", f, file_name="vocals_preview.wav")

                    if os.path.exists(other_path):
                        with col4:
                            st.subheader("🎹 Synths & Atmosphere")
                            st.audio(other_path)
                            with open(other_path, "rb") as f:
                                st.download_button("Download Other", f, file_name="other_preview.wav")
                else:
                    st.error("Separation failed. Details below:")
                    st.code(result.stderr)
