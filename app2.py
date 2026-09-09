import os

# Enforce single-threading BEFORE torch loads to control RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import subprocess
import sys
import tempfile
import glob
import gc
import torch

torch.set_num_threads(1)

st.set_page_config(page_title="Techno Stem Extractor", page_icon="🎛️", layout="centered")

st.title("🎛️ Techno Stem Extractor")
st.write("Extract Drums, Bass, Vocals, and Other stems using Meta's Demucs (`htdemucs_ft`).")

uploaded_file = st.file_uploader("Upload a techno track (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Extract Stems", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Save uploaded file to disk
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Split track into 60s WAV chunks via FFmpeg
            status_text.text("Splitting audio into 60s chunks for low-memory processing...")
            chunks_dir = os.path.join(temp_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            split_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-f", "segment",
                "-segment_time", "60",
                "-c:a", "pcm_s16le",
                os.path.join(chunks_dir, "chunk_%03d.wav")
            ]
            subprocess.run(split_cmd, capture_output=True, check=True)
            
            chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.wav")))
            total_chunks = len(chunk_files)
            
            if total_chunks == 0:
                st.error("Failed to split audio into chunks.")
                st.stop()

            # 3. Process each chunk sequentially
            stems_out_dir = os.path.join(temp_dir, "stems_chunks")
            os.makedirs(stems_out_dir, exist_ok=True)
            
            for idx, chunk_file in enumerate(chunk_files):
                status_text.text(f"Processing chunk {idx + 1} of {total_chunks} (~60s each)...")
                progress_bar.progress((idx) / total_chunks)
                
                chunk_out = os.path.join(stems_out_dir, f"chunk_{idx:03d}")
                
                cmd = [
                    sys.executable, "-m", "demucs.separate",
                    "-n", "htdemucs_ft",
                    "--segment", "7",
                    "-j", "1",
                    "--device", "cpu",
                    "-o", chunk_out,
                    chunk_file
                ]
                
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    st.error(f"Error on chunk {idx + 1}: {res.stderr}")
                    st.stop()
                
                gc.collect()

            # 4. Merge output chunks back into full-length stems
            status_text.text("Merging separated audio chunks into final stems...")
            stems = ["drums", "bass", "vocals", "other"]
            final_stems = {}
            song_name = os.path.splitext(uploaded_file.name)[0]
            
            for stem in stems:
                concat_list_path = os.path.join(temp_dir, f"{stem}_list.txt")
                with open(concat_list_path, "w") as f_concat:
                    for idx in range(total_chunks):
                        chunk_name = f"chunk_{idx:03d}"
                        stem_wav = os.path.join(stems_out_dir, chunk_name, "htdemucs_ft", chunk_name, f"{stem}.wav")
                        f_concat.write(f"file '{stem_wav}'\n")
                
                out_stem_path = os.path.join(temp_dir, f"{song_name}_{stem}.wav")
                concat_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    out_stem_path
                ]
                subprocess.run(concat_cmd, capture_output=True, check=True)
                final_stems[stem] = out_stem_path

            progress_bar.progress(1.0)
            status_text.text("")
            st.success("Extraction Complete!")

            # 5. Render audio players and download buttons
            col1, col2 = st.columns(2)
            if os.path.exists(final_stems["drums"]):
                with col1:
                    st.subheader("🥁 Drums (Kick & Perc)")
                    st.audio(final_stems["drums"])
                    with open(final_stems["drums"], "rb") as f:
                        st.download_button("Download Drums", f, file_name=f"{song_name}_drums.wav")

            if os.path.exists(final_stems["bass"]):
                with col2:
                    st.subheader("🔊 Bass & Sub")
                    st.audio(final_stems["bass"])
                    with open(final_stems["bass"], "rb") as f:
                        st.download_button("Download Bass", f, file_name=f"{song_name}_bass.wav")

            col3, col4 = st.columns(2)
            if os.path.exists(final_stems["vocals"]):
                with col3:
                    st.subheader("🎤 Vocals")
                    st.audio(final_stems["vocals"])
                    with open(final_stems["vocals"], "rb") as f:
                        st.download_button("Download Vocals", f, file_name=f"{song_name}_vocals.wav")

            if os.path.exists(final_stems["other"]):
                with col4:
                    st.subheader("🎹 Synths & Atmosphere")
                    st.audio(final_stems["other"])
                    with open(final_stems["other"], "rb") as f:
                        st.download_button("Download Other", f, file_name=f"{song_name}_other.wav")
