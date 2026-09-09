import os
import torch
import librosa
import soundfile as sf
import streamlit as st
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Enable automatic GPU acceleration if available
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="TechnoStem Extractor", layout="centered")
st.title("🎛️ TechnoStem Extractor")
st.markdown("Upload any track to isolate Vocals, Drums, Bass, and Other instruments.")

# -- Cache AI Model --
@st.cache_resource
def load_ai_model():
    model = get_model('htdemucs_ft')
    model.to(device)
    model.eval()
    return model

# -- Drag and Drop Interface --
uploaded_file = st.file_uploader("Drop an audio file here (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Extract Stems 🚀", use_container_width=True):
        with st.spinner(f"AI processing on {device.upper()} hardware..."):
            
            # 1. Save uploaded file temporarily
            file_ext = uploaded_file.name.split(".")[-1]
            temp_input = f"temp_upload.{file_ext}"
            
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. Load audio tensor
            data, sample_rate = librosa.load(temp_input, sr=44100, mono=False)
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.unsqueeze(0).to(device)

            # 3. AI Separation
            model = load_ai_model()
            with torch.no_grad():
                sources = apply_model(model, waveform, shifts=2, split=True, overlap=0.25)

            # 4. Display Results
            st.success("Extraction Complete!")
            
            col1, col2 = st.columns(2)
            columns = [col1, col2, col1, col2]
            
            for i, stem_name in enumerate(model.sources):
                stem_tensor = sources[0, i]
                stem_numpy = stem_tensor.cpu().numpy().T
                
                out_path = f"{stem_name}.wav"
                sf.write(out_path, stem_numpy, sample_rate, subtype='PCM_24')
                
                with columns[i]:
                    st.markdown(f"**{stem_name.capitalize()}**")
                    st.audio(out_path)