# 🎛️ AI Techno Stem Extractor

An AI-powered web application designed for music producers to isolate audio stems (Drums, Bass, Vocals, Other) from tracks and samples using Meta's `htdemucs` model—without expensive subscription paywalls.

<img width="1912" height="817" alt="Screenshot 2026-09-10 131422" src="https://github.com/user-attachments/assets/298cd085-b7a6-4eb7-af6c-41ec91c740e8" />

<img width="1915" height="810" alt="Screenshot 2026-09-10 131711" src="https://github.com/user-attachments/assets/ab8d0263-6768-4fea-8443-0c9c249c130f" />



## 🔗 Live Demo
Try the interactive web app: **[techno-stem-extractor.streamlit.app](https://techno-stem-extractor.streamlit.app)**

---

## 🛠️ Architecture & Optimization

* **AI Engine:** Meta's `htdemucs` hybrid transformer model running on PyTorch for high-fidelity 4-channel source separation.
* **Pre-Processing:** Automated FFmpeg trimming pipeline (`-t 30`) that clips uploaded audio into 30-second previews, dropping inference times from minutes down to ~15 seconds.
* **Memory Management:** Resolved Streamlit Cloud's 1GB RAM crash limit caused by PyTorch multi-threading. Enforced strict single-threading (`OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`), capping memory usage under 350MB.
* **Frontend:** Streamlit web UI featuring file drag-and-drop staging, interactive audio players, and instant stem export downloads.

---

