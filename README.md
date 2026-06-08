# 🌿 Smart Attendance Web Dashboard (Edge AI)

A real-time, automated classroom attendance system featuring localized facial recognition and a clean web-based control panel. This project replaces manual roll calls by combining local machine learning pipelines with an interactive presentation dashboard.

---

## 🚀 Key Features

- **Streamlit Web Interface:** An intuitive layout providing complete control over system states (AI training, camera tracking toggles, and dynamic data refreshes).
- **Edge AI Face Recognition:** Utilizes OpenCV's Haar Cascade Classifiers for face tracking alongside local **LBPH (Local Binary Patterns Histograms)** models for accurate, localized feature matching.
- **Dynamic Database Management:** Uses Pandas to instantly log verified students to a structured `.csv` spreadsheet with explicit timestamps.
- **Integrity Check System:** Built-in validation structures that actively prevent duplicate logging entries per user session.

---

## 📐 System Architecture

The following block diagram outlines the data flow of the Smart Attendance System, demonstrating how visual data transitions from raw webcam frames into secure spreadsheet logs:
[Local Image Directory] ──> [LBPH Face Model Trainer] ──> [Trained Weight Keys (trainer.yml)]
│
▼
[Live Webcam Feed]    ──> [Haar Cascade Detector]    ──> [Face Feature Comparison Engine]
│
▼
[CSV Log Database]     <── [Integrity Filter Check]   <── [Identity Validation Stream]
---

## 📦 Prerequisites & Installation

To run this project locally, ensure you have Python installed on your Windows machine, then set up the required dependencies.

1. **Clone the repository files or download the scripts locally.**
2. **Install the required system packages via the command prompt:**

```cmd
pip install opencv-python streamlit pandas numpy
