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
