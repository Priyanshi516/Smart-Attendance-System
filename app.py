import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd

# Set page layout to wide and nature-themed accent colors via configuration
st.set_page_config(page_title="Smart Attendance Dashboard", page_icon="🌿", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Smart Attendance Web Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Smart Attendance System Using Face Detection</p>", unsafe_allow_html=True)
st.write("---")

# Initialize Local Face Detection Systems
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Layout layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ System Control Panel")
    
    # 1. AI Training Engine Interface
    if st.button("🔄 Train AI Recognition Model", use_container_width=True):
        path = 'student_images'
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            st.error("⚠️ No images found in 'student_images' folder!")
        else:
            with st.spinner("Training model on student database..."):
                myList = os.listdir(path)
                faces, ids, names_mapping = [], [], {}
                
                for idx, cl in enumerate(myList):
                    name = os.path.splitext(cl)[0].split('_')[0].upper()
                    names_mapping[idx] = name
                    img = cv2.imread(f'{path}/{cl}')
                    if img is None: continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
                    for (x, y, w, h) in detected_faces:
                        face_crop = gray[y:y+h, x:x+w]
                        if face_crop.size > 0:
                            faces.append(face_crop)
                            ids.append(idx)
                            
                if len(faces) > 0:
                    recognizer.train(faces, np.array(ids))
                    recognizer.save("trainer.yml")
                    with open("labels.txt", "w") as f:
                        for idx, name in names_mapping.items(): f.write(f"{idx},{name}\n")
                    st.success("✅ AI Training Complete! Model keys generated.")
                else:
                    st.warning("⚠️ No clear faces could be extracted from your folder images.")

    st.write("")
    
    # 2. Live Webcam Execution Interface
    st.markdown("### 📹 Live Tracking Feed")
    run_camera = st.checkbox("🟢 Activate Webcam Tracking System")
    
    # Attendance logging logic
    def markAttendance(name):
        with open('attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]
            if name not in nameList:
                now = datetime.now()
                f.writelines(f'\n{name},{now.strftime("%d-%B-%Y")},{now.strftime("%H:%M:%S")}')

    # Load names database safely
    names_mapping = {}
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            for line in f:
                idx, name = line.strip().split(",")
                names_mapping[int(idx)] = name

    # Live feed frame streaming loop
    FRAME_WINDOW = st.image([])
    if run_camera:
        if not os.path.exists("trainer.yml"):
            st.error("❌ Please run the training script first before starting the tracker feed!")
        else:
            recognizer.read("trainer.yml")
            cap = cv2.VideoCapture(0)
            
            while run_camera:
                success, img = cap.read()
                if not success: break
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    name = "UNKNOWN"
                    if face_roi.size > 0:
                        label_id, confidence = recognizer.predict(face_roi)
                        if confidence < 80:
                            name = names_mapping.get(label_id, "UNKNOWN")
                            markAttendance(name)
                    
                    box_color = (0, 255, 0) if name != "UNKNOWN" else (255, 0, 0)
                    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), box_color, 3)
                    cv2.putText(img_rgb, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                FRAME_WINDOW.image(img_rgb)
            cap.release()
    else:
        st.info("Webcam is currently offline. Toggle the checkbox above to activate live detection.")

with col2:
    st.subheader("📊 Real-time Attendance Database")
    
    # 3. Dynamic CSV Database View
    if st.button("🔄 Refresh Data Table", use_container_width=True) or True:
        if os.path.exists('attendance.csv'):
            try:
                df = pd.read_csv('attendance.csv')
                # Styling the dataframe output
                st.dataframe(df.style.set_properties(**{'background-color': '#F4FBF7', 'color': '#2E7D32'}), use_container_width=True)
                
                # Download Button for the Examiner
                with open("attendance.csv", "rb") as file:
                    st.download_button(
                        label="📥 Export Attendance Sheet to Excel (.CSV)",
                        data=file,
                        file_name=f"Attendance_Report_{datetime.now().strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception:
                st.info("Log database is compiling...")
        else:
            st.info("No logs generated yet. Complete a face scan to initialize records.")