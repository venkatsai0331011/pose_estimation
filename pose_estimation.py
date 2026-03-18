import mediapipe as mp

from mediapipe import solutions
mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from IPython.display import HTML
from base64 import b64encode

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)

def evaluate_posture(landmarks):
    feedback = []

    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    shoulder = [shoulder.x, shoulder.y]
    elbow = [elbow.x, elbow.y]
    wrist = [wrist.x, wrist.y]
    hip = [hip.x, hip.y]
    knee = [knee.x, knee.y]
    ankle = [ankle.x, ankle.y]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    knee_angle = calculate_angle(hip, knee, ankle)
    back_angle = calculate_angle(shoulder, hip, knee)

    if elbow_angle < 160:
        feedback.append("Straighten your arm")

    if knee_angle < 160:
        feedback.append("Do not bend your knee too much")

    if back_angle < 160:
        feedback.append("Keep your back straight")

    return feedback, elbow_angle, knee_angle, back_angle

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    pose = mp_pose.Pose()
    data_log = []

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            feedback, elbow_angle, knee_angle, back_angle = evaluate_posture(landmarks)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            y = 30
            for f in feedback:
                cv2.putText(frame, f, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 30

            data_log.append({
                "frame": frame_id,
                "elbow_angle": elbow_angle,
                "knee_angle": knee_angle,
                "back_angle": back_angle,
                "feedback": "; ".join(feedback)
            })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    df = pd.DataFrame(data_log)
    df.to_csv("posture_report.csv", index=False)

    print("✅ Done processing!")

process_video("input.mp4", "output.mp4")


def show_video(path):
    mp4 = open(path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width=600 controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    """)

show_video("output.mp4")


from google.colab import files
files.download("output.mp4")
files.download("posture_report.csv")
