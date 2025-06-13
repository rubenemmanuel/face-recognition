from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
from pathlib import Path

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# === Load known faces ===
def load_known_faces(dataset_path):
    known_encodings = []
    known_names = []
    for image_path in dataset_path.glob("*.jpeg"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Unable to read {image_path}. Skipping...")
            continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(image_path.stem)
        else:
            print(f"Warning: No face found in {image_path}. Skipping...")
    return known_encodings, known_names

dataset_path = Path("dataset")
known_face_encodings, known_face_names = load_known_faces(dataset_path)

if not known_face_encodings:
    print("No known faces loaded. Exiting...")
    exit()

# === Generator: proses & stream frame ke browser ===
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1

            name = "Unknown"
            if best_match_index != -1 and face_distances[best_match_index] < 0.6:
                name = ''.join([char for char in known_face_names[best_match_index] if not char.isdigit()])

            # Lokasi wajah dalam skala asli (karena tadi resize)
            top, right, bottom, left = [v * 2 for v in face_location]

            # Gambar kotak + nama
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Encode sebagai JPEG dan stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)
