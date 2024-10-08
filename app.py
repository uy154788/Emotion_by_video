from flask import Flask, request, jsonify
import cv2
import os
from deepface import DeepFace

app = Flask(__name__)

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_faces(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_frames.append(frame[y:y+h, x:x+w])
    return face_frames

def analyze_emotion(face_frames):
    emotions = []
    for face_frame in face_frames:
        try:
            result = DeepFace.analyze(face_frame, actions=['emotion'])
            if isinstance(result, list):
                result = result[0]
            emotions.append(result.get('dominant_emotion', 'unknown'))
        except Exception as e:
            print(f"Error analyzing face: {e}")
            emotions.append('unknown')

    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print(emotion_counts)
    return emotions

def determine_confidence_level(emotions):
    confidence_level = 0
    for emotion in emotions:
        if emotion == 'happy':
            confidence_level += 1
        elif emotion == 'neutral':
            confidence_level += 0.8
        elif emotion == 'surprise':
            confidence_level += 0.6
        elif emotion in ['sad', 'fear']:
            confidence_level += 0.4
        elif emotion in ['angry', 'disgust']:
            confidence_level += 0.1
    return confidence_level / len(emotions) if emotions else 0

@app.route('/analyze', methods=['POST'])
def analyze_video():
    video_path = request.json.get('video_path')
    if not video_path:
        return jsonify({'error': 'Video path not provided'}), 400

    try:
        frames = extract_frames(video_path)
        print(f"Number of frames detected: {len(frames)}")
        face_frames = detect_faces(frames)
        print(f"Number of face frames detected: {len(face_frames)}")
        emotions = analyze_emotion(face_frames)
        confidence_level = determine_confidence_level(emotions)
        confidence_level = round(confidence_level * 100, 2)
        return jsonify({'confidence_level': confidence_level})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port="8001")
