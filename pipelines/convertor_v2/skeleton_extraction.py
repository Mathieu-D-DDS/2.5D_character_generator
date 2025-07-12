import cv2
import mediapipe as mp
import json
import os

def extract_pose_landmarks(image_path, output_json):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image non trouvée : {image_path}")

    h, w = image.shape[:2]
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("Aucun squelette détecté.")
        return

    keypoints = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        keypoints[f"landmark_{idx}"] = {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
            "visibility": float(landmark.visibility)
        }

    data = {
        "image_path": image_path,
        "image_width": w,
        "image_height": h,
        "keypoints": keypoints
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Squelette extrait et sauvegardé dans : {output_json}")