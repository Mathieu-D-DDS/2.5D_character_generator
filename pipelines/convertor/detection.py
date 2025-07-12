import cv2
import numpy as np
import mediapipe as mp

def run(image_path):
    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image non trouvée : {image_path}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if results.pose_landmarks:
        # Génération d'un masque approximatif du corps (contour des landmarks)
        points = []
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append((x, y))
        # On prend un hull autour des points pour le corps
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Si pas de détection, on retourne un masque vide
        print("Aucun corps détecté.")
        return image

    # Application du masque sur l'image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented