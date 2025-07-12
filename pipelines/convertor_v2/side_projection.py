import cv2
import numpy as np
import mediapipe as mp

def generate_side_masks(mask_front_path, mask_back_path, output_left_path, output_right_path, image_path=None):
    # Si l'image n'est pas fournie, essaye de deviner le chemin
    if image_path is None:
        image_path = mask_front_path.replace("_mask.png", ".jpeg")
    image = cv2.imread(image_path)
    
    # Fallback si l'image n'est pas trouvée
    if image is None:
        print(f"[WARN] Image introuvable : {image_path}. Fallback sur masque frontal réduit.")
        front = cv2.imread(mask_front_path, 0)
        h, w = front.shape
        scale_factor = 0.35
        side_mask = cv2.resize(front, (int(w * scale_factor), h), interpolation=cv2.INTER_NEAREST)
        x_offset = (w - side_mask.shape[1]) // 2
        left = np.zeros_like(front)
        right = np.zeros_like(front)
        left[:, x_offset:x_offset + side_mask.shape[1]] = side_mask
        right[:, x_offset:x_offset + side_mask.shape[1]] = side_mask
        cv2.imwrite(output_left_path, left)
        cv2.imwrite(output_right_path, right)
        return
    
    h, w = image.shape[:2]
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    mask_side = np.zeros((h, w), dtype=np.uint8)
    if results.pose_landmarks:
        # Tête (ellipse)
        nose = results.pose_landmarks.landmark[0]
        head_x = int(nose.x * w)
        head_y = int(nose.y * h)
        cv2.ellipse(mask_side, (head_x, head_y), (int(w*0.06), int(h*0.09)), 0, 0, 360, 255, -1)

        # Tronc
        left_shoulder = results.pose_landmarks.landmark[11]
        right_hip = results.pose_landmarks.landmark[24]
        trunk_x = int(left_shoulder.x * w)
        trunk_top = int(left_shoulder.y * h)
        trunk_bot = int(right_hip.y * h)
        cv2.rectangle(mask_side, (trunk_x-10, trunk_top), (trunk_x+10, trunk_bot), 255, -1)

        # Bras
        left_elbow = results.pose_landmarks.landmark[13]
        left_wrist = results.pose_landmarks.landmark[15]
        arm_y = int(left_elbow.y * h)
        cv2.rectangle(mask_side, (trunk_x+10, arm_y-5), (trunk_x+65, arm_y+5), 255, -1)

        # Jambe
        left_knee = results.pose_landmarks.landmark[25]
        left_ankle = results.pose_landmarks.landmark[27]
        leg_x = int(left_knee.x * w)
        leg_top = int(left_knee.y * h)
        leg_bot = int(left_ankle.y * h)
        cv2.rectangle(mask_side, (leg_x-5, leg_top), (leg_x+5, leg_bot), 255, -1)
    else:
        print("[WARN] Pose non détectée. Fallback sur masque frontal réduit.")
        front = cv2.imread(mask_front_path, 0)
        scale_factor = 0.35
        side_mask = cv2.resize(front, (int(w * scale_factor), h), interpolation=cv2.INTER_NEAREST)
        x_offset = (w - side_mask.shape[1]) // 2
        mask_side[:, x_offset:x_offset + side_mask.shape[1]] = side_mask

    cv2.imwrite(output_left_path, mask_side)
    cv2.imwrite(output_right_path, mask_side)