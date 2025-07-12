import cv2
import mediapipe as mp

def extract_silhouette(image_path, output_mask_path):
    image = cv2.imread(image_path)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        results = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = (results.segmentation_mask > 0.5).astype('uint8') * 255
        cv2.imwrite(output_mask_path, mask)