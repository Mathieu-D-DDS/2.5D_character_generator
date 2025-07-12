import cv2
import numpy as np

def generate_side_masks(mask_front_path, mask_back_path, output_left_path, output_right_path):
    front = cv2.imread(mask_front_path, 0)
    back = cv2.imread(mask_back_path, 0)
    left = ((front.astype(float) * 0.7 + back.astype(float) * 0.3)).astype('uint8')
    right = ((front.astype(float) * 0.3 + back.astype(float) * 0.7)).astype('uint8')
    cv2.imwrite(output_left_path, left)
    cv2.imwrite(output_right_path, right)