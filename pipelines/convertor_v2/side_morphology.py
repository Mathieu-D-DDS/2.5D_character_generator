import cv2
import numpy as np

def generate_side_mask_from_front_and_depth(mask_front_path, depth_map_path, output_path, min_thickness=4, scale=0.5):
    """
    Génère une silhouette latérale en projetant la morphologie du mask de face,
    et en utilisant la profondeur MiDaS pour estimer l'épaisseur à chaque ligne.
    Le résultat est centré, plein, sans détails internes.
    """
    # Chargement des fichiers
    mask = cv2.imread(mask_front_path, 0)
    depth = cv2.imread(depth_map_path, 0)
    h, w = mask.shape

    # Redimensionnement de la depth map si besoin
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalise la depth [0,1]
    depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    side = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        row = mask[y]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            # Largeur morphologique = écart entre les extrêmes du mask
            thickness_mask = indices[-1] - indices[0]
            # Épaisseur "profondeur" (moyenne sur la ligne, pondéré par le mask)
            dvals = depth_norm[y, indices]
            thickness_depth = int(np.mean(dvals) * w * scale)
            thickness = max(thickness_mask * scale, thickness_depth, min_thickness)
            thickness = int(thickness)

            # Position centrée verticalement
            x_center = w // 2
            x_left = max(0, x_center - thickness // 2)
            x_right = min(w - 1, x_center + thickness // 2)
            side[y, x_left:x_right] = 255

    # Lissage optionnel
    side = cv2.medianBlur(side, 5)

    # Sauvegarde
    cv2.imwrite(output_path, side)