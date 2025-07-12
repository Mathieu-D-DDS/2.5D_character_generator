import os
import urllib.request

# Liste des modèles à télécharger
MODELS = [
    {
        "name": "MiDaS v3.1 Small",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/model-small.onnx",
        "dest": "models/midas/model-small.onnx"
    },
    {
        "name": "MiDaS v3.1 Large",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/model-f6b98070.onnx",
        "dest": "models/midas/model-f6b98070.onnx"
    },
    # Stable Diffusion / ControlNet : liens de modèles poids publics (à adapter selon vos usages/licences)
    # Exemples :
    # {
    #     "name": "Stable Diffusion v1.5",
    #     "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
    #     "dest": "models/sd/v1-5-pruned-emaonly.ckpt"
    # },
    # {
    #     "name": "ControlNet pose",
    #     "url": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth",
    #     "dest": "models/controlnet/control_sd15_openpose.pth"
    # }
]

def download_model(model):
    os.makedirs(os.path.dirname(model["dest"]), exist_ok=True)
    if os.path.exists(model["dest"]):
        print(f"✓ {model['name']} déjà téléchargé.")
        return
    print(f"Téléchargement de {model['name']} ...")
    try:
        urllib.request.urlretrieve(model["url"], model["dest"])
        print(f"→ {model['name']} téléchargé dans {model['dest']}")
    except Exception as e:
        print(f"❌ Erreur pour {model['name']}: {e}")

if __name__ == "__main__":
    for m in MODELS:
        download_model(m)
    print("\nTéléchargement terminé.")
    print("Vous pouvez compléter ou ajuster la liste des modèles dans download_models.py selon vos besoins.")