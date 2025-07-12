import os
import urllib.request

# Liste des modèles à télécharger
MODELS = [
    {
        "name": "MiDaS v3.1 Small",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
        "dest": "model_weights/midas/dpt_swin2_tiny_256.pt"
    },
    {
        "name": "MiDaS v3.1 Large",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt",
        "dest": "model_weights/midas/dpt_large_384.pt"
    },
    # Ajoute ici d'autres modèles si nécessaire
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