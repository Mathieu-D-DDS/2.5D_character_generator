import os
import urllib.request
import zipfile
import io
import shutil

MIDAS_DIR = "midas"
MIDAS_CODE_URL = "https://github.com/isl-org/MiDaS/archive/refs/tags/v3_1.zip"
MIDAS_CODE_FOLDER_IN_ZIP = "MiDaS-3_1/midas/"

MODELS = [
    {
        "name": "MiDaS v3.1 Swin2 Tiny",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
        "dest": os.path.join(MIDAS_DIR, "dpt_swin2_tiny_256.pt")
    },
    {
        "name": "MiDaS v3.1 Swin2 Base",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
        "dest": os.path.join(MIDAS_DIR, "dpt_swin2_base_384.pt")
    }
]

def download_midas_code():
    if os.path.exists(MIDAS_DIR):
        print(f"Suppression du dossier existant {MIDAS_DIR} ...")
        shutil.rmtree(MIDAS_DIR)
    print("Téléchargement du code source MiDaS ...")
    try:
        response = urllib.request.urlopen(MIDAS_CODE_URL)
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in zip_file.namelist():
            if member.startswith(MIDAS_CODE_FOLDER_IN_ZIP):
                rel_path = member[len(MIDAS_CODE_FOLDER_IN_ZIP):]
                if not rel_path:
                    continue
                dest_path = os.path.join(MIDAS_DIR, rel_path.replace("/", os.sep).replace("\\", os.sep))
                if member.endswith("/"):
                    os.makedirs(dest_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with zip_file.open(member) as source, open(dest_path, "wb") as target:
                        target.write(source.read())
        # Ajoute __init__.py si absent
        init_path = os.path.join(MIDAS_DIR, "__init__.py")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# Rend le dossier 'midas' utilisable comme package Python\n")
        print(f"→ Code MiDaS extrait dans {MIDAS_DIR}")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du code MiDaS: {e}")

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
    download_midas_code()
    for m in MODELS:
        download_model(m)
    print("\nTéléchargement terminé.\nTout est dans le dossier 'midas'.")