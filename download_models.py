import os
import urllib.request
import zipfile
import io
import shutil

MODELS = [
    {
        "name": "MiDaS v3.1 Small",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
        "dest": os.path.join("model_weights", "midas", "dpt_swin2_tiny_256.pt")
    },
    {
        "name": "MiDaS v3.1 Base",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
        "dest": os.path.join("model_weights", "midas", "dpt_swin2_base_384.pt")
    },
]

MIDAS_CODE_URL = "https://github.com/isl-org/MiDaS/archive/refs/heads/master.zip"
MIDAS_CODE_DEST = os.path.join("external", "midas")

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

def download_midas_code():
    # Nettoie le dossier cible
    if os.path.exists(MIDAS_CODE_DEST):
        print(f"Suppression du dossier existant {MIDAS_CODE_DEST} ...")
        try:
            shutil.rmtree(MIDAS_CODE_DEST)
        except Exception as e:
            print(f"Erreur lors du nettoyage : {e}")
            return
    print("Téléchargement du code source MiDaS ...")
    try:
        response = urllib.request.urlopen(MIDAS_CODE_URL)
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in zip_file.namelist():
            if member.startswith("MiDaS-master/midas/"):
                # Chemin relatif à "midas/"
                rel_path = member[len("MiDaS-master/midas/"):]
                if not rel_path:
                    continue  # Ignore le dossier racine
                dest_path = os.path.join(MIDAS_CODE_DEST, rel_path.replace("/", os.sep).replace("\\", os.sep))
                # Vérifie si c'est un fichier (dans le zip, les dossiers se terminent par /)
                if member.endswith("/"):
                    # C'est un dossier, on le crée si besoin
                    os.makedirs(dest_path, exist_ok=True)
                else:
                    # C'est un fichier
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with zip_file.open(member) as source, open(dest_path, "wb") as target:
                        target.write(source.read())
                print(f"→ Code MiDaS extrait dans {MIDAS_CODE_DEST}")
        # Ajout du fichier __init__.py
        init_path = os.path.join(MIDAS_CODE_DEST, "__init__.py")
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("# Rend le dossier 'midas' utilisable comme package Python\n")
        print(f"✓ Fichier __init__.py ajouté dans {MIDAS_CODE_DEST}")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du code MiDaS: {e}")

if __name__ == "__main__":
    for m in MODELS:
        download_model(m)
    download_midas_code()
    print("\nTéléchargement terminé.\nVous pouvez compléter ou ajuster la liste des modèles dans download_models.py selon vos besoins.")