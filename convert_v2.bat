@echo off
REM Initialisation de l'environnement virtuel si nécessaire
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate

REM Mise à jour de pip
python -m pip install --upgrade pip

REM Installation CUDA de torch/torchvision/torchaudio uniquement si absents
python -c "import torch, torchvision, torchaudio" 2>NUL
if errorlevel 1 (
    echo [INSTALL] torch/torchvision/torchaudio CUDA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [OK] torch/torchvision/torchaudio déjà installés
)

REM Installation ou mise à jour des dépendances Python
pip install -r requirements.txt

REM Téléchargement ou mise à jour des modèles MiDaS seulement si dossier absent
if not exist midas (
    python download_models.py
) else (
    echo [OK] Dossier 'midas' déjà présent
)

REM Téléchargement du modèle SAM (Segment Anything) si non présent dans SAM/
if not exist SAM\sam_vit_h_4b8939.pth (
    echo [INFO] Téléchargement du checkpoint SAM...
    python -c "import urllib.request, os; os.makedirs('SAM', exist_ok=True); url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'; dest='SAM/sam_vit_h_4b8939.pth'; print('Téléchargement...'); urllib.request.urlretrieve(url, dest); print('→ Modèle SAM téléchargé dans SAM/sam_vit_h_4b8939.pth')"
) else (
    echo [OK] Modèle SAM déjà présent
)

REM Lancement du script principal avec les arguments fournis
python convert_v2.py %*
pause