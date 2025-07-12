@echo off
REM Initialisation de l'environnement virtuel si nécessaire
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate

REM Mise à jour de pip
python -m pip install --upgrade pip

REM Force l'installation CUDA de torch/torchvision/torchaudio
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Installation ou mise à jour des dépendances Python
pip install -r requirements.txt

REM Téléchargement ou mise à jour des modèles nécessaires
python download_models.py

REM Lancement du script principal avec les arguments fournis
python convert_v2.py %*
pause