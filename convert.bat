@echo off
REM Initialisation de l'environnement virtuel si nécessaire
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate

REM Installation ou mise à jour des dépendances Python
pip install --upgrade pip
pip install -r requirements.txt

REM Téléchargement ou mise à jour des modèles nécessaires
python download_models.py

REM Lancement du script principal avec les arguments fournis
python convert.py %*