@echo off
title AI Studio Launcher
echo Avvio della Workstation IA in corso...
echo Assicurati che Stable Diffusion sia gia' aperto!
echo.

:: Si sposta nella cartella dello script (ovunque essa sia)
cd /d "%~dp0"

:: Attiva l'ambiente virtuale
call .venv\Scripts\activate

:: Lancia il programma
python advanced_prompt_gen.py

:: Se il programma si chiude per errore, lascia la finestra aperta per leggere
pause