# 🎨 Ultimate AI Studio (Local Prompt & Image Generator)

A powerful local workstation for Stable Diffusion prompt engineering. 
It creates professional prompts from images (Image-to-Text) or raw ideas, and sends them directly to your local Stable Diffusion instance via API.

**Features:**
*   **Image Analysis:** Extracts prompts from existing images using BLIP.
*   **Prompt Enhancement:** Uses GPT-2 to expand simple ideas into artistic descriptions.
*   **Direct Generation:** Connects to Automatic1111/Forge to generate images directly within the UI.
*   **Batch & Gallery:** Generate multiple images at once and view them in a gallery.
*   **Style Presets:** One-click styles (Cyberpunk, Anime, Photorealistic, etc.).
*   **100% Local & Private.**

---

## 🇮🇹 Guida all'Installazione (Italiano)

### Prerequisiti
1.  **Python 3.10** o superiore installato.
2.  **Git** installato.
3.  Una scheda video **NVIDIA** (Consigliato per la velocità).
4.  **Stable Diffusion WebUI (Automatic1111)** installato (opzionale, serve solo per generare le immagini).

### Passo 1: Clona il Repository
Scarica il progetto in una cartella locale.

### Passo 2: Configura l'Ambiente
Apri il terminale nella cartella del progetto ed esegui:

```bash
# 1. Crea l'ambiente virtuale
python -m venv .venv

# 2. Attiva l'ambiente
# Su Windows:
.\.venv\Scripts\activate
# Su Mac/Linux:
# source .venv/bin/activate
Passo 3: Installa le Librerie (Importante!)
Dobbiamo installare PyTorch con supporto GPU e le altre dipendenze. Esegui questi comandi in ordine:
code
Bash
# 1. Installa PyTorch versione GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Installa le altre librerie necessarie
pip install -r requirements.txt
Passo 4: Configura Stable Diffusion (Opzionale)
Se vuoi generare le immagini direttamente dall'interfaccia:
Vai nella cartella del tuo Stable Diffusion (Automatic1111).
Modifica il file webui-user.bat.
Aggiungi --api agli argomenti. Esempio:
set COMMANDLINE_ARGS=--xformers --api
Avvia Stable Diffusion e lascialo aperto.
Passo 5: Avvia AI Studio
Puoi avviare il programma in due modi:
Doppio click sul file avvia_studio.bat (Windows).
Oppure da terminale: python advanced_prompt_gen.py
L'interfaccia si aprirà nel browser all'indirizzo http://127.0.0.1:7863.
🇬🇧 Installation Guide (English)
Prerequisites
Python 3.10 or higher.
Git.
NVIDIA GPU (Recommended for performance).
Stable Diffusion WebUI (Automatic1111) installed (Optional, required only for image generation).
Step 1: Clone the Repository
Download the project to a local folder.
Step 2: Environment Setup
Open a terminal in the project folder and run:
code
Bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the environment
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate
Step 3: Install Dependencies (Important!)
You need to install PyTorch with GPU support specifically. Run these commands in order:
code
Bash
# 1. Install PyTorch with GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies
pip install -r requirements.txt
Step 4: Configure Stable Diffusion (Optional)
To enable direct image generation from the UI:
Go to your Stable Diffusion (Automatic1111) folder.
Edit the webui-user.bat file.
Add --api to the arguments line. Example:
set COMMANDLINE_ARGS=--xformers --api
Launch Stable Diffusion and keep it running in the background.
Step 5: Launch AI Studio
You can run the program using:
Double-click on avvia_studio.bat (Windows).
Or via terminal: python advanced_prompt_gen.py
The interface will open in your browser at http://127.0.0.1:7863.