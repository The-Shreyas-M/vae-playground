@echo off
REM Start the VAE Latent Space Explorer dashboard with the A-Z alphabet model pre-selected.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py --server.showEmailPrompt false --browser.gatherUsageStats false -- --dataset emnist-letters
pause
