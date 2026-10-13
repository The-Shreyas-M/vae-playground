@echo off
REM Start the VAE Latent Space Explorer dashboard.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py --server.showEmailPrompt false --browser.gatherUsageStats false
pause