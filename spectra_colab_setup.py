# ==============================================================================
# SPECTRA SURGICAL AI — GOOGLE COLAB CINEMATIC PIPELINE
# ==============================================================================
# PASTE THIS ENTIRE CELL INTO GOOGLE COLAB (Runtime: GPU T4)
# ==============================================================================

import os
import time
import subprocess
from IPython.display import Image, display

# 1. SETUP ENVIRONMENT
print("[+] Initializing Cloud Environment...")
!git clone https://github.com/ayushmantrivedi/SPECTRA-AI.git
%cd SPECTRA-AI
!pip install -q diffusers accelerate transformers torch torchvision requests pillow

# 2. ORCHESTRATE OLLAMA (The "Brain" for Spectral Syncing)
print("[+] Orchestrated Ollama Background Service...")
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(10) # Grant time for service to heartbeat
!ollama pull phi3:mini

# 3. CONFIGURE HIGH-SPEED GPU LOADING
# On Colab (16GB VRAM), we bypass "Nuclear Stability" to get 10x faster inference.
os.environ["SPECTRA_FULL_GPU"] = "1" 

# 4. EXECUTE CINEMATIC EDIT
prompt = "Make the woman's hair silver and her suit red" # @param {type:"string"}
input_image = "real_woman.jpg" # @param ["real_woman.jpg", "real_mountainsky.jpg", "real_object.jpg"]

print(f"\n[🚀] EXECUTING SPECTRA PIPELINE: {prompt}")
!python test_e2e_edit.py --image {input_image} --prompt "{prompt}" --out colab_result.png

# 5. DISPLAY RESULTS
print("\n--- SPECTRA SURGICAL OUTPUT -------------------------------------------")
display(Image("colab_result.png", width=512))
