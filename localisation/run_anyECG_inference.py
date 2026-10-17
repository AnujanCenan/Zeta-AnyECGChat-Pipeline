import os
import sys
import argparse
import torch
import numpy as np
from dotenv import load_dotenv

# 1. Load Environment Variables & Enforce Offline Mode
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"

ANYECG_HOME = os.getenv("ANYECG_HOME")
PROJECTION_CKPT = os.getenv("PROJECTION_CKPT")
ECG_MODEL_CKPT = os.getenv("ECG_MODEL_CKPT")
LORA_CKPT = os.getenv("LORA_CKPT")

if not ANYECG_HOME or not os.path.exists(ANYECG_HOME):
    raise ValueError("Invalid or missing ANYECG_HOME path in your .env file.")

# 2. Append repository to system path
sys.path.append(ANYECG_HOME)

try:
    from anyecg.ecg_language_modeling import ECG_Language_Model
    from anyecg.utils import ecg_transform
except ImportError as e:
    print(f"Error importing from anyECG repository: {e}")
    sys.exit(1)


class AnyECGPipeline:
    def __init__(self):
        print("Initializing ECG_Language_Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model as done in inference.py
        self.model = ECG_Language_Model()
        
        # Load Checkpoints if provided
        if PROJECTION_CKPT and os.path.exists(PROJECTION_CKPT):
            res = self.model.projection.load_state_dict(torch.load(PROJECTION_CKPT, map_location=self.device))
            print(f"Loaded projection checkpoint: {res}")
            
        if ECG_MODEL_CKPT and os.path.exists(ECG_MODEL_CKPT):
            res = self.model.ecg_model.load_state_dict(torch.load(ECG_MODEL_CKPT, map_location=self.device))
            print(f"Loaded ECG model checkpoint: {res}")
            
        if LORA_CKPT and os.path.exists(LORA_CKPT):
            self.model.language_model.load_adapter(LORA_CKPT)
            print(f"Loaded LoRA adapter from: {LORA_CKPT}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_report(self, ecg_tensor):
        prompt = "Please provide the report for the following ECG."
        return self._inference(ecg_tensor, prompt)

    def localize_waveform(self, ecg_tensor, abnormality_name):
        prompt = f"Can you show me where the {abnormality_name} occurred on this ECG?"
        return self._inference(ecg_tensor, prompt)

    def _inference(self, ecg_tensor, prompt):
        try:
            # Format messages exactly like inference.py expects
            messages = [[{"role": "user", "content": prompt}]]
            
            # Ensure batch dimension and proper device placement
            if ecg_tensor.dim() == 2:
                ecg_tensor = ecg_tensor.unsqueeze(0)
            ecg_tensor = ecg_tensor.to(self.device)

            with torch.no_grad():
                # Call the repository's native ecg_chat method
                response = self.model.ecg_chat(ecg_tensor, messages, temperature=0.6)
                
            return response[0] if isinstance(response, list) else response
            
        except Exception as e:
            print(f"An error occurred during inference execution: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AnyECG-chat Inference.")
    parser.add_argument("--ecg_path", type=str, required=True, help="Path to input ECG file (.npy)")
    parser.add_argument("--abnormality", type=str, default="Right Bundle Branch Block", help="Abnormality to localize")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.ecg_path):
            print(f"Error: ECG path '{args.ecg_path}' does not exist.")
            sys.exit(1)

        # Load raw sample (ensure it matches expected preprocessing/shape)
        raw_data = np.load(args.ecg_path)
        ecg_tensor = torch.tensor(raw_data, dtype=torch.float32)

        pipeline = AnyECGPipeline()
        
        # 1. Report Generation Test
        report = pipeline.generate_report(ecg_tensor)
        if report:
            print("\n--- Generated Report ---")
            print(report)
        
        # 2. Localization Test (e.g., RBBB or PVC)
        loc_result = pipeline.localize_waveform(ecg_tensor, args.abnormality)
        if loc_result:
            print(f"\n--- Localization Result ({args.abnormality}) ---")
            print(loc_result)

    except Exception as e:
        print(f"Execution failed: {e}")