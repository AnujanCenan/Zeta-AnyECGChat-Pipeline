import os
import sys
import argparse
import torch
import numpy as np
import wfdb
from scipy.signal import resample
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

# 2. Append repository root to system path
sys.path.append(ANYECG_HOME)

try:
    from anyecg.ecg_language_modeling import ECG_Language_Model
    from anyecg.utils import ecg_transform
except ImportError as e:
    print(f"Error importing from anyECG repository: {e}")
    sys.exit(1)


def load_flexible_ecg(record_path, sampling_freq=100):
    """
    Loads raw WFDB records (.dat and .hea) natively via wfdb,
    resamples to target frequency, and formats to [Leads, T].
    """
    ecg, meta_info = wfdb.rdsamp(record_path)
    
    if sampling_freq != meta_info['fs']:
        ecg = resample(ecg, int(ecg.shape[0] * sampling_freq / meta_info['fs']), axis=0)
        
    # Transpose from [T, Leads] to model-expected shape [Leads, T]
    ecg = ecg.T
    return ecg


class AnyECGPipeline:
    def __init__(self):
        print("Initializing ECG_Language_Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model following the repository's native structure[cite: 1]
        self.model = ECG_Language_Model()
        
        # Load Checkpoints
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

    def process_and_predict(self, record_path, prompt, sampling_freq=100):
        try:
            # 1. Load flexible lead data using wfdb
            raw_ecg = load_flexible_ecg(record_path, sampling_freq=sampling_freq)
            
            # 2. Apply repository's native normalization / transformation pipeline
            processed_ecg = ecg_transform(raw_ecg)
            
            # 3. Convert to tensor and add batch dimension: [1, Leads, T]
            ecg_tensor = torch.tensor(processed_ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            messages = [[{"role": "user", "content": prompt}]]
            
            with torch.no_grad():
                # Call repository's native ecg_chat method[cite: 1]
                response = self.model.ecg_chat(ecg_tensor, messages, temperature=0.6)
                
            return response[0] if isinstance(response, list) else response
            
        except Exception as e:
            print(f"Error during inference execution: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AnyECG-chat Offline Inference on WFDB Records.")
    parser.add_argument("--record_path", type=str, required=True, help="Path to WFDB record WITHOUT extension (e.g., datasets/mit_bih/100)")
    parser.add_argument("--mode", type=str, default="report", choices=["report", "locate"], help="Task mode: report generation or waveform localization")
    parser.add_argument("--abnormality", type=str, default="Right Bundle Branch Block", help="Abnormality name for localization query")
    args = parser.parse_args()

    try:
        pipeline = AnyECGPipeline()

        # Build prompt based on task mode
        if args.mode == "report":
            prompt = "Please provide the report for the following ECG."
        else:
            prompt = f"Can you show me where the {args.abnormality} occurred on this ECG?"

        result = pipeline.process_and_predict(args.record_path, prompt)
        if result:
            print(f"\n--- Model Output ({args.mode.upper()}) ---")
            print(result)

    except Exception as e:
        print(f"Execution failed: {e}")