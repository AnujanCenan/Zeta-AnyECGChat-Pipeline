import os
import sys
import argparse
import torch
import numpy as np
from dotenv import load_dotenv

# Load environment and set up path
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
ANYECG_HOME = os.getenv("ANYECG_HOME")
sys.path.append(ANYECG_HOME)

try:
    from anyecg.ecg_language_modeling import ECG_Language_Model
    from anyecg.dataset import get_ecg_from_path  # Import the repository's native loader
    from anyecg.utils import ecg_transform
except ImportError as e:
    print(f"Error importing from anyECG repository: {e}")
    sys.exit(1)

class AnyECGPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ECG_Language_Model().to(self.device)
        self.model.eval()
        # Load your checkpoints here as needed...

    def process_and_predict(self, record_path, prompt, sampling_freq=100):
        try:
            # 1. Use the repo's native function to read .dat/.hea via wfdb, resample, and align leads
            raw_ecg = get_ecg_from_path(record_path, sampling_freq=sampling_freq)
            
            # 2. Apply the repo's built-in ecg_transform (normalization to [-1, 1], etc.)
            processed_ecg = ecg_transform(raw_ecg)
            
            # 3. Convert to tensor and add batch dimension [1, Leads, T]
            ecg_tensor = torch.tensor(processed_ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            messages = [[{"role": "user", "content": prompt}]]
            
            with torch.no_grad():
                response = self.model.ecg_chat(ecg_tensor, messages, temperature=0.6)
                
            return response[0] if isinstance(response, list) else response
            
        except Exception as e:
            print(f"Error processing record: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, required=True, help="Path to WFDB record WITHOUT extension (e.g., data/mit_bih/100)")
    parser.add_argument("--prompt_type", type=str, default="report", choices=["report", "locate"])
    parser.add_argument("--abnormality", type=str, default="Right Bundle Branch Block")
    args = parser.parse_args()

    pipeline = AnyECGPipeline()
    
    if args.prompt_type == "report":
        prompt = "Please provide the report for the following ECG."
    else:
        prompt = f"Can you show me where the {args.abnormality} occurred on this ECG?"

    result = pipeline.process_and_predict(args.record_path, prompt)
    if result:
        print("\n--- Model Output ---")
        print(result)