import os
import sys
import argparse
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

# 1. Load Environment Variables & Enforce Offline Mode
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"  # Ensures local files are used exclusively

ANYECG_HOME = os.getenv("ANYECG_HOME")
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH")

if not ANYECG_HOME or not os.path.exists(ANYECG_HOME):
    raise ValueError("Invalid or missing ANYECG_HOME path in your .env file.")
if not HF_MODEL_PATH or not os.path.exists(HF_MODEL_PATH):
    raise ValueError("Invalid or missing HF_MODEL_PATH path in your .env file.")

# 2. Append repository to system path to import repository modules
sys.path.append(ANYECG_HOME)

try:
    # Importing model components based on anyECG repository structure
    from anyecg.ecg_language_modeling import AnyECGModel
    from anyecg.utils import preprocess_ecg  # Assuming utility preprocessing exists
except ImportError as e:
    print(f"Error importing from anyECG repository: {e}")
    sys.exit(1)


class AnyECGPipeline:
    def __init__(self):
        print("Initializing AnyECG-chat pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load local tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_PATH, 
            local_files_only=True
        )
        
        # Load the AnyECG model instance using local weights / checkpoint directory
        self.model = AnyECGModel.from_pretrained(
            HF_MODEL_PATH,
            local_files_only=True
        ).to(self.device)
        self.model.eval()

    def generate_report(self, ecg_tensor_or_path):
        """
        Runs the ReportGen task using standard prompting[cite: 3].
        """
        prompt = "Please provide the report for the following ECG."
        response = self._inference(ecg_tensor_or_path, prompt)
        return response

    def localize_waveform(self, ecg_tensor_or_path, abnormality_name):
        """
        Runs the Localization task to find second-level intervals of a condition[cite: 11].
        """
        prompt = f"Can you show me where the {abnormality_name} occurred on this ECG?"
        response = self._inference(ecg_tensor_or_path, prompt)
        return response

    def _inference(self, ecg_input, prompt):
        """
        Internal helper handling data preparation, tokenization, and generation.
        """
        try:
            # Preprocess or load ECG data into the expected tensor shape (e.g., [Leads, T])
            if isinstance(ecg_input, str):
                import numpy as np
                raw_data = np.load(ecg_input)
                ecg_tensor = torch.tensor(raw_data, dtype=torch.float32)
            else:
                ecg_tensor = ecg_input

            # Format input through tokenizer / model processor
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    ecg_signals=ecg_tensor.unsqueeze(0).to(self.device),
                    max_new_tokens=256,
                    do_sample=False
                )
                
            result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            print(f"An error occurred during inference execution: {e}")
            return None


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Run AnyECG-chat Report Generation and Waveform Localization.")
    parser.add_argument(
        "--ecg_path", 
        type=str, 
        required=True, 
        help="Path to the input ECG file (e.g., .npy format)"
    )
    parser.add_argument(
        "--abnormality", 
        type=str, 
        default="Premature ventricular contraction", 
        help="Target waveform abnormality to locate"
    )
    
    args = parser.parse_args()

    try:
        if not os.path.exists(args.ecg_path):
            print(f"Error: The ECG file path '{args.ecg_path}' does not exist.")
            sys.exit(1)

        pipeline = AnyECGPipeline()
        
        # 1. Test Report Generation
        report = pipeline.generate_report(args.ecg_path)
        if report:
            print("\n--- Generated Report ---")
            print(report)
        
        # 2. Test Waveform Localization
        localization_result = pipeline.localize_waveform(args.ecg_path, args.abnormality)
        if localization_result:
            print(f"\n--- Localization Result ({args.abnormality}) ---")
            print(localization_result)
            
    except Exception as e:
        print(f"Failed to initialize or run the pipeline: {e}")