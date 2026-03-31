import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
from transformers import T5TokenizerFast
from models.cmelt import M3AEModel
from types import SimpleNamespace
import torch.nn.functional as F
from data_load import ECGDataset

# Define category labels for different datasets
categories = {
    "icbeb": ["AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD"],
    "chapman": ["AQW", "UW", "SR", "WPW", "2AVB", "AT", "VB", "ARS", "STTC", "SA", "STE", "VPB", "TWO", 
                "STTU", "ALS", "APB", "2AVB1", "PRIE", "CCR", "CR", "AF", "AVB", "QTIE", "LBBB", "VEB", 
                "SVT", "RBBB", "1AVB", "STDD", "MI", "AFIB", "TWC", "PWC", "ERV", "RVH", "LVH", "ST", "JEB"],
    "super_class": ["HYP", "NORM", "MI", "CD", "STTC"],
    "sub_class": ["AMI", "LAFB/LPFB", "LVH", "STTC", "IMI", "SEHYP", "CRBBB", "WPW", "LAO/LAE", "NORM", 
                "ISC", "AVB", "RAO/RAE", "LMI", "ISCI", "ISCA", "NST", "CLBBB", "ILBBB", "IRBBB", "PMI", 
                "RVH", "IVCD"],
    "rhythm": ["SVARR", "BIGU", "STACH", "SARRH", "SBRAD", "TRIGU", "AFIB", "SR", "SVTAC", "PSVT", "AFLT", "PACE"],
    "form": ["PAC", "DIG", "HVOLT", "STD", "LPR", "QWAVE", "VCLVH", "NDT", "TAB", "LOWT", "ABQRS", "LNGQT", 
            "PRC(S)", "NT", "STE", "NST", "PVC", "INVT", "LVOLT"]
} 

# Dataset mapping configuration
DATASET_CONFIG = {
    "icbeb": {
        "dataset_name": "icbeb",
        "dataset_path": "path/to/icbeb",
        "test_csv_path": "data_split/icbeb/icbeb_test.csv"
    },
    "chapman": {
        "dataset_name": "chapman",
        "dataset_path": "path/to/chapman",
        "test_csv_path": "data_split/chapman/chapman_test.csv"
    },
    "super_class": {
        "dataset_name": "ptbxl",
        "dataset_path": "path/to/ptbxl",
        "test_csv_path": "data_split/ptbxl/super_class/ptbxl_super_class_test.csv"
    },
    "sub_class": {
        "dataset_name": "ptbxl",
        "dataset_path": "path/to/ptbxl",
        "test_csv_path": "data_split/ptbxl/sub_class/ptbxl_sub_class_test.csv"
    },
    "rhythm": {
        "dataset_name": "ptbxl",
        "dataset_path": "path/to/ptbxl",
        "test_csv_path": "data_split/ptbxl/rhythm/ptbxl_rhythm_test.csv"
    },
    "form": {
        "dataset_name": "ptbxl",
        "dataset_path": "path/to/ptbxl",
        "test_csv_path": "data_split/ptbxl/form/ptbxl_form_test.csv"
    }
}

def load_encoders():
    """
    Load models and encoders
    
    Returns:
        tuple: Contains all necessary model components
    """
    try:
        with open('configs/config.json', 'r') as json_file:
            cfg = json.load(json_file)
        cfg = SimpleNamespace(**cfg['model'])
        model = M3AEModel(cfg)
        checkpoint = torch.load("checkpoints/best.pt")
        if "ecg_encoder.mask_emb" in checkpoint["model"].keys():
            del checkpoint["model"]["ecg_encoder.mask_emb"]
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return (
            model.cuda(), model.ecg_encoder.cuda(), model.language_encoder.cuda(),
            model.unimodal_ecg_pooler.cuda(), model.unimodal_language_pooler.cuda(),
            model.multi_modal_ecg_proj.cuda(), model.multi_modal_language_proj.cuda(),
            model.class_embedding.cuda()
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
def extract_language_features(model, texts, pooler=None, proj=None, add_content=False, all_labels=[]):
    """
    Extract language features from text
    
    Args:
        model: Language model
        texts: List of texts
        pooler: Feature pooler
        proj: Feature projector
        add_content: Whether to add content indicator
        all_labels: List of all labels
        
    Returns:
        dict: Dictionary containing features for each label
    """
    model = model.cuda()
    model_name = "google/flan-t5-base"
    tokenizer = T5TokenizerFast.from_pretrained(model_name, do_lower_case="uncased" in model_name)
    
    features_dict = {}

    for type_feature, biclass in enumerate(tqdm(texts, desc="Extracting language features")):
        max_pooled_features_list_bi = []
        for cls in biclass:
            max_pooled_features_list = []
            for text in cls:
                if add_content:
                    text = f'{text} indicating {all_labels[type_feature]}'
                encoded_input = tokenizer(text, truncation=True, return_tensors="pt").to('cuda')

                with torch.no_grad():
                    outputs = model(**encoded_input)[0]
                    outputs = proj(outputs)

                if pooler is None:
                    max_pooled_features, _ = torch.max(outputs, dim=1)
                else:
                    max_pooled_features = pooler(outputs)

                max_pooled_features_list.append(max_pooled_features.cpu().squeeze(0).detach().numpy())
            max_pooled_features_list_bi.append(max_pooled_features_list)
        features_dict[all_labels[type_feature]] = max_pooled_features_list_bi

    return features_dict

def extract_ecg_features(model, ecgs, pooler=None, proj=None, class_embedding=None, batch_size=100, datasets="ptbxl"):
    """
    Extract features from ECG data
    
    Args:
        model: ECG model
        ecgs: ECG data
        pooler: Feature pooler
        proj: Feature projector
        class_embedding: Class embedding
        batch_size: Batch size
        datasets: Dataset name
        
    Returns:
        numpy.ndarray: Array of ECG features
    """
    features = []
    num_ecgs = len(ecgs)

    for start_idx in tqdm(range(0, num_ecgs, batch_size), desc="Extracting ECG features"):
        end_idx = min(start_idx + batch_size, num_ecgs)
        ecg_batch = torch.tensor(ecgs[start_idx:end_idx], dtype=torch.float32).cuda()
        if datasets != "ptbxl":
            ecg_batch = ecg_batch.permute(0, 2, 1)
        with torch.no_grad():

            uni_modal_ecg_feats, ecg_padding_mask = (
                model.get_embeddings(ecg_batch, padding_mask=None)
            )

            cls_emb = class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
            uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
            uni_modal_ecg_feats = model.get_output(uni_modal_ecg_feats, ecg_padding_mask)
            out = proj(uni_modal_ecg_feats)
            ecg_features = pooler(out)
            features.append(ecg_features.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def get_diseases_probs(list_features):
    """
    Calculate disease probabilities
    
    Args:
        list_features: List of features
        
    Returns:
        tuple: Lists of positive and negative probabilities
    """
    combined_tensor = torch.tensor(list_features).T
    feature_p_list, feature_n_list = [], []
    for i in combined_tensor:
        feature_p, feature_n = torch.softmax(i / 0.5, dim=0)
        feature_p_list.append(feature_p)
        feature_n_list.append(feature_n)
    return feature_p_list, feature_n_list

def change_arg(cls, args):
    """
    Update parameters based on specified category
    
    Args:
        cls: Category name
        args: Parameter object
        
    Returns:
        argparse.Namespace: Updated parameter object
    """
    if cls in DATASET_CONFIG:
        config = DATASET_CONFIG[cls]
        args.dataset = cls
        args.dataset_name = config["dataset_name"]
        args.dataset_path = config["dataset_path"]
        args.test_csv_path = config["test_csv_path"]
    return args

def inference(args):
    """
    Execute inference process
    
    Args:
        args: Command line arguments
        
    Returns:
        float: Average AUC value
    """
    try:
        ecg_dataset = ECGDataset(args)        
        all_labels = categories[args.dataset]
        
        with open(args.json_path, 'r') as file:
            converting_tool = json.load(file)
            potential_labels_C = []
            for label in all_labels:
                if label in converting_tool:
                    potential_labels_C.append([
                        [m.lower() for m in converting_tool[label]["P"]],
                        [m.lower() for m in converting_tool[label]["N"]]
                    ])

        print("Loading model encoders...")
        model, ecg_model, language_model, unimodal_ecg_pooler, unimodal_language_pooler, multi_modal_ecg_proj, multi_modal_language_proj, class_embedding = load_encoders()
        
        print("Extracting language features...")
        potential_language_features_dict = extract_language_features(language_model, potential_labels_C, unimodal_language_pooler, multi_modal_language_proj, all_labels=all_labels)
        
        train_loader = torch.utils.data.DataLoader(ecg_dataset, batch_size=800, shuffle=False, num_workers=4)
        
        print("Extracting ECG features...")
        ecg_features, diagnoses_test = [], []
        for step, (ecg, target) in tqdm(enumerate(train_loader), desc="Processing data batches"):
            ecg = ecg.permute(0, 2, 1)
            ecg_feature = extract_ecg_features(ecg_model, ecg.to('cuda'), unimodal_ecg_pooler, multi_modal_ecg_proj, class_embedding, batch_size=100, datasets=args.dataset_name)
            ecg_features.append(torch.tensor(ecg_feature))
            diagnoses_test.append(torch.tensor(target))
            
        ecg_features = torch.cat(ecg_features, dim=0)
        diagnoses_test = torch.cat(diagnoses_test, dim=0)
            
        diagnoses_test_np = diagnoses_test.numpy()
        results = []

        for i, label in enumerate(all_labels):
            pos_count = int(diagnoses_test_np[:, i].sum())
            neg_count = diagnoses_test_np.shape[0] - pos_count
            results.append({
                "label": label,
                "positive_samples": pos_count,
                "negative_samples": neg_count
            })
        
        all_auc = []

        print("\nCalculating AUC for each disease...")
        for i_label, class_label in enumerate(all_labels):
            similarities = []
            for i, ecg_feature in enumerate(ecg_features):
                ecg_feature = torch.tensor(ecg_feature).cuda()
                for label, language_feature in potential_language_features_dict.items():
                    if label == class_label:                
                        language_feature_p = torch.tensor(language_feature[0]).cuda()
                        language_feature_n = torch.tensor(language_feature[1]).cuda()
                        ecg_feature = F.normalize(ecg_feature, dim=0)
                        
                        similarities_p = [(ecg_feature @ F.normalize(l, dim=0).reshape(1, -1).T)[0] for l in language_feature_p]
                        similarities_n = [(ecg_feature @ F.normalize(l, dim=0).reshape(1, -1).T)[0] for l in language_feature_n]
                        
                        bicls = [similarities_p, similarities_n]
                        feature_p, _ = get_diseases_probs(bicls)
                        similarity = torch.mean(torch.tensor(feature_p).cpu())
                        similarities.append(similarity)

            position = all_labels.index(class_label)
            auc_mean = roc_auc_score(diagnoses_test_np[:, position], torch.tensor(similarities))
            all_auc.append(auc_mean)
            print(f"{class_label} mean AUC: {auc_mean:.4f}")
            
        auc_value = sum(all_auc) / len(all_auc)
        allauc_formatted = f"{auc_value:.4f}"
        print(f"\nAll mean AUC: {allauc_formatted}")
        return auc_value
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="ECG Disease Classification Inference Tool")
    argparser.add_argument('--dataset', type=str, default="icbeb", choices=list(DATASET_CONFIG.keys()),
                          help="Dataset name")
    argparser.add_argument('--json_path', type=str, default='configs/observations.json',
                          help="Observation config JSON path")
    argparser.add_argument('--dataset_name', type=str, default='',
                          help="Dataset name, usually set by change_arg function")
    argparser.add_argument('--dataset_path', type=str, default='',
                          help="Dataset path, usually set by change_arg function")
    argparser.add_argument('--test_csv_path', type=str, default='',
                          help="Test CSV file path, usually set by change_arg function")
    args = argparser.parse_args()


    inference(change_arg("icbeb", args))

