

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
from dataset import PatientDataset
from model import ImmunovaMultimodalModel

# === 설정 ===
task = "response"  # or 'til', 'survival'
input_dims = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 데이터 로드 ===
with open("wsi_feature/wsi_feature.pkl", "rb") as f:
    wsi_features = pickle.load(f)

with open("omics_feature/omics_dict.pkl", "rb") as f:
    omics_dict = pickle.load(f)

with open("label_feature/label_dict_ACC.pkl", "rb") as f:
    label_dict = pickle.load(f)

with open("val_ids.txt", "r") as f:
    val_ids = [line.strip() for line in f.readlines()]

val_dataset = PatientDataset(val_ids, wsi_features, omics_dict, label_dict, input_dims)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === 모델 로딩 ===
model = ImmunovaMultimodalModel(input_dims=(1000, 512, 256, 128))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# === 검증 ===
total_loss = 0
with torch.no_grad():
    for batch in val_loader:
        wsi_feat = batch["wsi_feat"]
        if wsi_feat is not None:
            wsi_feat = wsi_feat.to(device)

        rna = batch["rna"].to(device)
        methyl = batch["methyl"].to(device)
        prot = batch["protein"].to(device)
        mirna = batch["mirna"].to(device)

        til_label = batch["til_label"].to(device)
        response_label = batch["response_label"].to(device)
        survival_time = batch["survival_time"].to(device)

        til_pred, resp_pred = model(
            wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
        )

        if task == "til":
            mask = (til_label.sum(dim=1) >= 0)
            loss = F.binary_cross_entropy_with_logits(til_pred[mask], til_label[mask])
        elif task == "response":
            mask = (response_label != -1)
            loss = F.binary_cross_entropy_with_logits(resp_pred[mask], response_label[mask])
        elif task == "survival":
            mask = (survival_time != -1)
            pred = resp_pred.squeeze()
            loss = F.mse_loss(pred[mask], survival_time[mask])
        else:
            raise ValueError("Unknown task")

        total_loss += loss.item()

print(f"[VALIDATION] {task} Loss: {total_loss:.4f}")