import torch
from torch.utils.data import DataLoader
from model import ImmunovaMultimodalModel
from dataset import PatientDataset
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle

task = "response"  # options: 'til', 'response', 'survival'

# Load features and labels
with open("wsi_feature/wsi_feature.pkl", "rb") as f:
    wsi_features = pickle.load(f)

with open("omics_feature/omics_dict.pkl", "rb") as f:
    omics_dict = pickle.load(f)

with open("label_feature/label_dict_ACC.pkl", "rb") as f:
    label_dict = pickle.load(f)

input_dims = {"rna": 1000, "methyl": 512, "protein": 256, "mirna": 128}

# Train-validation split
common_ids = list(set(wsi_features.keys()) & set(label_dict.keys()))
train_ids, val_ids = train_test_split(common_ids, test_size=0.2, random_state=42)

with open("val_ids.txt", "w") as f:
    f.writelines([pid + "\n" for pid in val_ids])

# Dataset and loader
train_dataset = PatientDataset(train_ids, wsi_features, omics_dict, label_dict, input_dims)
val_dataset = PatientDataset(val_ids, wsi_features, omics_dict, label_dict, input_dims)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model and optimizer
model = ImmunovaMultimodalModel(input_dims=(1000, 512, 256, 128))
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(5):  # adjust as needed
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move tensors to device
        wsi_feat = batch["wsi_feat"]
        if wsi_feat is not None:
            wsi_feat = wsi_feat.to(device)  # (B, N, 1024)

        rna = batch["rna"].to(device)
        methyl = batch["methyl"].to(device)
        prot = batch["protein"].to(device)
        mirna = batch["mirna"].to(device)

        til_label = batch["til_label"].to(device)
        response_label = batch["response_label"].to(device)
        status_label = batch["status_label"].to(device)
        survival_time = batch["survival_time"].to(device)

        til_pred, resp_pred, survival_pred = model(
            wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
        )

        if task == "til":
            mask = (til_label.sum(dim=1) >= 0)  # valid if any label exists
            loss = F.binary_cross_entropy_with_logits(til_pred[mask], til_label[mask])
        elif task == "response":
            mask = (response_label != -1)
            loss = F.binary_cross_entropy_with_logits(resp_pred[mask], response_label[mask])
        elif task == "survival":
            mask = (survival_time != -1)
            pred = survival_pred.squeeze()
            loss = F.mse_loss(pred[mask], survival_time[mask])
        else:
            raise ValueError("Invalid task")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            wsi_feat = batch["wsi_feat"]
            if wsi_feat is not None:
                wsi_feat = wsi_feat.to(device)  # (B, N, 1024)

            rna = batch["rna"].to(device)
            methyl = batch["methyl"].to(device)
            prot = batch["protein"].to(device)
            mirna = batch["mirna"].to(device)

            til_label = batch["til_label"].to(device)
            response_label = batch["response_label"].to(device)
            survival_time = batch["survival_time"].to(device)

            til_pred, resp_pred, survival_pred = model(
                wsi_feat=wsi_feat, rna=rna, methyl=methyl, prot=prot, mirna=mirna
            )

            if task == "til":
                mask = (til_label.sum(dim=1) >= 0)
                val_loss += F.binary_cross_entropy_with_logits(til_pred[mask], til_label[mask]).item()
            elif task == "response":
                mask = (response_label != -1)
                val_loss += F.binary_cross_entropy_with_logits(resp_pred[mask], response_label[mask]).item()
            elif task == "survival":
                mask = (survival_time != -1)
                pred = survival_pred.squeeze()
                val_loss += F.mse_loss(pred[mask], survival_time[mask]).item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f} - Val Loss: {val_loss:.4f}")