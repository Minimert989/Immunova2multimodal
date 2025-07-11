import torch
import os
import json
import numpy as np
from PIL import Image
import pickle
import glob

import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # feature vector only
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cancer_types = ["acc", "blca", "brca", "cesc", "coad", "esca", "hnsc", "luad", "lusc", "meso", "thym", "ucec", "uvm"]
base_dir = "/Volumes/MInwoo/Immunova/Immunova2_Module1"

os.makedirs("wsi_feature", exist_ok=True)

for cancer in cancer_types:
    print(f"Processing {cancer}...")
    pt_files = glob.glob(os.path.join(base_dir, cancer, "*.pt"))
    wsi_features = {}

    with torch.no_grad():
        for pt_path in pt_files:
            try:
                data = torch.load(pt_path)
                patient_id = os.path.basename(pt_path).replace(".pt", "")
                patches = data["images"]  # (N, 3, 224, 224)

                processed = []
                for img in patches:
                    np_img = img.permute(1, 2, 0).numpy()
                    pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
                    processed.append(transform(pil_img))

                batch = torch.stack(processed).to(device)
                feats = model(batch)  # (N, 512)
                slide_feat = feats.mean(dim=0)  # (512,)
                wsi_features[patient_id] = slide_feat.cpu()

            except Exception as e:
                print(f"Failed {pt_path}: {e}")

    out_path = os.path.join("wsi_feature", f"wsi_features_{cancer.upper()}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(wsi_features, f)