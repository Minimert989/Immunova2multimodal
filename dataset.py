


import torch
from torch.utils.data import Dataset
import pandas as pd

class PatientDataset(Dataset):
    def __init__(self, metadata_csv, wsi_features, omics_dict, label_dict, input_dims):
        """
        metadata_csv: path to patient_data_modality_status.csv
        wsi_features: dict of {patient_id: Tensor [1024]}
        omics_dict: dict of {omics_type: {patient_id: Tensor}}
            omics_type ∈ {'rna', 'methyl', 'protein', 'mirna'}
        label_dict: dict of {patient_id: {'til': Tensor, 'response': Tensor}}
        input_dims: dict of omics dimensions for zero vector fallback
        """
        self.df = pd.read_csv(metadata_csv)
        self.wsi_features = wsi_features
        self.omics_dict = omics_dict
        self.label_dict = label_dict
        self.input_dims = input_dims

        self.omics_keys = ['rna', 'methyl', 'protein', 'mirna']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['patient_id']

        sample = {}

        # WSI
        if row['has_wsi'] and pid in self.wsi_features:
            sample['wsi_feat'] = self.wsi_features[pid]
        else:
            sample['wsi_feat'] = None  # fusion handles None

        # Omics
        for k in self.omics_keys:
            if row['has_omics'] and pid in self.omics_dict.get(k, {}):
                sample[k] = self.omics_dict[k][pid]
            else:
                sample[k] = torch.zeros(self.input_dims[k], dtype=torch.float32)

        # Labels
        label = self.label_dict.get(pid, {})
        sample['til_label'] = label.get('til', torch.tensor([0, 0, 0, 0], dtype=torch.float32))  # default: no TIL
        sample['response_label'] = label.get('response', torch.tensor([-1.0], dtype=torch.float32))  # mask if unavailable
        sample['status_label'] = label.get('status', torch.tensor([-1.0], dtype=torch.float32))  # mask if unavailable
        sample['survival_time'] = label.get('survival_time', torch.tensor([-1.0], dtype=torch.float32))  # mask if unavailable

        return sample