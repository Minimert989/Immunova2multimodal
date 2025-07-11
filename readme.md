📘 Immunova Multimodal AI Pipeline — Reproducibility Instructions

This document explains how to run the full Immunova pipeline, including data preparation, model training, validation, and prediction. It assumes that you have access to both WSI (.pt) and omics data.


📁 Project Directory Structure

Immunova/
├── model.py                    # Defines the multimodal transformer model
├── dataset.py                  # PatientDataset for dataloader
├── train.py                    # Training loop
├── validate.py                 # Evaluation loop
├── predict.py                  # Prediction and visualization
├── Omics_feature.py            # Generate omics_dict.pkl
├── label_feature.py            # Generate label_dict.pkl
├── WSI_feature.py              # Extract WSI features from .pt
├── til_model.pth               # (Optional) Pretrained model
│
├── wsi_feature/
│   └── wsi_feature.pkl         # Pre-extracted WSI features (from .pt)
│
├── omics_feature/
│   └── omics_dict.pkl          # Preprocessed omics data per patient
│
├── label_feature/
│   └── label_dict_ACC.pkl      # Labels (TIL, response, survival)
│
├── val_ids.txt                 # Patient IDs for validation
├── predictions_response.csv    # Output predictions (generated)
│
├── Immunova2_Module1/          # WSI patch-level .pt files
│   ├── acc/
│   │   ├── TCGA-XX-YYYY.pt
│   │   ├── TCGA-XX-ZZZZ.pt
│   │   └── ...
│   ├── brca/
│   ├── blca/
│   └── ...                     # Other cancer types
│
└── Immunova2_module2/          # Omics and clinical data per cancer type
    ├── ACC/
    │   ├── TCGA_clinical_ACC.csv
    │   ├── TCGA_rnaseq_ACC_immune_markers_with_metadata.csv
    │   ├── TCGA_methylation_ACC_immune_sites_with_metadata.csv
    │   ├── TCGA_mirna_ACC_immune_markers_with_metadata.csv
    │   ├── TCGA_rppa_ACC_immune_proteins_with_metadata.csv
    │   └── ...
    ├── BRCA/
    ├── BLCA/
    └── ...                     # Other cancer types










Code  structure :          ┌────────────────────┐
                           │    Raw Data Files  │
                           │ (WSI, RNA, Methyl…)│
                           └────────┬───────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │                             │
      ┌──────────────▼────────────┐   ┌────────────▼────────────┐
      │   WSI_feature.py          │   │   Omics_feature.py       │
      │ → pt → wsi_feature.pkl    │   │ → CSV → omics_dict.pkl   │
      └──────────────┬────────────┘   └────────────┬─────────────┘
                     │                             │
                     │    ┌────────────────────────▼────────────────────┐
                     │    │         label_feaure.py                     │
                     │    │ → clinical CSV → label_dict.pkl             │
                     │    └─────────────────────────────────────────────┘
                     │                             │
          ┌──────────▼─────────────────────────────▼───────────────┐
          │                      dataset.py                        │
          │  → PatientDataset  (wsi + omics + label)               │
          └──────────┬─────────────────────────────┬───────────────┘
                     │                             │
        ┌────────────▼────────────┐   ┌────────────▼────────────┐
        │       train.py          │   │      validate.py        │
        │ 학습 수행, model.pth      │         Validation , loss   │
        └────────────┬────────────┘   └────────────┬────────────┘
                     │                             │
                     │    ┌────────────────────────▼──────────────────────────┐
                     │    │                    predict.py                     │
                     │    │ visulizaton : CSV, ROC, PR Curve                  │
                     │    └───────────────────────────────────────────────────┘
                     ▼
               model.pth 


 Model structure : 
             +-----------------------+       +-----------------------+
              |  WSI Patch Input (.pt)|       |    RNA-seq (10k+)     |
              |   (B, N, 1024)        |       +-----------------------+
              +----------+------------+       |   Methylation (450k+) |
                         |                    |   Protein (3k~)       |
                         |                    |   miRNA (1k~)         |
                         |                    +-----------+-----------+
           +-------------▼------------+                    |
           |  Transformer Encoder     |       +------------▼------------+
           |  (ViT-like WSI Encoding) |       | OmicsSubEncoder × 4     |
           |  (shared across patches) |       | Each:                   |
           +-------------+------------+       |  - Linear projection    |
                         |                    |  - TransformerEncoder   |
                         |                    +------------+-----------+
                         |                                 |
        +----------------▼----------------+      +---------▼---------+
        |  WSI Patch Tokens (B, N, D)     |      | Omics Token (B, 1, D)|
        +----------------+----------------+      +---------+-----------+
                         |                           |
                         +------------+--------------+
                                      |
                  +------------------▼--------------------+
                  |    [CLS] Token + WSI + Omics Tokens   |
                  |        → CrossModal Transformer       |
                  |     (Multi-head Attention Layers)     |
                  +------------------+--------------------+
                                     |
                       +------------▼-------------+
                       | Integrated Representation |
                       | (B, D) via [CLS] token     |
                       +------------+--------------+
                                     |
      +------------------------------+-------------------------------+
      |                                                              |
+-----▼-----+                                              +--------▼--------+
|  TIL Head |                                              | Response Head   |
| MultiLabel|                                              | Binary Classify |
+-----------+                                              +-----------------+
                                                      
                                                    +------------------------+
                                                    | Survival Head          |
                                                    | Regression / Time-to-E |
                                                    +------------------------+


Instructions:                                                   j
