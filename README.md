# CT-GRAPH

This repository contains the code for the ICCVW paper CT-GRAPH: Hierarchical Graph Attention Network for Anatomy-Guided CT Report Generation. 

Official Paper: [📄 CT-GRAPH](https://www.arxiv.org/pdf/2508.05375)

---

## Installation

```bash
git clone https://github.com/your-username/CT-GRAPH.git
cd CT-GRAPH
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```
---

## Datasets

We utilize two chest CT datasets:

- **CT-RATE** ([HF dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)): Large-scale dataset with ~49k training samples. We filter the reports for duplicates using the official CSV (study/report columns), resulting in **~22.7k** train samples. Anatomy masks are generated with **TotalSegmentator**. Data, masks and .csv files are stored in directory &lt;ctrate_dir&gt.
- **RadGenome-Chest CT** ([HF dataset](https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT)): Only the reports are utilized from this dataset. We use **full reports** for our main models and **region-level reports** for the Reg2RG baseline. Reports/study IDs come from the dataset’s CSV, which is also stored in <ctrate_dir>.

**Expected paths (train):**

```bash
<ctrate_dir>/dataset/train/train_1/train_1_a/train_1_a_1.nii.gz  # CT image
<ctrate_dir>/dataset/train/train_1/train_1_a/ana_train_1_a_1.nii.gz # anatomy mask
<ctrate_dir>/train_reports.csv # reports from CT-RATE
<ctrate_dir>/train_region_reports.csv #reports from RadGenome-Chest CT
```

---

## Feature extraction

Models (or methods) with architecture &lt;arch&gt, defined in feature_extraction/models.py and initialized from checkpoints at &lt;weight_path&gt;, can be used to extract training features with the following command:

```bash
python feature_extraction/extract_features.py --ctrate_dir <ctrate_dir> --mode train --arch <arch> --weight_path <weight_path> 
```

To evaluate features via linear probing from a specific layer &lt;layer&gt;, use the following command:

```bash
python feature_extraction/probing.py --ctrate_dir <ctrate_dir> --arch <arch> --layer <layer> --pool_level all --save_dir <save_dir>
```

To evaluate using a specific pooling level &lt;pool_level&gt;:

```bash
python feature_extraction/extract_features.py --ctrate_dir <ctrate_dir> --arch <arch> --concat_feats True --pool_level <pool_level> --save_dir <save_dir>
```
---

## Training

The main ablations/variants are defined in report_gen/models/variants.py and can be extended by additional feature extractors. All necessary training parameters like the method/architecture, variant, save_paths, etc... 
can be set in the config at report_gen/config.py. To train the report generation model:

```bash
python report_gen/training/train.py
```

---

## Inference

Assuming your report generation model checkpoint is saved at &lt;model_path&gt;, you can generate reports on the validation dataset with the following command:

```bash
python report_gen/inference.py --model_path <model_path>
```

---

## Evaluation

The generated reports are stored in reports.json within <model_dir>. Evaluation includes both NLG metrics and CE metrics. For the latter, the text classification model from 
[CT-CLIP](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/models/RadBertClassifier.pth) is used, which must be stored at <ce_model_path>. Run evaluation with:

```bash
python report_gen/evaluation/evaluation.py --model_path <model_path> --ce_model_path <ce_model_path> 
```

---

## Citation

If you find our work useful, please consider citing:

```bash
@article{kalisch2025ct,
  title={CT-GRAPH: Hierarchical Graph Attention Network for Anatomy-Guided CT Report Generation},
  author={Kalisch, Hamza and H{\"o}rst, Fabian and Kleesiek, Jens and Herrmann, Ken and Seibold, Constantin},
  journal={arXiv preprint arXiv:2508.05375},
  year={2025}
}
```

--- 

## Acknowledgement

This work is built on the projects [M3D](https://github.com/BAAI-DCAI/M3D/tree/main) and the [Llama 2 model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
We thank the authors for their contributions to the community.
