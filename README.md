# MultiSAAl: Sequence-informed Antibody-Antigen Interactions Prediction using Multi-scale Deep Learning

## 🧪 Method Overview

Input the antibody heavy chain variable region, light chain variable region and antigen sequence to predict the binding probability of antibody and antigen.

---

## 🚀 Getting Started


To run this project, You should make sure you have ANARCI and CALIBER installed.
In addition:
numpy==1.22.4/n
pandas==1.4.3/n
scikit_learn==1.2.2/n
scipy==1.7.3/n
torch==1.12.1/n
tqdm==4.64.1/n
transformers==4.24.0


## 🏃 Running the MultiSAAI
Functional Features:
After installation, run CALIBER to extract epitopes and CDR-H3 to generate functional features: feature/abag
```
Manual Features:
feature/miler
```
Insert features into: trainer/bert_finetuning_er_trainer_sabdab.py (bert_finetuning_er_trainer_covabdab.py)
💡 Example: Run MultiSAAI
run_train.sh/Online Server (http://zhanglab-bioinf.com/MultiSAAI/)


