# LaMGen: Multi-Target Molecule Generation Framework

![image](https://github.com/user-attachments/assets/6660cc47-8105-4bb1-a215-3a50dbf10a7f)


## Introduction

Multi-target drugs hold great promise for treating complex diseases but remain challenging to design due to the need to satisfy multiple binding site constraints while maintaining favorable pharmacokinetics.

Existing AI drug design methods mainly target single proteins and struggle to generalize to multi-target scenarios, especially for **3D dual-target molecule generation**, where effective solutions are lacking.

To address this, we present **LaMGen**:

> The first universal multi-target drug design framework based on large language models.

## Key Features

* 🔹 **Multi-target capability**: Supports **dual-target and triple-target molecular generation**.
* 🔹 **ESM-C embedding**: Encodes multiple protein sequences efficiently.
* 🔹 **Rotation-aware molecular tokens**: Enables spatially valid 3D molecule generation.
* 🔹 **TriCoupleAttention module**: Captures detailed multi-level interactions between targets and ligands.
* 🔹 **Differentiable potential energy surfaces**: Ensures chemical plausibility.
* 🔹 **Task-free generation**: No task-specific retraining required.
* 🔹 **High efficiency**: Generation speed is **30× faster** compared to traditional methods.
* 🔹 **Superior performance**: Outperforms DualDiff and AIxFuse in binding affinity and molecular quality across multiple benchmarks.

---

## Project Structure

```text
LaMGen/
├── .idea/               # IDE configuration files (can ignore)
├── Pretrained_model/    # Pretrained model checkpoints
├── __pycache__/         # Python cache files
├── data/                # Dataset files
├── docking/             # Docking-related scripts
├── model/               # Model architecture and modules
├── scripts/             # Training and generation scripts
├── utils/               # Helper functions and utilities
└── LICENSE              # Project license
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/cholin01/LaMGen.git
cd LaMGen

# (Recommended) Create a new Python environment
conda create -n lamgen python=3.8
conda activate lamgen

# Install required packages
pip install -r requirements.txt
```

---

## Quick Start

### 1. Preprocess Data

Ensure your dual-target datasets are placed under the `data/` directory.

### 2. Train Model

```bash
python scripts/train_triple.py --model_path ./Pretrained_model/RTM_torsion_continue_v2_epoch7 --vocab_path ./data/torsion_voc.csv
```

### 3. Generate Molecules

```bash
python scripts/gen_triple.py --model_path ./Pretrained_model/RTM_torsion_continue_v2_epoch7 --vocab_path ./data/torsion_voc.csv
```

### 4. Evaluate Docking Results

Use the provided `docking/` scripts to perform molecular docking evaluation.

---

## Results

* Outperformed DualDiff in **over 75%** of dual-target benchmarks.
* Achieved **30× faster** conformer generation.
* Demonstrated superior performance in **GSK3β–JNK3** dual-target tasks, surpassing AIxFuse.
* Successfully extended to **triple-target** molecule generation.

---

## Visualization Example

*(You can insert visual results here, for example, UMAP plots or molecule structures)*

---

## Citation

If you use LaMGen in your research, please cite:

```bibtex
@article{LaMGen2025,
  title={LaMGen: A Universal Multi-Target Molecular Generation Framework Based on Large Language Models},
  author={Your authors...},
  journal={Your journal...},
  year={2025}
}
```

---

## License

This project is licensed under the terms of the MIT license.

---

## Contact

For questions or collaborations, please contact:
**Cholin** – [GitHub](https://github.com/cholin01)
