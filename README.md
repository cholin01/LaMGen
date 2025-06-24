# 🚀 LaMGen: Multi-Target Molecule Generation Framework

---

## 📖 Introduction

Multi-target drugs hold great promise for treating complex diseases but remain challenging to design due to the need to satisfy multiple binding site constraints while maintaining favorable pharmacokinetics.

Existing AI drug design methods mainly target **single proteins** and struggle to generalize to multi-target scenarios, especially for **3D dual-target molecule generation**.

💡 **LaMGen** is the first **universal multi-target drug design framework based on large language models.**

![Figure1](https://github.com/user-attachments/assets/c3d0ec7a-8e68-4cd5-ac98-9c2197f21baa)

---

## ✨ Key Features

* ✅ **Multi-target support**: Dual-target & triple-target molecule generation.
* 🧬 **ESM-C protein encoding**: Captures multiple protein sequences efficiently.
* 🌀 **Rotation-aware tokens**: Enables spatially valid 3D molecule generation.
* 🔀 **TriCoupleAttention module**: Captures deep multi-level target-ligand interactions.
* ⚡ **Differentiable AI potential energy surfaces**: Guarantees chemical plausibility.
* 🔧 **No task-specific retraining**: Supports arbitrary target combinations.
* 🚀 **High speed**: Up to **30× faster** than traditional methods.
* 🏆 **Superior performance**: Outperforms DualDiff on over 75% of dual-target tasks.

---

## 📂 Project Structure

```text
LaMGen/
├── .idea/               # IDE configuration files (ignore)
├── Pretrained_model/    # Pretrained model checkpoints
├── __pycache__/         # Python cache files
├── data/                # Dataset files
├── docking/             # Docking-related scripts
├── model/               # Model architecture and core modules
├── scripts/             # Training and molecule generation scripts
├── utils/               # Helper functions and utilities
└── LICENSE              # License file
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/cholin01/LaMGen.git
cd LaMGen

# (Recommended) Create a Python environment
conda create -n lamgen python=3.8
conda activate lamgen

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 🛠️ Data Preparation

Place your dual-target datasets under the `data/` directory.

### 🔥 Training

```bash
python scripts/train_triple.py --model_path ./Pretrained_model/RTM_torsion_continue_v2_epoch7 --vocab_path ./data/torsion_voc.csv
```

### 🎯 Molecule Generation

```bash
python scripts/gen_triple.py --model_path ./Pretrained_model/RTM_torsion_continue_v2_epoch7 --vocab_path ./data/torsion_voc.csv
```

### 🧩 Docking Evaluation

Use scripts in the `docking/` folder to perform molecular docking and affinity evaluation.

---

## 📊 Results

* 🎯 Outperformed DualDiff in **over 75%** of dual-target benchmarks.
* ⚡ Achieved **30× faster** conformer generation.
* 🥇 In **GSK3β–JNK3** tasks, surpassed AIxFuse under multiple constraints.
* 🔬 Successfully extended to **triple-target** molecule generation.

---

## 🖼️ Visualization Example

*(Insert UMAP plots, molecule structures, or performance graphs here)*

---

## 📚 Citation

If you use LaMGen in your work, please cite:

```bibtex
@article{LaMGen2025,
  title={LaMGen: A Universal Multi-Target Molecular Generation Framework Based on Large Language Models},
  author={Your authors...},
  journal={Your journal...},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 💬 Contact

For questions or collaborations:

* GitHub: [@cholin01](https://github.com/cholin01)
* Email: *(qlgxx0917@gmail.com)*

---

## 🎉 Acknowledgements

Thanks to the developers of:

* 🧬 ESM protein embeddings
* 💊 PyTorch & Hugging Face Transformers
* 🔍 RDKit & docking tools
