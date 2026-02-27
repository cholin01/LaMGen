# 🚀 LaMGen: LLM-Based 3D Molecular Generation for Multi-Target Drug Design

---

## 📖 Introduction

Multi-target drugs hold great promise for treating complex diseases, yet their rational design remains highly challenging due to the need to simultaneously satisfy multiple binding-site constraints while ensuring favorable pharmacokinetic properties. 

Existing methodologies predominantly rely on **ligand-based approaches**, which lack sufficient biological context and are often confined to specific target pairs, resulting in **limited generalizability**, particularly for triple-target tasks, which remain largely unaddressed.

💡 **LaMGen** is a **general-purpose multi-target drug design framework powered by large language models.**

Built upon the MTD2025 dataset, which comprises over 4,000 protein targets, 100,000 ligands, more than 600,000 quantum-precision 3D conformations, and over 700,000 dual- and triple-target associations, LaMGen enables direct generation of spatially valid 3D molecules by integrating ESM-C protein embeddings with rotation-aware molecular representations. Its design explicitly addresses both data efficiency and scalability, providing a unified solution for dual- and triple-target molecular generation.

<img width="7028" height="8410" alt="Figure1" src="https://github.com/user-attachments/assets/1c567556-22cc-479b-b88c-73365b9de9fe" />



---

## ✨ Key Features

* ✅ **Multi-target support**: Dual-target & triple-target molecule generation.
* 🧬 **ESM-C protein encoding**: Captures multiple protein sequences efficiently.
* 🌀 **Rotation-aware tokens**: Enables spatially valid 3D molecule generation.
* 🔀 **TriCoupleAttention module**: Captures deep multi-level target-ligand interactions.
* ⚡ **Differentiable AI potential energy surfaces**: Guarantees chemical plausibility.
* 🔧 **No task-specific retraining**: Supports arbitrary target combinations.
* 🚀 **High speed**: Up to **30× faster** than traditional methods.
* 🏆 **Superior performance**: Outperforms diffusion-based approaches in 17 of the 20 dual-target tasks.

---

## 📂 Project Structure

```text
LaMGen/
├── .idea/               # IDE configuration files (ignore)
├── ESMC_example/        # Example ESM-C protein embeddings used for dual-target and triple-target molecule generation scripts (gen_dual.py and gen_triple.py)
├── __pycache__/         # Python cache files
├── checkpoint/          # Pretrained and multi-target (dual- & triple-target) model checkpoints
├── data/                # Dataset files
├── docking/             # Docking-related scripts
├── model/               # Model architecture and core modules
├── scripts/             # Training and molecule generation scripts
├── utils/               # Helper functions and utilities
├── LICENSE              # License file
├── README.md            # Overview of LaMGen introduction and usage instructions
├── lamgen_env.yml       # Conda environment file with all dependencies required to run LaMGen
└── requirements.txt     # Python pip dependencies for LaMGen, in case Conda is not used

```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone -b master https://github.com/cholin01/LaMGen.git
cd LaMGen

# (Recommended) Create a Python environment
conda create -n lamgen python=3.8
conda activate lamgen

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 📂 Dataset

If you want to train the model, the **MTD2025 dataset** can be accessed via: [Zenodo](https://zenodo.org/records/18297683)
The dataset includes:  

- **Dual_targets.csv** — Contains all dual-target molecules  
- **Triple_targets.csv** — Contains all triple-target molecules  
- **LiTEN_OPT.sdf** — All ligand molecules after LiTEN-FF optimization; provides low-energy 3D conformations  
- **ESMC_embedding.tar.gz** — All protein embeddings for training and test sets (see Version v1)

### 🛠️ Data Preparation

Place your dual-targes or triple-targets under the `data/` directory.

### 🔥 Training

```bash
# If you want to train a dual-target molecule generation model, run:
python scripts/train_dual.py

# If you want to train a triple-target molecule generation model, run:
python scripts/train_triple.py
```

### 🎯 Molecule Generation

### LaMGen Model Checkpoints

The **LaMGen** model checkpoints are publicly available at [Zenodo](https://zenodo.org/records/18218936), including:

- **Small-molecule pretraining model** trained on the GEOM dataset  
- **Dual-target and triple-target generation models** for multi-target molecule design

```bash
# If you want to generate dual-target molecules, download the corresponding checkpoints and run:
python scripts/gen_dual.py

# If you want to generate triple-target molecules, download the corresponding checkpoints and run
python scripts/gen_triple.py
```

### 🧩 Docking Evaluation

Use scripts in the `docking/` folder to perform molecular docking and affinity evaluation.

---

## 📊 Results

* 🎯 Consistently outperforms diffusion-based baselines across independent dual-target benchmarks.
* ⚡ Achieves substantially faster 3D conformer generation while preserving high structural plausibility.
* 🧪 Demonstrates robust zero-shot generalization in the JNK3–GSK3β case study, with further improvements after fine-tuning.
* 🔬 Retrospective analyses show successful de novo generation of known active molecules for multiple dual-target pairs.
* 🧩 Generates structurally novel candidates with conserved core scaffolds, favorable binding affinities, and diverse conformational profiles.
* 🚀 Naturally extends to triple-target molecular generation, reproducing reference active compounds.

---

## 📊 Model Performance

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/5d261e8e-18e6-40eb-9820-03c16f5d0243" width="400"><br>
       Comparison of JNK3 and GSK3β inhibitors
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/4cd7710e-e492-4e6b-8764-5f9d7466376c" width="400"><br>
       Representative results for three dual-target systems.
    </td>
  </tr>
</table>


---

## 📚 Citation

If you use LaMGen in your work, please cite:

```bibtex
@article{LaMGen2026,
  title={LaMGen: LLM-Based 3D Molecular Generation for Multi-Target Drug Design},
  author={Qun Su, Qiaolin Gou},
  year={2026}
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
