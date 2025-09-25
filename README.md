# 🚀 LaMGen: LLM-Based 3D Molecular Generation for Multi-Target Drug Design

---

## 📖 Introduction

Multi-target drugs hold great promise for treating complex diseases, yet their rational design remains highly challenging due to the need to simultaneously satisfy multiple binding-site constraints while ensuring favorable pharmacokinetic properties. 

Existing methodologies predominantly rely on **ligand-based approaches**, which lack sufficient biological context and are often confined to specific target pairs, resulting in **limited generalizability**, particularly for triple-target tasks, which remain largely unaddressed.

💡 **LaMGen** is the first **general-purpose multi-target drug design framework powered by large language models.**

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
* 🏆 **Superior performance**: Outperforms diffusion-based approaches in 15 of the 20 dual-target tasks.

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

### 📂 Dataset

If you want to train the model, the **MTD2025 dataset** can be accessed via: [https://zenodo.org/records/17197079](https://zenodo.org/records/17197079)

The dataset includes:  

- **Dual_targets.csv** — Contains all dual-target molecules  
- **Triple_targets.csv** — Contains all triple-target molecules  
- **LiTEN_OPT.sdf** — All ligand molecules after LiTEN-FF optimization; provides low-energy 3D conformations  
- **ESMC_embedding.tar.gz** — All protein embeddings for training and test sets 

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

# LaMGen Model Checkpoints

The **LaMGen** model checkpoints are publicly available at [Zenodo](https://zenodo.org/records/17198652), including:

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

* 🎯 Outperformed DualDiff in **over 75%** of dual-target benchmarks.
* ⚡ Achieved **30× faster** conformer generation.
* 🥇 In **GSK3β–JNK3** tasks, surpassed AIxFuse under multiple constraints.
* 🔬 Successfully extended to **triple-target** molecule generation.

---

## 📊 Model Performance

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/43190783-87f5-47b4-832f-8fb65f598d06" width="400"><br>
       Comparison of JNK3 and GSK3β inhibitors
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/a000fe60-ec16-41e5-9842-f18b94e20ea8" width="400"><br>
       Representative results for three dual-target systems.
    </td>
  </tr>
</table>


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
