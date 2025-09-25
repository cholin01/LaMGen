# MTD2025 Dataset

**MTD2025** is a high-quality dataset curated for multi-target molecule generation. It provides molecules with associated protein targets, optimized 3D conformations, and protein embeddings, facilitating the development and benchmarking of multi-target drug design models.

## Dataset Access

The MTD2025 dataset can be accessed via: https://zenodo.org/records/17197079

## File Overview

* **Dual\_targets.csv**
  Contains all dual-target molecules with:

  * Two associated protein targets
  * Corresponding UniProt IDs
  * Ligand torsion angles

* **Triple\_targets.csv**
  Contains all triple-target molecules with:

  * Three associated protein targets
  * Corresponding UniProt IDs
  * Ligand torsion angles

* **LiTEN\_OPT.tar.gz**

  * Compressed archive of all ligand molecules after **LiTEN-FF optimization**
  * Provides low-energy 3D conformations
  * Suitable for extracting molecular 3D information

* **ESMC\_embedding.tar.gz**

  * Compressed archive of protein embeddings for training and test sets
  * Generated with the **ESM-C model**
  * Enables direct use of protein features without raw sequence processing

## Usage Notes

* CSV files link molecules with targets for model training and evaluation.
* Optimized 3D structures can be used for molecular conformation analysis.
* Protein embeddings allow integration of protein information in downstream tasks efficiently.

