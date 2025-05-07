# 🧩 Chunk 

**Chunk** is a computational framework designed to leverage phenotype information from **bulk transcriptomic data** to uncover robust associations between **cell-cell interactions (CCIs)** and clinical or biological phenotypes in **single-cell** or **spatial transcriptomic** data.

> 🧠 **Core Hypothesis**: Intercellular communication mediated by **ligand–receptor interactions (LRIs)** drives phenotypic heterogeneity across patients.

Guided by diverse phenotypic data types (binary, linear, ordinal, survival), **Chunk** identifies phenotype-associated LRIs from large-scale bulk cohorts and maps them to the single-cell or spatial level to uncover CCI events associated with disease-related phenotypic variation.

![Overview](https://github.com/yyp1999/Chunk/blob/main/Chunk.png)

---

## 🔧 Installation

Chunk is implemented in **Python 3** and can be installed via:

```bash
pip install chunk-py
```

---

## 📘 Usage Guide

Explore how to apply **Chunk** to various datasets and phenotype types:

> 🔍 **Note**: Spatial transcriptomic analysis in Chunk is fundamentally similar to single-cell analysis. For example:
> - To conduct **binary phenotype + spatial analysis**, combine:
>   - The first half of the *binary + single-cell* tutorial
>   - The second half of the *ordinal + spatial* tutorial.

### 📜 Tutorial Links

| Phenotype Type | Dataset Type | Notebook Link |
|----------------|--------------|----------------|
| Binary         | Single-cell  | [🔗 View Tutorial](https://github.com/yyp1999/Chunk/blob/main/tutorial/Binary_phenotype_single_cell_analysis.ipynb) |
| Linear         | Single-cell  | [🔗 View Tutorial](https://github.com/yyp1999/Chunk/blob/main/tutorial/Linear_phenotype_single_cell_analysis.ipynb) |
| Ordinal        | Spatial      | [🔗 View Tutorial](https://github.com/yyp1999/Chunk/blob/main/tutorial/Ordinal_phenotype_spatial_transcriptome_analysis.ipynb) |
| Survival       | Single-cell  | [🔗 View Tutorial](https://github.com/yyp1999/Chunk/blob/main/tutorial/Survival_phenotype_single_cell_analysis.ipynb.ipynb) |

---

## 📦 Toy Dataset

You can download the example dataset for tutorials here: [https://drive.google.com/drive/folders/17RgFhzNYNzFHYUq1Oo0bjhOZDNkfUtff?usp=sharing](https://drive.google.com/drive/folders/17RgFhzNYNzFHYUq1Oo0bjhOZDNkfUtff?usp=sharing)

---

## ✨ Citation

If you use **Chunk** in your research, please consider citing our paper (coming soon).

