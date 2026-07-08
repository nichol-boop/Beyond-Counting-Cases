# A Structural Comparison of Climate Judgment Networks in Australia and the United Kingdom

This repository contains data and code for analysing climate law judgments and their citation networks.  
It is structured to be reproducible, collaborative, and easy to navigate.

---

## 📂 Project Structure

```
Climate_Law_Paper_Stats_+_Communities/
├── src/                   # Reusable Python modules (e.g., my_utils.py)
├── data/                  # Raw input datasets (Excel/CSV)
├── outputs/               # Generated figures, tables, GIFs, reports
├── environment.full.yml   # Fully pinned Conda environment (exact rebuild)
└── README.md              # This file
```

---

## 🛠️ Setup Instructions

### 1. Create a Conda Environment
From the project root:

> ```bash
> conda env create -f environment.full.yml
> ```

### 2. Launch Jupyter Lab
```bash
jupyter lab
```

Then open notebooks.

---

## 📊 Typical Workflow

1. **Load data** from `data/` using helper functions in `src/my_utils.py`.
2. **Build citation networks** with `networkx` and visualise with `pyvis`.
3. **Export results** to `outputs/` (figures, interactive HTMLs, GIFs).
4. **Iterate & document** findings inside the notebooks.

---

## 🧩 Helper Library

Reusable code lives in `src/`.  
In a notebook, add this at the top so you can import from `src/`:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path.cwd().parent / "src"))

from my_utils import load_edges_from_excel, build_graph, compute_network_metrics
```

---

## 📜 Notes

- **Do not modify raw data** in `data/`. If cleaning is required, do it in code and save cleaned outputs to `outputs/`.
- Keep notebooks clean: restart and run all cells before committing.
- Add new dependencies to `environment.yml` after installing with `conda install <package>`.

---

## 👥 Contributor
- Nicholas Young 

