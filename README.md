# HAIRpred2

**HAIRpred2** is a structure-based standalone tool for predicting antibody-interacting residues in antigen structures. It uses Relative Solvent Accessibility (RSA) combined with physicochemical properties in a sliding window framework, fed into a pre-trained Random Forest model.

---

## Requirements

### Python packages
```bash
pip install numpy pandas joblib gemmi biopython scipy
```

### System dependency — mkdssp
HAIRpred2 uses DSSP to compute RSA values. You need `mkdssp` installed on your system.

**Linux / Ubuntu:**
```bash
sudo apt install dssp
```

**Conda (any platform):**
```bash
conda install -c salilab dssp
```

**macOS (Homebrew):**
```bash
brew install dssp
```

---

## Installation

```bash
git clone https://github.com/your-lab/hairpred2
cd hairpred2
pip install numpy pandas joblib gemmi biopython scipy
```

Make sure `best_model_random_forest.pkl` is in the same folder as `hairpred2.py`.

**Folder structure:**
```
hairpred2/
├── hairpred2.py
├── best_model_random_forest.pkl
└── README.md
```

---

## Usage

```bash
# Basic usage
python hairpred2.py -i antigen.pdb -c A

# Custom output prefix
python hairpred2.py -i antigen.pdb -c A -o my_results

# Multiple antigen chains
python hairpred2.py -i antigen.pdb -c A,B

# Filter out buried residues (RSA < 0.05)
python hairpred2.py -i antigen.pdb -c A --min-rsa 0.05

# Custom probability threshold
python hairpred2.py -i antigen.pdb -c A -t 0.4
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `-i` / `--input` | Yes | Input antigen PDB file |
| `-c` / `--chain` | Yes | Chain ID(s) of the antigen (e.g. `A` or `A,B`) |
| `-o` / `--output` | No | Output file prefix (default: `hairpred2_results`) |
| `-t` / `--threshold` | No | Probability threshold for 'Interacting' label (default: `0.5`) |
| `--min-rsa` | No | Minimum RSA to include in output — filters buried residues (e.g. `0.05`) |

---

## Outputs

All output files share the same prefix (set with `-o`):

| File | Description |
|------|-------------|
| `<prefix>.csv` | Per-residue predictions |
| `<prefix>_summary.txt` | Statistics report + top 10 residues |
| `<prefix>_bfactor.pdb` | PDB with probability as B-factor column |
| `<prefix>.pml` | PyMOL coloring script |
| `<prefix>_patches.txt` | Spatially clustered epitope patches |

### Prediction CSV columns

| Column | Description |
|--------|-------------|
| `Residue` | Amino acid + position (e.g. `A45`) |
| `RSA` | Relative Solvent Accessibility (0–1) |
| `Probability` | Predicted probability of being an interacting residue |
| `Prediction` | `Interacting` or `Non-interacting` |

**Example:**
```
Residue,RSA,Probability,Prediction
T22,0.1823,0.3812,Non-interacting
N23,0.6541,0.6247,Interacting
S24,0.7102,0.7103,Interacting
K25,0.0431,0.2891,Non-interacting
```

---

## Visualization

### PyMOL
```bash
# After running the tool, open PyMOL and run:
@hairpred2_results.pml
```
This colors interacting residues **red** and non-interacting residues **blue**, with a surface overlay on the predicted epitope.

### B-factor coloring (PyMOL or ChimeraX)
```
# PyMOL
load hairpred2_results_bfactor.pdb
spectrum b, blue_white_red

# ChimeraX
open hairpred2_results_bfactor.pdb
color bfactor
```

---

## How It Works

1. **Validate** — checks PDB file, ATOM records, and chain availability
2. **Extract** — isolates the antigen chain(s) using gemmi
3. **RSA** — computes per-residue Relative Solvent Accessibility via DSSP
4. **Features** — builds a 105-element feature vector per residue (7 features × 15 window positions)
5. **Predict** — runs the pre-trained Random Forest and outputs interaction probabilities
6. **Epitope patches** — clusters spatially proximal interacting residues (Cα distance < 10 Å)
7. **Save** — writes all 5 output files

### Feature Set (per window position)

| Feature | Description |
|---------|-------------|
| RSA | Relative Solvent Accessibility (from DSSP) |
| pI | Isoelectric point |
| pKa1 | First acid dissociation constant |
| pKa2 | Second acid dissociation constant |
| Hydrophobicity | Hydrophobicity index |
| Steric | Steric parameter |
| EIIP | Electron-ion interaction pseudopotential |

---

## Web Server

HAIRpred2 is also available as a web server:
[https://webs.iiitd.edu.in/raghava/hairpred2/](https://webs.iiitd.edu.in/raghava/hairpred2/)

---

## Citation

If you use HAIRpred2 in your work, please cite our paper *(citation to be added upon publication)*.