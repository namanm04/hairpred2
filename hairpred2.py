#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import gemmi

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from scipy.spatial.distance import cdist


AA_PROPERTIES = {
    "A": {"pI": 6.00, "pKa1": 2.34, "pKa2": 9.69, "hydrophobicity": -0.5, "steric": 0.52, "eiip": 0.0373},
    "R": {"pI": 10.76, "pKa1": 2.17, "pKa2": 9.04, "hydrophobicity": 3.0, "steric": 0.68, "eiip": 0.0959},
    "N": {"pI": 5.41, "pKa1": 2.02, "pKa2": 8.80, "hydrophobicity": 0.2, "steric": 0.76, "eiip": 0.0036},
    "D": {"pI": 2.77, "pKa1": 1.88, "pKa2": 9.60, "hydrophobicity": 3.0, "steric": 0.76, "eiip": 0.1263},
    "C": {"pI": 5.05, "pKa1": 1.96, "pKa2": 8.18, "hydrophobicity": -1.0, "steric": 0.62, "eiip": 0.0829},
    "Q": {"pI": 5.65, "pKa1": 2.17, "pKa2": 9.13, "hydrophobicity": 0.2, "steric": 0.68, "eiip": 0.0761},
    "E": {"pI": 3.22, "pKa1": 2.19, "pKa2": 9.67, "hydrophobicity": 3.0, "steric": 0.68, "eiip": 0.0058},
    "G": {"pI": 5.97, "pKa1": 2.34, "pKa2": 9.60, "hydrophobicity": 0.0, "steric": 0.00, "eiip": 0.0050},
    "H": {"pI": 7.59, "pKa1": 1.82, "pKa2": 9.17, "hydrophobicity": -0.5, "steric": 0.70, "eiip": 0.0242},
    "I": {"pI": 6.02, "pKa1": 2.36, "pKa2": 9.60, "hydrophobicity": -1.8, "steric": 1.02, "eiip": 0.0000},
    "L": {"pI": 5.98, "pKa1": 2.36, "pKa2": 9.60, "hydrophobicity": -1.8, "steric": 0.98, "eiip": 0.0000},
    "K": {"pI": 9.74, "pKa1": 2.18, "pKa2": 8.95, "hydrophobicity": 3.0, "steric": 0.68, "eiip": 0.0371},
    "M": {"pI": 5.74, "pKa1": 2.28, "pKa2": 9.21, "hydrophobicity": 0.2, "steric": 0.76, "eiip": 0.0823},
    "F": {"pI": 5.48, "pKa1": 1.83, "pKa2": 9.13, "hydrophobicity": -2.5, "steric": 0.70, "eiip": 0.0946},
    "P": {"pI": 6.30, "pKa1": 1.99, "pKa2": 10.60, "hydrophobicity": 0.0, "steric": 0.36, "eiip": 0.0198},
    "S": {"pI": 5.68, "pKa1": 2.21, "pKa2": 9.15, "hydrophobicity": 0.3, "steric": 0.53, "eiip": 0.0829},
    "T": {"pI": 5.60, "pKa1": 2.09, "pKa2": 9.10, "hydrophobicity": -0.4, "steric": 0.50, "eiip": 0.0941},
    "W": {"pI": 5.89, "pKa1": 2.83, "pKa2": 9.39, "hydrophobicity": -3.4, "steric": 0.70, "eiip": 0.0548},
    "Y": {"pI": 5.66, "pKa1": 2.20, "pKa2": 9.11, "hydrophobicity": -2.3, "steric": 0.70, "eiip": 0.0516},
    "V": {"pI": 5.96, "pKa1": 2.32, "pKa2": 9.62, "hydrophobicity": -1.5, "steric": 0.76, "eiip": 0.0057},
}

WINDOW_SIZE = 7
INTERACTION_THRESHOLD = 0.5
PATCH_DISTANCE_CUTOFF = 10.0

MODEL_FILENAME = "best_model_random_forest.pkl"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)


def validate_pdb(pdb_file):
    if not os.path.isfile(pdb_file):
        sys.exit(f"[ERROR] File not found: {pdb_file}")

    with open(pdb_file) as f:
        if not any(line.startswith(("ATOM", "HETATM")) for line in f):
            sys.exit("[ERROR] Invalid PDB file")


def validate_chains(pdb_file, chain_ids):
    chains = set()
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                chains.add(line[21])

    missing = [c for c in chain_ids if c not in chains]
    if missing:
        sys.exit(f"[ERROR] Missing chains {missing}")


def extract_chains(input_pdb, chain_ids, output_pdb):
    st = gemmi.read_structure(input_pdb)
    for model in st:
        for chain in list(model):
            if chain.name not in chain_ids:
                model.remove_chain(chain.name)
    st.write_minimal_pdb(output_pdb)


def calculate_rsa(model, pdb_path):
    dssp = DSSP(model, pdb_path, dssp='mkdssp')
    data = []
    for (chain, rid) in dssp.keys():
        aa = dssp[(chain, rid)][1]
        rsa = dssp[(chain, rid)][3]
        if aa != 'X':
            data.append({"Chain": chain, "Residue": aa, "Residue_ID": rid[1], "RSA": rsa})
    return pd.DataFrame(data)


def extract_ca_coordinates(model):
    coords = []
    for chain in model:
        for res in chain:
            if 'CA' in res:
                x, y, z = res['CA'].coord
                coords.append({"Chain": chain.id, "Residue_ID": res.get_id()[1], "X": x, "Y": y, "Z": z})
    return pd.DataFrame(coords)


def build_residue_dataframe(pdb_file, chain_ids):
    temp = "temp_chain.pdb"
    extract_chains(pdb_file, chain_ids, temp)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antigen", temp)[0]

    rsa = calculate_rsa(structure, temp)
    coords = extract_ca_coordinates(structure)

    return pd.merge(rsa, coords, on=["Chain", "Residue_ID"]), temp


def generate_features(df, window=WINDOW_SIZE):
    n = 2 * window + 1
    X = []

    for i in range(len(df)):
        mat = np.zeros((7, n))
        for offset in range(-window, window + 1):
            j = i + offset
            if 0 <= j < len(df):
                row = df.iloc[j]
                aa = row["Residue"]
                if aa in AA_PROPERTIES:
                    props = AA_PROPERTIES[aa]
                    idx = offset + window
                    mat[:, idx] = [
                        row["RSA"], props["pI"], props["pKa1"],
                        props["pKa2"], props["hydrophobicity"],
                        props["steric"], props["eiip"]
                    ]
        X.append(mat.flatten())
    return np.array(X)


def load_model():
    if not os.path.isfile(MODEL_PATH):
        sys.exit("[ERROR] Model not found")
    return joblib.load(MODEL_PATH)


def predict(model, X, threshold):
    probs = model.predict_proba(X)[:, 1]
    labels = np.where(probs >= threshold, "Interacting", "Non-interacting")
    return probs, labels


def save_csv(df, probs, labels, path, min_rsa=None):
    res = pd.DataFrame({
        "Residue": df["Residue"] + df["Residue_ID"].astype(str),
        "RSA": df["RSA"],
        "Probability": probs,
        "Prediction": labels
    })
    if min_rsa:
        res = res[res["RSA"] >= min_rsa]
    res.to_csv(path, index=False)
    return res


def detect_epitope_patches(df, labels, probs):
    mask = labels == "Interacting"
    df = df[mask].reset_index(drop=True)
    probs = probs[mask]

    if df.empty:
        return []

    coords = df[["X", "Y", "Z"]].values
    dist = cdist(coords, coords)

    visited = [False] * len(df)
    patches = []

    def bfs(i):
        queue = [i]
        visited[i] = True
        patch = []
        while queue:
            j = queue.pop(0)
            patch.append(df.iloc[j]["Residue"] + str(df.iloc[j]["Residue_ID"]))
            for k in range(len(df)):
                if not visited[k] and dist[j][k] < PATCH_DISTANCE_CUTOFF:
                    visited[k] = True
                    queue.append(k)
        return patch

    for i in range(len(df)):
        if not visited[i]:
            p = bfs(i)
            if len(p) > 1:
                patches.append(p)

    return patches

def save_summary(df, probs, labels, pdb_file, chains, threshold, output_path):
    total = len(labels)
    inter = np.sum(labels == "Interacting")
    non = total - inter
    avg = np.mean(probs)

    top = pd.DataFrame({
        "Residue": df["Residue"] + df["Residue_ID"].astype(str),
        "Probability": probs
    }).nlargest(10, "Probability")

    lines = [
        f"PDB: {pdb_file}",
        f"Chains: {', '.join(chains)}",
        f"Threshold: {threshold}",
        "",
        f"Total: {total}",
        f"Interacting: {inter} ({inter/total:.2%})",
        f"Non-interacting: {non} ({non/total:.2%})",
        f"Avg probability: {avg:.4f}",
        "",
        "Top 10 residues:",
        top.to_string(index=False)
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

def save_bfactor_pdb(input_pdb, chains, df, probs, output_path):
    lookup = {
        (row["Chain"], int(row["Residue_ID"])): probs[i]
        for i, row in df.iterrows()
    }

    out = []
    with open(input_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                res = int(line[22:26].strip())

                if chain in chains and (chain, res) in lookup:
                    prob = lookup[(chain, res)]
                    line = line[:60] + f"{prob*100:6.2f}" + line[66:]
            out.append(line)

    with open(output_path, "w") as f:
        f.writelines(out)



def save_pymol_script(bfactor_pdb, df, labels, probs, chains, output_path):
    inter = []
    non = []
    labels_cmd = []

    for i, row in df.iterrows():
        chain = row["Chain"]
        res = int(row["Residue_ID"])
        aa = row["Residue"]
        prob = probs[i]

        sel = f"(chain {chain} and resi {res})"

        if labels[i] == "Interacting":
            inter.append(sel)
            labels_cmd.append(
                f'label (chain {chain} and resi {res} and name CA), "{aa}{res} ({prob:.2f})"'
            )
        else:
            non.append(sel)

    inter_sel = " or ".join(inter) if inter else "none"
    non_sel = " or ".join(non) if non else "none"

    lines = [
        f"load {os.path.abspath(bfactor_pdb)}, antigen",
        "bg_color white",
        "hide everything",
        "show cartoon, antigen",
        "color gray80, antigen",

        f"select interacting, {inter_sel}",
        "color red, interacting",
        "show sticks, interacting",
        "show surface, interacting",
        "set transparency, 0.3, interacting",

        f"select non_interacting, {non_sel}",
        "color blue, non_interacting",
    ]

    lines += labels_cmd

    lines += [
        "set label_size, 14",
        "set label_color, black",
        "set label_font_id, 7",
        "zoom antigen",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_pipeline(pdb_file, chains, prefix, threshold, min_rsa):
    validate_pdb(pdb_file)
    validate_chains(pdb_file, chains)

    df, temp = build_residue_dataframe(pdb_file, chains)
    X = generate_features(df)

    model = load_model()
    probs, labels = predict(model, X, threshold)

    save_csv(df, probs, labels, f"{prefix}.csv", min_rsa)

    # ---- Outputs ----
    summary_path = f"{prefix}_summary.txt"
    bfactor_path = f"{prefix}_bfactor.pdb"
    pymol_path = f"{prefix}.pml"

    save_summary(df, probs, labels, pdb_file, chains, threshold, summary_path)
    save_bfactor_pdb(pdb_file, chains, df, probs, bfactor_path)
    save_pymol_script(bfactor_path, df, labels, probs, chains, pymol_path)
    # -----------------

    patches = detect_epitope_patches(df, labels, probs)

    if os.path.exists(temp):
        os.remove(temp)

    print(f"Done. Results saved with prefix '{prefix}'")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-c", "--chain", required=True)
    p.add_argument("-o", "--output", default="results")
    p.add_argument("-t", "--threshold", type=float, default=INTERACTION_THRESHOLD)
    p.add_argument("--min-rsa", type=float)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chains = [c.strip() for c in args.chain.split(",")]

    run_pipeline(
        args.input,
        chains,
        args.output,
        args.threshold,
        args.min_rsa
    )