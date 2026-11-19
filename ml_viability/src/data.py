"""Load model data, clean labels, define feature sets, and make the single random 80/10/10 split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "cleaned" / "model_data_with_ref.csv"

TARGET = "Cell viability"
GROUP = "ref"

NUMERIC = ["Concentration(μg/mL)", "Diameter(nm)", "Exposure time"]  # all numeric features are log10-transformed

PARTICLE_CAT = ["Core", "Shape", "Surface modification", "Plastic source"]
CELL_CAT = ["Cell source species", "Cell anatomical type", "Cell-tissue-organ-origin", "Cell origin"]
EXTRA_CAT = ["Cell line", "Assay type"]

# name -> (numeric columns, categorical columns)
FEATURE_SETS = {
    # Full sets (nested CV)
    "C": (NUMERIC, PARTICLE_CAT + CELL_CAT + EXTRA_CAT),  # all 13 features
    "B": (NUMERIC, PARTICLE_CAT + CELL_CAT),              # 11: C without cell line and assay type
    "E4": (NUMERIC, ["Surface modification"]),                      # concentration, diameter, time, surface
    "E5": (NUMERIC, ["Surface modification", "Core"]),              # + polymer type
    "E6": (NUMERIC, ["Surface modification", "Core", "Cell line"]),  # + cell line
}

# Spelling variants that remain after stripping whitespace
SYNONYMS = {
    "Surface modification": {"unmodified": "Unmodified"},
    "Cell source species": {"rat": "Rat"},
    "Cell line": {"Raw264.7": "RAW 264.7", "RAW264.7": "RAW 264.7"},
    "Assay type": {"LIVE/DEAD Bac-Light": "Live/Dead Bac-Light"},
}


def load_data(path=DEFAULT_DATA):
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    if GROUP not in df.columns:
        raise SystemExit(f"{path.name} has no '{GROUP}' column; use data/cleaned/model_data_with_ref.csv")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            df[c] = df[c].replace(SYNONYMS.get(c, {}))
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {path.name}: {n} rows, {n - len(df)} duplicate dropped -> {len(df)} rows, "
          f"{df[GROUP].nunique()} papers")
    return df


def random_split(df, seed, val_frac=0.1, test_frac=0.1, n_bins=10):
    """Single random 80/10/10 split of rows (train / validation / test), stratified on viability."""
    bins = pd.qcut(df[TARGET], n_bins, labels=False, duplicates="drop")
    train, rest = train_test_split(df, test_size=val_frac + test_frac, random_state=seed, stratify=bins)
    val, test = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                                 random_state=seed, stratify=bins.loc[rest.index])
    return train, val, test


def xy(df, feature_set):
    """Inputs of the chosen feature set and the target, cell viability (%)."""
    return df[sum(FEATURE_SETS[feature_set], [])], df[TARGET].to_numpy(dtype=float)
