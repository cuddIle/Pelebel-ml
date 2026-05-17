"""Convert a sklearn RandomForestClassifier .joblib file to a JSON file
that can be loaded in Unity.

Usage:
    python joblib_to_json.py model/single-letter-models/A.joblib
    python joblib_to_json.py model/single-letter-models/  # converts all .joblib files in directory
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np


def export_forest(input_path: Path, output_path: Path):
    clf = joblib.load(input_path)
    trees = []
    for est in clf.estimators_:
        t = est.tree_
        values = t.value.squeeze()  # [n_nodes, n_classes]
        row_sums = values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs = (values / row_sums).tolist()

        trees.append({
            "feature": t.feature.tolist(),
            "threshold": [round(v, 6) for v in t.threshold.tolist()],
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "value": [[round(p, 6) for p in row] for row in probs],
        })

    with open(output_path, "w") as f:
        json.dump(trees, f, separators=(",", ":"))

    print(f"  {input_path.name} -> {output_path.name}: {len(trees)} trees")


