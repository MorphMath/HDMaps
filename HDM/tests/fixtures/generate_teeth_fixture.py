"""Regenerate the teeth end-to-end fixture and its golden outputs.

The teeth bundle is the top-left NxN corner of Tingran Gao's platyrrhine molar
data (consolidated_data.pkl): a dense block of soft correspondence maps plus the
matching base-distance matrix. Only the author needs to run this; the committed
teeth_bundle.npz / teeth_expected.npz are what the test consumes.

Usage: python generate_teeth_fixture.py [path/to/consolidated_data.pkl]
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from HDM import run_hdm, HDMConfig
from _fixture_common import TEETH_CONFIG, golden_outputs

DEFAULT_PKL = "/home/sofus/hdm/workspace/50_teeth/data/tingran_extracted_data/consolidated_data.pkl"
N_SAMPLES = 5
HERE = Path(__file__).parent


def build_bundle(pkl_path):
    # Trusted local data produced by our own MATLAB/Python pipeline.
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    base_dist = np.ascontiguousarray(data["base_dists"][:N_SAMPLES, :N_SAMPLES], dtype=np.float64)

    raw = data["maps"][:N_SAMPLES, :N_SAMPLES]
    maps = np.empty((N_SAMPLES, N_SAMPLES), dtype=object)
    for i in range(N_SAMPLES):
        for j in range(N_SAMPLES):
            maps[i, j] = sp.csr_matrix(raw[i, j]).astype(np.float64)

    return base_dist, maps


def main():
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
    base_dist, maps = build_bundle(pkl_path)

    np.savez(HERE / "teeth_bundle.npz", base_dist=base_dist, maps=maps)

    result = run_hdm(base_dist, maps, HDMConfig(**TEETH_CONFIG))
    np.savez(HERE / "teeth_expected.npz", **golden_outputs(result))

    print(f"teeth fixture: {N_SAMPLES} samples, "
          f"{sum(maps[i, j].nnz for i in range(N_SAMPLES) for j in range(N_SAMPLES))} total nnz")
    print("eigvals:", np.round(result.eigvals, 6))


if __name__ == "__main__":
    main()
