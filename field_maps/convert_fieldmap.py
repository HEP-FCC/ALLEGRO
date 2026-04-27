#!/usr/bin/env python3
import argparse
import numpy as np
import uproot
import sys

def detect_scale(filename):
    """Detect length unit from header."""
    scale = 0.0
    with open(filename) as f:
        for line in f:
            if "Length unit:" in line:
                unit = line.split(":")[1].strip()
                if unit == "m":
                    scale = 1.0
                break
    return scale

def load_data(filename):
    """Load numeric columns ignoring comments."""
    return np.loadtxt(filename, comments='%', dtype=np.float64)

def main():
    parser = argparse.ArgumentParser(
        description="Convert COMSOL txt field map to ROOT TTree"
    )
    parser.add_argument("input", help="Input text file")
    parser.add_argument("output", help="Output ROOT file")
    args = parser.parse_args()

    # Detect unit scaling
    scale = detect_scale(args.input)
    if (scale == 0.0):
        print("Error: length units not detected")
        sys.exit(-1)

    # Load data (vectorized)
    data = load_data(args.input)

    # Columns: r, z, Br, Bz
    rho = np.round(data[:, 0] * scale, 2).astype(np.float32)
    z   = np.round(data[:, 1] * scale, 2).astype(np.float32)
    Brho = data[:, 2].astype(np.float32)
    Bz   = data[:, 3].astype(np.float32)

    # Write ROOT file
    with uproot.recreate(args.output) as fout:
        fout["fieldmap"] = {
            "r": rho,
            "z": z,
            "Br": Brho,
            "Bz": Bz,
        }
    print(f"Created {args.output} with {len(rho)} entries")

if __name__ == "__main__":
    main()
