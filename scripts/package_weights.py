#!/usr/bin/env python3
"""Package the weights/ folder into a zip archive for GitHub Releases.

Usage:
    python scripts/package_weights.py                # -> weights.zip
    python scripts/package_weights.py -o custom.zip  # -> custom.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DEFAULT_OUTPUT = ROOT_DIR / "weights.zip"


def package_weights(output_path: Path) -> None:
    """Recursively zip the weights/ directory, preserving its folder structure."""
    if not WEIGHTS_DIR.is_dir():
        print(f"Error: weights directory not found at {WEIGHTS_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(p for p in WEIGHTS_DIR.rglob("*") if p.is_file())
    if not files:
        print("Error: weights directory is empty, nothing to package.", file=sys.stderr)
        sys.exit(1)

    print(f"Packaging {len(files)} file(s) from {WEIGHTS_DIR} ...")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filepath in files:
            arcname = filepath.relative_to(ROOT_DIR)
            zf.write(filepath, arcname)
            print(f"  + {arcname}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! Created {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zip the weights/ folder for GitHub Releases upload.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output zip path (default: {DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args()
    package_weights(args.output)


if __name__ == "__main__":
    main()
