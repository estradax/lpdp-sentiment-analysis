#!/usr/bin/env python3
"""Unpack a weights zip archive into the project root.

The zip is expected to contain paths like weights/indobert/...,
so extracting to the project root recreates the weights/ folder.

Usage:
    python scripts/unpack_weights.py weights.zip
    python scripts/unpack_weights.py /path/to/weights.zip --dest /other/project
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def unpack_weights(zip_path: Path, dest: Path) -> None:
    """Extract the weights zip into *dest*, creating a weights/ folder there."""
    if not zip_path.is_file():
        print(f"Error: zip file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    if not zipfile.is_zipfile(zip_path):
        print(f"Error: {zip_path} is not a valid zip file.", file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if not members:
            print("Error: zip archive is empty.", file=sys.stderr)
            sys.exit(1)

        # Verify the archive contains the expected weights/ prefix
        has_weights = any(m.startswith("weights/") for m in members)
        if not has_weights:
            print(
                "Warning: zip does not contain a weights/ directory prefix. "
                "Files will be extracted as-is.",
                file=sys.stderr,
            )

        print(f"Extracting {len(members)} entries from {zip_path} to {dest} ...")
        zf.extractall(dest)

    weights_dir = dest / "weights"
    if weights_dir.is_dir():
        file_count = sum(1 for _ in weights_dir.rglob("*") if _.is_file())
        print(f"\nDone! weights/ directory restored with {file_count} file(s).")
    else:
        print("\nDone! Extraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack a weights zip archive into the project root.",
    )
    parser.add_argument(
        "zip_file",
        type=Path,
        help="Path to the weights zip file.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT_DIR,
        help=f"Destination directory (default: project root {ROOT_DIR})",
    )
    args = parser.parse_args()
    unpack_weights(args.zip_file, args.dest)


if __name__ == "__main__":
    main()
