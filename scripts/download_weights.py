#!/usr/bin/env python3
"""Download pre-trained weights and unpack them into the project root.

Downloads the weights zip from the GitHub release, then delegates to
``scripts/unpack_weights.py`` to extract and restore the ``weights/`` folder.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --dest /other/project
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

WEIGHTS_URL = (
    "https://github.com/estradax/lpdp-sentiment-analysis"
    "/releases/download/v1.0.0/weights.zip"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
UNPACK_SCRIPT = ROOT_DIR / "scripts" / "unpack_weights.py"


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownloading: {pct:3d}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\rDownloading: {mb_down:.1f} MB", end="", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download pre-trained weights and unpack them.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT_DIR,
        help=f"Destination directory for unpacking (default: project root {ROOT_DIR})",
    )
    args = parser.parse_args()

    # Download to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "weights.zip"
        print(f"Downloading weights from:\n  {WEIGHTS_URL}\n")

        try:
            urlretrieve(WEIGHTS_URL, zip_path, reporthook=_reporthook)
        except Exception as exc:
            print(f"\n\nError: failed to download weights: {exc}", file=sys.stderr)
            sys.exit(1)

        print("\n\nDownload complete. Unpacking weights ...\n")

        # Delegate to unpack_weights.py
        cmd = [sys.executable, str(UNPACK_SCRIPT), str(zip_path), "--dest", str(args.dest)]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
