"""Download the two UJIIndoorLoc CSV files from UCI if they are missing, then check them.

    python scripts/download_data.py

The files go to data/raw/UJIndoorLoc/ and must match the SHA-256 checksums in
indoorloc.config. Exit code 0 means both files are present and correct.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # no install needed

from indoorloc.config import DATA_URL, RAW_DIR, SHA256  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    missing = [name for name in SHA256 if not (RAW_DIR / name).exists()]
    if missing:
        print(f"Downloading {DATA_URL}")
        # Why the browser-like User-Agent: the UCI server refuses some scripted downloads
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for member in archive.namelist():
            name = Path(member).name
            if name in missing:
                (RAW_DIR / name).write_bytes(archive.read(member))
    all_ok = True
    for name, expected in SHA256.items():
        path = RAW_DIR / name
        ok = path.exists() and sha256(path) == expected
        print(f"{'OK    ' if ok else 'FAILED'} {path}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
