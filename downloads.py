"""Publish only a profile's held-out sample for a direct browser download."""

from hashlib import sha256
from pathlib import Path

from utils import ROOT


def sample_download_url(csv_bytes, static_root=None):
    """Return a stable, same-origin URL whose file matches these exact bytes.

    Content-addressed folders keep samples from different profiles separate.
    Uploaded CSVs must never be passed to this function.
    """
    folder = Path(static_root or ROOT / "static")
    relative = Path("downloads") / sha256(csv_bytes).hexdigest() / "test_data.csv"
    destination = folder / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(csv_bytes)
    return "app/static/" + relative.as_posix()
