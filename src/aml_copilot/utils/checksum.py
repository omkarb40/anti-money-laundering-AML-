"""SHA-256 freeze utilities for immutable artifacts.

Checksum entries are stored as **repo-relative** paths so the file is portable
across machines and clone locations.  Entries written by older versions of this
module that used absolute paths are still accepted by the reader
(backwards-compatible via _resolve_key).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Repo root: three levels up from src/aml_copilot/utils/checksum.py
_REPO_ROOT: Path = Path(__file__).parents[3].resolve()


def compute_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_key(path: Path) -> str:
    """Return the checksum-file storage key for path.

    Prefers a repo-relative form (e.g. ``data/fixtures/eval.jsonl``).
    Falls back to the absolute path string if path is outside the repo root.
    """
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_key(key: str) -> Path:
    """Convert a checksum-file key to an absolute Path.

    Handles both repo-relative keys (current format) and legacy absolute paths
    written by older versions of this module.
    """
    p = Path(key)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def append_checksum(path: str | Path, checksum_file: str | Path) -> str:
    """Compute SHA-256 of path and append to checksum_file.

    Raises RuntimeError if an entry for this path already exists (write-once).
    Stores a repo-relative key so the checksum file is portable.
    Returns the hex digest.
    """
    path = Path(path)
    checksum_file = Path(checksum_file)
    digest = compute_sha256(path)
    entry_key = _to_key(path)

    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            parts = line.strip().split("  ", 1)
            if len(parts) == 2 and parts[1] == entry_key:
                raise RuntimeError(
                    f"{path.name} is frozen — checksum already recorded in "
                    f"{checksum_file}. Delete the entry explicitly to rebuild."
                )

    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checksum_file, "a") as fh:
        fh.write(f"{digest}  {entry_key}\n")
    return digest


def verify_checksums(checksum_file: str | Path) -> None:
    """Verify every entry in checksum_file.

    Raises RuntimeError naming the first mismatch or missing file.
    Silently succeeds if checksum_file does not exist (nothing frozen yet).
    Resolves both repo-relative keys (current format) and legacy absolute paths.
    """
    checksum_file = Path(checksum_file)
    if not checksum_file.exists():
        return

    for line in checksum_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        expected_digest, key = parts
        p = _resolve_key(key)
        if not p.exists():
            raise RuntimeError(f"Frozen file missing: {p}")
        actual = compute_sha256(p)
        if actual != expected_digest:
            raise RuntimeError(
                f"Checksum mismatch for {p}\n"
                f"  expected: {expected_digest}\n"
                f"  actual:   {actual}"
            )


# ── CLI: python -m aml_copilot.utils.checksum --verify artifacts/checksums.sha256 ──

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Verify frozen artifact checksums")
    parser.add_argument("--verify", metavar="FILE", help="checksums.sha256 to verify")
    args = parser.parse_args()
    if args.verify:
        try:
            verify_checksums(args.verify)
            print(f"[OK] All checksums verified: {args.verify}")
        except RuntimeError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
