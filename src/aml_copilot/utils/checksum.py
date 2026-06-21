from __future__ import annotations


def compute_sha256(path: str) -> str:
    """Return hex SHA-256 digest of the file at path."""
    ...


def record_checksum(path: str, checksums_file: str) -> None:
    """Append or update the checksum entry for path in checksums_file."""
    ...


def verify_checksums(checksums_file: str) -> bool:
    """
    Verify every entry in checksums_file against the file on disk.
    Raises RuntimeError naming the first mismatched file.
    """
    ...
