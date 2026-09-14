#!/usr/bin/env python3
"""Verify downloaded checkpoints against ``results/checkpoints.json``.

Equivalent to ``deepseanet verify``, kept as a standalone script so it runs
without installing the package.

Usage::

    python tools/verify_checkpoints.py [--manifest PATH] [--root DIR]

Exit code is non-zero if any file is missing or has the wrong checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def md5(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through MD5 so large checkpoints do not blow up memory."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/checkpoints.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.root)
    failures = 0

    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists():
            print(f"MISSING   {entry['path']}")
            failures += 1
            continue
        digest = md5(path)
        note = entry.get("note", "")
        if digest == entry["md5"]:
            print(f"OK        {entry['path']}  {note}")
        else:
            print(f"MISMATCH  {entry['path']}")
            print(f"          expected {entry['md5']}")
            print(f"          got      {digest}")
            failures += 1

    print(f"\n{len(manifest['files']) - failures}/{len(manifest['files'])} verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
