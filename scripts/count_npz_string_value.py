#!/usr/bin/env python3
"""Count values in a string array stored inside an NPZ file.

This intentionally avoids NumPy so Slurm launch scripts can run it with the
host Python before entering the training container.
"""

from __future__ import annotations

import argparse
import ast
import struct
import sys
import zipfile
from pathlib import Path


def _read_npy_string_values(raw: bytes) -> list[str]:
    if not raw.startswith(b"\x93NUMPY"):
        raise ValueError("file inside npz is not an NPY array")
    major = raw[6]
    minor = raw[7]
    if (major, minor) == (1, 0):
        header_len = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    elif (major, minor) in ((2, 0), (3, 0)):
        header_len = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")

    header_text = raw[offset : offset + header_len].decode("latin1")
    header = ast.literal_eval(header_text)
    if bool(header.get("fortran_order")):
        raise ValueError("Fortran-order string arrays are not supported")

    shape = tuple(int(x) for x in header["shape"])
    if len(shape) != 1:
        raise ValueError(f"expected a 1D string array, got shape={shape}")
    count = shape[0]
    descr = str(header["descr"])
    data = raw[offset + header_len :]

    if descr.startswith(("<U", "|U")):
        width = int(descr[2:])
        item_size = width * 4
        encoding = "utf-32le" if descr[0] in ("<", "|") else "utf-32be"
        return [
            data[i * item_size : (i + 1) * item_size]
            .decode(encoding)
            .rstrip("\x00")
            for i in range(count)
        ]
    if descr.startswith("|S"):
        item_size = int(descr[2:])
        return [
            data[i * item_size : (i + 1) * item_size]
            .decode("utf-8", errors="strict")
            .rstrip("\x00")
            for i in range(count)
        ]
    raise ValueError(f"unsupported string dtype {descr!r}")


def count_npz_value(path: Path, array_name: str, value: str) -> int:
    member = array_name if array_name.endswith(".npy") else f"{array_name}.npy"
    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.read(member)
        except KeyError as exc:
            raise ValueError(f"{path} does not contain {member}") from exc
    return sum(1 for item in _read_npy_string_values(raw) if item == value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--array", default="split")
    parser.add_argument("--value", default="holdout")
    args = parser.parse_args(argv)

    try:
        print(count_npz_value(args.npz, args.array, args.value))
    except Exception as exc:
        print(f"failed to count {args.value!r} in {args.npz}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
