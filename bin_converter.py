#!/usr/bin/env python3
"""
Convert *_bin.txt files to raw binary files.
Reads all files matching *_bin.txt in the current directory,
extracts only '0' and '1' characters, packs them into bytes,
and writes them as raw binary files with a .bin extension.
"""

import glob
import os
import sys


def convert_file(txt_path):
    """Convert a single *_bin.txt file to raw binary."""
    base = os.path.basename(txt_path)
    # Output name: replace _bin.txt with .bin
    if base.endswith("_bin.txt"):
        out_name = base[:-len("_bin.txt")] + ".bin"
    else:
        out_name = base + ".bin"

    with open(txt_path, "r", newline="", encoding="utf-8") as f:
        text = f.read()

    # Keep only binary digits
    bits = "".join(ch for ch in text if ch in "01")

    if len(bits) % 8 != 0:
        print(f"Warning: {base} has {len(bits)} bits (not a multiple of 8). "
              f"Truncating to {len(bits) - (len(bits) % 8)} bits.")
        bits = bits[:len(bits) - (len(bits) % 8)]

    if len(bits) == 0:
        print(f"Skipping {base}: no binary data found.")
        return

    # Pack bits into bytes
    raw = bytearray()
    for i in range(0, len(bits), 8):
        byte = int(bits[i:i+8], 2)
        raw.append(byte)

    with open(out_name, "wb") as f:
        f.write(raw)

    print(f"Converted {base} -> {out_name} ({len(raw)} bytes)")


def main():
    files = glob.glob("./*_bin.txt")
    if not files:
        print("No *_bin.txt files found in the current directory.")
        sys.exit(1)

    for path in sorted(files):
        convert_file(path)


if __name__ == "__main__":
    main()
