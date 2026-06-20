#!/usr/bin/env bash
# Build a standalone `ssh2` executable for macOS using PyInstaller.
#
# Run this ON YOUR MAC (PyInstaller does not cross-compile; building on Linux
# or Windows produces a binary for that OS, not macOS).
#
# Usage:
#   chmod +x build_macos.sh
#   ./build_macos.sh
#
# Result: ./dist/ssh2   (a single self-contained file you can run anywhere)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "Using interpreter: $($PY --version)"

# Isolated build environment so nothing pollutes your system Python.
$PY -m venv .build-venv
# shellcheck disable=SC1091
source .build-venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install pyinstaller >/dev/null

# The data/ folder (models, ranges, feature lists, examples) must ship inside
# the binary. On macOS/Linux the --add-data separator is ':'.
pyinstaller \
  --onefile \
  --name ssh2 \
  --add-data "data:data" \
  --hidden-import ssh2_core \
  ssh2.py

deactivate
echo
echo "Done. Your executable is at: ./dist/ssh2"
echo "Try it:   ./dist/ssh2 --example"
echo "          ./dist/ssh2 --hc heavy.fasta --lc light.fasta -o results.tsv"
