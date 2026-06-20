# SSH2.0 (command-line) — antibody hydrophobic-interaction predictor

A cross-platform rebuild of the **SSH2.0 / AB-Amy** tool
(`http://i.uestc.edu.cn/AB-Amy/index.html`), which predicts whether a
monoclonal antibody is prone to **hydrophobic interaction** from its
heavy- and light-chain sequences.

The original was a Windows-only GUI that shelled out to bundled
`svm-scale.exe` / `svm-predict.exe`. This version reimplements the entire
pipeline in **pure Python (standard library only — no numpy, no native
binaries)**, so it runs on macOS, Linux, and Windows, and can be frozen into a
single self-contained executable.

The numerical SVM output was validated to be **bit-for-bit identical** to the
reference LIBSVM implementation on the bundled example data.

## What it computes

For each antibody it:

1. Concatenates the heavy chain and light chain (HC + LC) into one sequence.
2. Builds the **CKSAAGP** encoding (composition of k-spaced amino-acid group
   pairs, gap = 5 → 150 features).
3. Runs **three** RBF support-vector machines, each on its own selected feature
   subset, scaling inputs with that model's range file.
4. Takes a **majority vote** of the three models and reports a confidence
   probability.

Output columns: `No  Name  Length  Probability  Result`
- `Result = 1` → predicted **positive** (prone to hydrophobic interaction)
- `Result = 0` → predicted **negative**
- `Probability` → model confidence in the reported call

## Quick start (no build needed)

You need Python 3.8+ (already on macOS). From this folder:

```bash
# built-in example
python3 ssh2.py --example

# your own paired heavy/light chain FASTA files (records paired by name)
python3 ssh2.py --hc heavy.fasta --lc light.fasta -o results.tsv

# a single FASTA where each record is already one antibody
python3 ssh2.py --fasta combined.fasta
```

## Build a standalone macOS executable

Run this **on your Mac** (PyInstaller cannot cross-compile):

```bash
chmod +x build_macos.sh
./build_macos.sh
```

This produces `./dist/ssh2`, a single file you can copy anywhere and run with
no Python installed:

```bash
./dist/ssh2 --example
./dist/ssh2 --hc heavy.fasta --lc light.fasta -o results.tsv
```

> On Apple Silicon the binary is native arm64. macOS Gatekeeper may warn that
> it's from an unidentified developer; allow it under
> System Settings → Privacy & Security, or run `xattr -dr com.apple.quarantine ./dist/ssh2`.

## FASTA format

Standard FASTA. When using `--hc`/`--lc`, the heavy and light files are paired
by record name:

```
>antibody_1
QVQLQQSGGEL...        (heavy chain)
```
```
>antibody_1
DIQMTQSPSSL...        (light chain)
```

Any character outside the 20 standard amino acids is replaced with `-` and
ignored, matching the original tool.

## Files

- `ssh2.py` — command-line interface
- `ssh2_core.py` — encoding + scaling + SVM prediction (pure Python)
- `data/` — the three trained models, range files, feature lists, examples
- `build_macos.sh` — produces the standalone executable
