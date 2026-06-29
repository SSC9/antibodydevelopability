#!/usr/bin/env python3
"""
SSH2.0 — antibody hydrophobic-interaction predictor (command-line edition).

Reimplementation of the AB-Amy / SSH2.0 tool that runs natively on macOS,
Linux, and Windows with no bundled binaries.

Examples
--------
Predict from paired heavy- and light-chain FASTA files:

    ssh2 --hc heavy.fasta --lc light.fasta -o results.tsv

Run the built-in example:

    ssh2 --example

Predict from a single FASTA whose sequences are already HC+LC concatenated
(or single-chain), one record per antibody:

    ssh2 --fasta combined.fasta

Output columns: No, Name, Length, Probability, Result
  Result = 1  -> predicted positive (prone to hydrophobic interaction)
  Result = 0  -> predicted negative
  Probability -> model confidence in the reported call.
"""

import argparse
import os
import sys

# Make the module importable whether run from source or frozen by PyInstaller.
if getattr(sys, 'frozen', False):
    BASE = sys._MEIPASS  # type: ignore[attr-defined]
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import ssh2_core as core  # noqa: E402

DATA_DIR = os.path.join(BASE, 'data')


def _format_table(rows):
    cols = ['No', 'Name', 'Length', 'Probability', 'P_positive', 'Result']
    widths = {c: len(c) for c in cols}
    str_rows = []
    for r in rows:
        sr = {
            'No': str(r['No']),
            'Name': str(r['Name']),
            'Length': str(r['Length']),
            'Probability': f"{r['Probability']:.5f}",
            'P_positive': f"{r['P_positive']:.5f}",
            'Result': str(r['Result']),
        }
        for c in cols:
            widths[c] = max(widths[c], len(sr[c]))
        str_rows.append(sr)
    line = lambda sr: '  '.join(sr[c].ljust(widths[c]) for c in cols)
    header_sr = {c: c for c in cols}
    out = [line(header_sr), '  '.join('-' * widths[c] for c in cols)]
    out += [line(sr) for sr in str_rows]
    return '\n'.join(out)


def _write_tsv(rows, path):
    cols = ['No', 'Name', 'Length', 'Probability', 'P_positive', 'Result']
    with open(path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join([
                str(r['No']), str(r['Name']), str(r['Length']),
                f"{r['Probability']:.5f}", f"{r['P_positive']:.5f}", str(r['Result']),
            ]) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='ssh2',
        description='Predict monoclonal-antibody hydrophobic interaction from sequence.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--hc', help='FASTA file of heavy-chain sequences.')
    p.add_argument('--lc', help='FASTA file of light-chain sequences (paired by name with --hc).')
    p.add_argument('--fasta', help='Single FASTA; each record taken as one antibody (HC+LC concatenated or single chain).')
    p.add_argument('--example', action='store_true', help='Run the bundled example dataset.')
    p.add_argument('-o', '--output', help='Write results to this TSV file (also printed to stdout).')
    args = p.parse_args(argv)

    if args.example:
        hc = core.read_fasta(os.path.join(DATA_DIR, 'example_HC.fasta'))
        lc = core.read_fasta(os.path.join(DATA_DIR, 'example_LC.fasta'))
        merged = core.merge_chains(hc, lc)
    elif args.fasta:
        recs = core.read_fasta(args.fasta)
        merged = [(n, s) for n, s in recs]
    elif args.hc or args.lc:
        hc = core.read_fasta(args.hc) if args.hc else []
        lc = core.read_fasta(args.lc) if args.lc else []
        merged = core.merge_chains(hc, lc)
    else:
        p.error('provide --hc/--lc, or --fasta, or --example. See --help.')

    if not merged:
        sys.exit('No sequences found in input.')

    results = core.predict(merged, DATA_DIR)
    print(_format_table(results))
    if args.output:
        _write_tsv(results, args.output)
        print(f'\nResults written to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
