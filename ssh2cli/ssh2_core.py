"""
SSH2.0 core — sequence-based hydrophobic-interaction prediction for monoclonal
antibodies, faithfully reimplemented in pure Python.

Original tool: AB-Amy / SSH2.0 (http://i.uestc.edu.cn/AB-Amy/index.html),
distributed as a Windows GUI bundling LIBSVM's svm-scale.exe / svm-predict.exe.
This module reproduces the exact pipeline without any native binaries:

    1. CKSAAGP encoding (composition of k-spaced amino-acid group pairs, gap=5)
    2. Min/max scaling to [-1, 1] using each model's .range file (libsvm svm-scale)
    3. RBF C-SVC prediction with Platt probability (libsvm svm-predict -b 1)
    4. Majority vote over three models; probability derived as in the original.

No third-party native code; only the Python standard library + math are used.
"""

import os
import math

# ---------------------------------------------------------------------------
# Amino-acid grouping for CKSAAGP. Order matters: it defines the column order
# of the 150-feature encoding and therefore which columns the feature_*.txt
# subsets refer to. This mirrors the original codes/CKSAAGP.py exactly.
# ---------------------------------------------------------------------------
GROUP = {
    'alphaticr':       'GAVLMI',
    'aromatic':        'FYW',
    'postivecharger':  'KRH',
    'negativecharger': 'DE',
    'uncharger':       'STCPNQ',
}
AA = 'ARNDCQEGHILKMFPSTWYV'
GAP = 5

_GROUP_KEYS = list(GROUP.keys())
_AA_TO_GROUP = {aa: key for key, aas in GROUP.items() for aa in aas}
_GPAIR_INDEX = [f'{k1}.{k2}' for k1 in _GROUP_KEYS for k2 in _GROUP_KEYS]


def cksaagp_header():
    """Column names of the full CKSAAGP encoding, in order."""
    header = []
    for g in range(GAP + 1):
        for p in _GPAIR_INDEX:
            header.append(f'{p}.gap{g}')
    return header


def cksaagp_encode(sequence):
    """Return {feature_name: value} for one sequence (gaps stripped)."""
    sequence = sequence.replace('-', '')
    header = cksaagp_header()
    values = []
    for g in range(GAP + 1):
        gpair = {p: 0 for p in _GPAIR_INDEX}
        total = 0
        for p1 in range(len(sequence)):
            p2 = p1 + g + 1
            if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                key = _AA_TO_GROUP[sequence[p1]] + '.' + _AA_TO_GROUP[sequence[p2]]
                gpair[key] += 1
                total += 1
        if total == 0:
            values.extend(0 for _ in _GPAIR_INDEX)
        else:
            values.extend(gpair[p] / total for p in _GPAIR_INDEX)
    return dict(zip(header, values))


# ---------------------------------------------------------------------------
# FASTA reading (port of codes/readFasta.py — same illegal-char handling).
# ---------------------------------------------------------------------------
import re

_ALLOWED = re.compile('[^ARNDCQEGHILKMFPSTWYV-]')


def read_fasta(path):
    """Parse a FASTA file -> list of (name, sequence). Returns [] if no '>'.""" 
    with open(path) as f:
        records = f.read()
    if '>' not in records:
        raise ValueError(f'"{path}" does not look like FASTA (no ">" found).')
    out = []
    for block in records.split('>')[1:]:
        lines = block.split('\n')
        name = lines[0].split()[0]
        seq = _ALLOWED.sub('-', ''.join(lines[1:]).upper())
        out.append((name, seq))
    return out


def merge_chains(hc_records, lc_records):
    """Pair heavy and light chains by name and concatenate HC+LC.

    Matches the original mergeData: for every name present in either input,
    the merged sequence is HC+LC when both exist, otherwise whichever exists.
    Order follows first-seen heavy-chain names, then any light-only names.
    """
    hc = dict(hc_records)
    lc = dict(lc_records)
    merged = []
    seen = set()
    for name, _ in hc_records:
        if name in seen:
            continue
        seen.add(name)
        merged.append((name, hc.get(name, '') + lc.get(name, '')))
    for name, _ in lc_records:
        if name in seen:
            continue
        seen.add(name)
        merged.append((name, lc.get(name, '')))
    return merged


# ---------------------------------------------------------------------------
# libsvm svm-scale (-r range): scale each feature to [lower, upper].
# Reproduces the logic of libsvm/svm-scale.c output().
# ---------------------------------------------------------------------------
def load_range(path):
    """Return (lower, upper, {index: (fmin, fmax)}) from a libsvm .range file."""
    with open(path) as f:
        lines = [ln.rstrip('\r\n') for ln in f if ln.strip() != '']
    assert lines[0] == 'x', f'Unexpected range header: {lines[0]!r}'
    lower, upper = (float(x) for x in lines[1].split())
    ranges = {}
    for ln in lines[2:]:
        idx, fmin, fmax = ln.split()
        ranges[int(idx)] = (float(fmin), float(fmax))
    return lower, upper, ranges


def scale_vector(values, lower, upper, ranges):
    """Scale a dense 1-indexed feature vector. Single-valued features (fmax==fmin)
    are dropped (treated as 0), exactly as svm-scale does."""
    scaled = [0.0] * len(values)
    for i, v in enumerate(values, start=1):
        fmin, fmax = ranges[i]
        if fmax == fmin:
            scaled[i - 1] = 0.0
        elif v == fmin:
            scaled[i - 1] = lower
        elif v == fmax:
            scaled[i - 1] = upper
        else:
            scaled[i - 1] = lower + (upper - lower) * (v - fmin) / (fmax - fmin)
    return scaled


# ---------------------------------------------------------------------------
# libsvm model loading + RBF C-SVC prediction with probability (svm-predict -b 1).
# Reproduces svm_predict_values + sigmoid_predict for the binary case.
# ---------------------------------------------------------------------------
class SvmModel:
    def __init__(self, path):
        self.gamma = None
        self.rho = None
        self.label = None        # e.g. [1, 0]
        self.probA = None
        self.probB = None
        self.n_features = 0
        self.sv_coef = []        # one coefficient per SV (binary case)
        self.svs = []            # list of dense feature vectors
        self._load(path)

    def _load(self, path):
        with open(path) as f:
            lines = [ln.rstrip('\r\n') for ln in f]
        i = 0
        n = len(lines)
        while i < n:
            ln = lines[i].strip()
            if ln == 'SV':
                i += 1
                break
            parts = ln.split()
            key = parts[0]
            if key == 'gamma':
                self.gamma = float(parts[1])
            elif key == 'rho':
                self.rho = float(parts[1])
            elif key == 'label':
                self.label = [int(x) for x in parts[1:]]
            elif key == 'probA':
                self.probA = float(parts[1])
            elif key == 'probB':
                self.probB = float(parts[1])
            i += 1
        # Support vectors: "<coef> idx:val idx:val ..."
        max_idx = 0
        raw = []
        for ln in lines[i:]:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            coef = float(toks[0])
            feats = {}
            for t in toks[1:]:
                idx, val = t.split(':')
                idx = int(idx)
                feats[idx] = float(val)
                if idx > max_idx:
                    max_idx = idx
            raw.append((coef, feats))
        self.n_features = max_idx
        for coef, feats in raw:
            vec = [0.0] * max_idx
            for idx, val in feats.items():
                vec[idx - 1] = val
            self.sv_coef.append(coef)
            self.svs.append(vec)

    def decision_value(self, x):
        """Binary C-SVC decision value: sum_i coef_i * K(sv_i, x) - rho."""
        g = self.gamma
        total = 0.0
        for coef, sv in zip(self.sv_coef, self.svs):
            ss = 0.0
            for a, b in zip(sv, x):
                d = a - b
                ss += d * d
            total += coef * math.exp(-g * ss)
        return total - self.rho

    def predict_proba(self, x):
        """Return (predicted_label, P(label[0]), P(label[1])).

        Mirrors libsvm: sigmoid_predict on the decision value gives P(label[0]);
        the predicted label is label[0] iff that probability >= 0.5.
        """
        dec = self.decision_value(x)
        fApB = dec * self.probA + self.probB
        if fApB >= 0:
            p0 = math.exp(-fApB) / (1.0 + math.exp(-fApB))
        else:
            p0 = 1.0 / (1.0 + math.exp(fApB))
        p1 = 1.0 - p0
        pred = self.label[0] if p0 >= p1 else self.label[1]
        return pred, p0, p1


# ---------------------------------------------------------------------------
# Top-level pipeline.
# ---------------------------------------------------------------------------
def _load_feature_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() != '']


def predict(merged_records, data_dir):
    """Run the full three-model pipeline on merged (name, sequence) records.

    Returns a list of dicts: {No, Name, Length, Probability, Result}.
    Result 1 = positive (hydrophobic interaction), 0 = negative.
    """
    feature_lists = [
        _load_feature_list(os.path.join(data_dir, f'feature{i}.txt'))
        for i in (1, 2, 3)
    ]
    ranges = [load_range(os.path.join(data_dir, f'range{i}.range')) for i in (1, 2, 3)]
    models = [SvmModel(os.path.join(data_dir, f'model{i}.model')) for i in (1, 2, 3)]

    results = []
    for n, (name, seq) in enumerate(merged_records, start=1):
        enc = cksaagp_encode(seq)
        votes = 0
        prob_class1 = []
        for feats, (lower, upper, rng), model in zip(feature_lists, ranges, models):
            raw_vec = [enc[col] for col in feats]
            scaled = scale_vector(raw_vec, lower, upper, rng)
            # predict_proba returns (label, P(label[0]), P(label[1])).
            # The models are trained with label order "1 0", so P(label[0]) is
            # the probability of the positive class (class 1).
            pred, p_pos, _ = model.predict_proba(scaled)
            votes += int(pred)            # label 1 == positive vote
            prob_class1.append(p_pos)

        result = 1 if votes >= 2 else 0
        # Probability: average of the agreeing models' P(class 1), as in original.
        if votes > 1:
            s = sum(p for p in prob_class1 if p >= 0.5)
            prob = s / votes
        else:
            s = sum(p for p in prob_class1 if p < 0.5)
            prob = s / (3 - votes)
        results.append({
            'No': n,
            'Name': name,
            'Length': len(seq),
            'Probability': round(prob, 5),
            'Result': result,
        })
    return results
