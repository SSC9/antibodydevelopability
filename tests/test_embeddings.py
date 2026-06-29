"""Embedding-reader tests — skipped if the cached embeddings aren't present locally."""
import pytest

from abdev.data.embeddings import EMB_DIR, align_xy, load_features
from abdev.data.gdpa3 import load_gdpa3

pytestmark = pytest.mark.skipif(
    not (EMB_DIR / "esm2-8m" / "gdpa1").exists(), reason="embeddings not present locally"
)


def test_esm2_concat_dims():
    ids, X = load_features("esm2-8m", "gdpa1", pool="mean")
    assert X.shape[1] == 640 and X.shape[0] == len(ids) > 200  # 320 (H) + 320 (L)


def test_ablang2_paired_default():
    ids, X = load_features("ablang2", "gdpa1", pool="mean")
    assert X.shape[1] == 480  # native paired vector (chain 'P')


def test_align_xy_matches_labels():
    ids, X = load_features("esm2-650m", "gdpa3", pool="cls")
    Xa, y = align_xy(ids, X, load_gdpa3())
    assert Xa.shape[0] == len(y) and 70 <= len(y) <= 80 and Xa.shape[1] == 2560
