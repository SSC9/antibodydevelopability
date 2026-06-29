"""Loader tests. GDPa1 ships in-repo; GDPa3 is skipped if the xlsx isn't present locally."""
import pytest

from abdev.config import GDPA3_XLSX
from abdev.data.gdpa1 import load_gdpa1
from abdev.data.gdpa3 import load_gdpa3

SCHEMA = {"id", "vh", "vl", "hic"}


def test_gdpa1_schema():
    df = load_gdpa1()
    assert SCHEMA | {"fold"} <= set(df.columns)
    assert 200 < len(df) <= 246
    assert set(df.fold.unique()) == {0, 1, 2, 3, 4}
    assert df.hic.notna().all() and df.vh.str.len().min() > 50


@pytest.mark.skipif(not GDPA3_XLSX.exists(), reason="GDPa3 xlsx not present locally")
def test_gdpa3_schema():
    df = load_gdpa3()
    assert SCHEMA <= set(df.columns)
    assert 70 <= len(df) <= 80
    assert df.hic.notna().all() and df.vl.str.len().max() < 160  # variable light, not full chain
