"""Smoke tests: the environment is usable and the core data stack works together.

These tests use tiny synthetic data only. They never touch the H&M files.
"""

import sys

import numpy as np
import polars as pl
import pyarrow
import yaml

import hm_forecast


def test_python_version_supported():
    assert (3, 12) <= sys.version_info[:2] < (3, 14)


def test_package_importable_and_versioned():
    assert hm_forecast.__version__ == "0.1.0"


def test_numpy_seeded_generator_is_deterministic():
    a = np.random.default_rng(42).integers(0, 100, size=5)
    b = np.random.default_rng(42).integers(0, 100, size=5)
    assert (a == b).all()


def test_polars_parquet_lazy_roundtrip(tmp_path):
    """Write Parquet with Polars, read it lazily, aggregate: the pipeline pattern we will use."""
    df = pl.DataFrame(
        {
            "t_dat": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "article_id": [101, 101, 102],
        }
    ).with_columns(pl.col("t_dat").str.to_date())
    path = tmp_path / "tx.parquet"
    df.write_parquet(path)

    counts = (
        pl.scan_parquet(path)
        .group_by(["article_id", "t_dat"])
        .agg(pl.len().alias("n_events"))
        .sort(["article_id", "t_dat"])
        .collect()
    )
    assert counts["n_events"].to_list() == [2, 1]
    assert counts["n_events"].sum() == df.height  # conservation check


def test_pyarrow_and_yaml_available():
    assert pyarrow.__version__
    assert yaml.safe_load("seed: 42\nhorizon: 7") == {"seed": 42, "horizon": 7}
