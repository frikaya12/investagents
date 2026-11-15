"""Tests anti-fuite des variables ML et du split : colonnes interdites, coffre-fort, temps."""

import pandas as pd
import pytest

from investagents.ml import features as ft
from investagents.ml import split as sp

POST_PROJECT = [
    "quality_at_entry",
    "bank_performance",
    "quality_of_supervision",
    "me_quality",
    "final_closing_fy",
    "evaluation_fy",
    "evaluation_type",
]


def ieg_rows(n=6, **overrides):
    base = {
        "project_id": [f"P{i:06d}" for i in range(n)],
        "wb_region": ["Africa"] * n,
        "practice_group": ["Human Development"] * n,
        "global_practice": ["Education", ""] * (n // 2),
        "lending_instrument_type": ["IPF"] * n,
        "agreement_type": ["IDA"] * n,
        "country": ["Chad"] * n,
        "approval_fy": [2000 + i for i in range(n)],
        "amount_at_approval_usd": [1e6, None, 0.0, 5e7, 2e8, 3e6][:n],
        "satisfactory": [1, 0] * (n // 2),
        "evaluation_fy": [2010] * n,
        # colonnes post-projet présentes dans la table d'entrée : ne doivent PAS passer
        "quality_at_entry": ["Satisfactory"] * n,
        "bank_performance": ["Satisfactory"] * n,
        "final_closing_fy": [2008] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


@pytest.mark.parametrize("col", POST_PROJECT)
def test_post_project_columns_are_forbidden(col):
    assert ft.is_forbidden(col)
    with pytest.raises(ValueError, match="interdites"):
        ft.assert_no_forbidden(["approval_fy", col])


def test_feature_list_contains_no_forbidden_column():
    assert not [c for c in ft.FEATURES if ft.is_forbidden(c)]


def test_build_features_never_outputs_forbidden_columns():
    x = ft.build_features(ieg_rows())
    assert list(x.columns) == ft.FEATURES
    assert not [c for c in x.columns if ft.is_forbidden(c)]


def test_adding_a_forbidden_column_to_features_fails(monkeypatch):
    """Si quelqu'un ajoute une colonne post-projet à la liste, build_features doit échouer."""
    monkeypatch.setattr(ft, "CATEGORICAL", [*ft.CATEGORICAL, "quality_at_entry"])
    monkeypatch.setattr(ft, "FEATURES", ft.CATEGORICAL + ft.NUMERIC)
    with pytest.raises(ValueError, match="interdites"):
        ft.build_features(ieg_rows())


def test_amount_transform_and_missing_flag():
    x = ft.build_features(ieg_rows())
    assert x["amount_missing"].tolist() == [0, 1, 1, 0, 0, 0]  # None et 0 -> manquant
    assert x["log_amount_at_approval"].iloc[0] == pytest.approx(13.8155, abs=1e-3)
    assert pd.isna(x["global_practice"].iloc[1])  # chaîne vide -> manquant


def test_target_is_failure():
    y = ft.target_failure(ieg_rows())
    assert y.tolist() == [0, 1, 0, 1, 0, 1]


def test_vault_mask():
    df = pd.DataFrame({"evaluation_fy": [2024, 2025, 2027]})
    assert sp.vault_mask(df).tolist() == [False, True, True]


def test_assign_split_is_temporal():
    fy = pd.Series([1990, 2007, 2008, 2011, 2012, 2017, 2018])
    assert sp.assign_split(fy).tolist() == [
        "train",
        "train",
        "valid",
        "valid",
        "test",
        "test",
        "excluded",
    ]


def test_load_modeling_table_excludes_vault(tmp_path):
    ieg = ieg_rows(evaluation_fy=[2010, 2024, 2025, 2026, 2015, 2020])
    ieg.drop(columns=["amount_at_approval_usd"]).to_parquet(tmp_path / "ieg.parquet")
    pd.DataFrame({"project_id": ["P000000"], "amount_at_approval_usd": [1.0]}).to_parquet(
        tmp_path / "am.parquet"
    )
    df = sp.load_modeling_table(str(tmp_path / "ieg.parquet"), str(tmp_path / "am.parquet"))
    assert (df["evaluation_fy"] < 2025).all()
    assert set(df["project_id"]) == {"P000000", "P000001", "P000004", "P000005"}
