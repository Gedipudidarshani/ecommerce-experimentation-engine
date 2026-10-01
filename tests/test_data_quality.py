import os
import pandas as pd
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_features.csv")


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(DATA_PATH), f"File not found at {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


def test_grain_uniqueness(df):
    """Ensure grain integrity: user_id must be strictly unique."""
    assert df["user_id"].is_unique, "Cartesian fan-out detected: duplicate user_id found!"


def test_non_negative_financials(df):
    """Ensure financial integrity: revenue and costs cannot be negative."""
    assert (df["gross_revenue"] >= 0).all(), "Negative revenue values found!"
    assert (df["cogs"] >= 0).all(), "Negative COGS values found!"
    assert (df["outbound_shipping_cost"] >= 0).all(), "Negative shipping costs found!"
    assert (df["reverse_logistics_cost"] >= 0).all(), "Negative reverse logistics costs found!"


def test_logical_order_consistency(df):
    """Ensure relational logic: returns and breaches cannot exceed shipped items."""
    assert (df["total_returned"] <= df["total_shipped"]).all(), "Returns exceed items shipped!"
    assert (df["sla_breaches"] <= df["total_shipped"]).all(), "SLA breaches exceed items shipped!"


def test_variant_allocation(df):
    """Ensure experiment assignment integrity."""
    variants = set(df["variant"].unique())
    assert variants == {"Control", "Treatment"}, f"Unexpected variants: {variants}"