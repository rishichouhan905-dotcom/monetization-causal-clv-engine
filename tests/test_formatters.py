"""Tests for numerical and currency formatting helpers."""

import pytest
from clv_causal.app.formatters import fmt_ci, fmt_num, fmt_pp, fmt_usd


def test_fmt_usd() -> None:
    assert fmt_usd(-3.26) == "−$3.26"
    assert fmt_usd(4.48, plus_sign=True) == "+$4.48"
    assert fmt_usd(4.48, plus_sign=False) == "$4.48"
    assert fmt_usd(0.0) == "$0.00"
    assert fmt_usd(-0.0001) == "$0.00"
    assert fmt_usd(12500.6, decimals=0) == "$12,501"


def test_fmt_pp() -> None:
    assert fmt_pp(-3.72) == "−3.72 pp"
    assert fmt_pp(1.86, plus_sign=True) == "+1.86 pp"
    assert fmt_pp(0.50, plus_sign=False) == "0.50 pp"
    assert fmt_pp(-1.86, unit=False) == "−1.86"
    assert fmt_pp(0.0) == "0.00 pp"


def test_fmt_num() -> None:
    assert fmt_num(0.0087, plus_sign=True) == "+0.0087"
    assert fmt_num(-0.0081) == "−0.0081"
    assert fmt_num(0.0) == "0.0000"


def test_fmt_ci() -> None:
    assert fmt_ci(2.13, 8.88) == "[+2.13, +8.88]"
    assert fmt_ci(-17.27, 10.75, is_usd=True) == "[−$17.27, +$10.75]"
    assert fmt_ci(-3.72, -0.05, is_pp=True) == "[−3.72 pp, −0.05 pp]"
    assert fmt_ci(-0.0098, 0.0361, decimals=4) == "[−0.0098, +0.0361]"
