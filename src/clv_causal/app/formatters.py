"""Numerical and Currency Formatting Utilities.

Provides standardized presentation helpers (fmt_usd, fmt_pp, fmt_ci, fmt_num)
guaranteeing consistent signs (e.g. '−$3.26', '+0.0087', '[+2.13, +8.88]'),
omitting markdown backtick code blocks around numbers, and avoiding backslash escapes.
"""

from typing import Union


def fmt_usd(val: Union[float, int], plus_sign: bool = False, decimals: int = 2) -> str:
    """Formats a monetary value in USD with true minus/plus signs and thousands separators.

    Examples:
        fmt_usd(-3.26) -> '−$3.26'
        fmt_usd(4.48, plus_sign=True) -> '+$4.48'
        fmt_usd(12500.5, decimals=0) -> '$12,501'
    """
    if val is None:
        return "$0.00"

    v = float(val)
    # Check for near-zero to avoid negative zero '−$0.00'
    if round(v, decimals) == 0:
        return f"${0.0:,.{decimals}f}" if decimals > 0 else "$0"

    if v < 0:
        return f"−${abs(v):,.{decimals}f}"
    if plus_sign and v > 0:
        return f"+${v:,.{decimals}f}"
    return f"${v:,.{decimals}f}"


def fmt_pp(val: Union[float, int], plus_sign: bool = False, decimals: int = 2, unit: bool = True) -> str:
    """Formats a percentage-point metric with explicit sign and optional 'pp' suffix.

    Examples:
        fmt_pp(-3.72) -> '−3.72 pp'
        fmt_pp(1.86, plus_sign=True) -> '+1.86 pp'
        fmt_pp(0.0087, decimals=4) -> '0.0087 pp'
    """
    if val is None:
        return "0.00 pp" if unit else "0.00"

    v = float(val)
    if round(v, decimals) == 0:
        num_str = f"{0.0:.{decimals}f}"
    elif v < 0:
        num_str = f"−{abs(v):.{decimals}f}"
    elif plus_sign and v > 0:
        num_str = f"+{v:.{decimals}f}"
    else:
        num_str = f"{v:.{decimals}f}"

    return f"{num_str} pp" if unit else num_str


def fmt_num(val: Union[float, int], plus_sign: bool = False, decimals: int = 4) -> str:
    """Formats a generic float with explicit sign (e.g. '+0.0087', '−0.0081').

    Examples:
        fmt_num(0.0087, plus_sign=True) -> '+0.0087'
        fmt_num(-0.0081) -> '−0.0081'
    """
    if val is None:
        return "0.0000"

    v = float(val)
    if round(v, decimals) == 0:
        return f"{0.0:.{decimals}f}"

    if v < 0:
        return f"−{abs(v):.{decimals}f}"
    if plus_sign and v > 0:
        return f"+{v:.{decimals}f}"
    return f"{v:.{decimals}f}"


def fmt_ci(
    lower: Union[float, int],
    upper: Union[float, int],
    is_usd: bool = False,
    is_pp: bool = False,
    decimals: int = 2,
    plus_sign: bool = True,
) -> str:
    """Formats a 95% confidence interval [lower, upper].

    Examples:
        fmt_ci(2.13, 8.88) -> '[+2.13, +8.88]'
        fmt_ci(-17.27, 10.75, is_usd=True) -> '[−$17.27, +$10.75]'
        fmt_ci(-3.72, -0.05, is_pp=True) -> '[−3.72 pp, −0.05 pp]'
    """
    if is_usd:
        l_str = fmt_usd(lower, plus_sign=plus_sign, decimals=decimals)
        u_str = fmt_usd(upper, plus_sign=plus_sign, decimals=decimals)
        return f"[{l_str}, {u_str}]"
    if is_pp:
        l_str = fmt_pp(lower, plus_sign=plus_sign, decimals=decimals, unit=True)
        u_str = fmt_pp(upper, plus_sign=plus_sign, decimals=decimals, unit=True)
        return f"[{l_str}, {u_str}]"

    l_str = fmt_num(lower, plus_sign=plus_sign, decimals=decimals)
    u_str = fmt_num(upper, plus_sign=plus_sign, decimals=decimals)
    return f"[{l_str}, {u_str}]"
