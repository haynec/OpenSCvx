"""Parser handlers for linear algebra operations.

Handlers: Norm, Sum, Diag, Inv, Transpose
"""

from openscvx.symbolic.expr.expr import Constant
from openscvx.symbolic.expr.linalg import Diag, Inv, Norm, Sum, Transpose
from openscvx.symbolic.parser._registry import function


def _extract_ord(val):
    """Extract a norm order from a value or Constant."""
    if isinstance(val, str):
        return val
    if isinstance(val, Constant) and val.value.ndim == 0:
        v = float(val.value)
        iv = int(v)
        return iv if float(iv) == v else v
    if isinstance(val, (int, float)):
        return val
    raise ValueError(f"Invalid norm ord value: {val!r}")


@function("Norm")
def _parse_norm(args, kwargs):
    if len(args) < 1:
        raise ValueError("Norm() requires at least 1 argument")
    operand = args[0]
    ord_val = kwargs.get("ord", "fro")
    if len(args) > 1:
        ord_val = _extract_ord(args[1])
    elif "ord" in kwargs:
        ord_val = _extract_ord(kwargs["ord"])
    return Norm(operand, ord=ord_val)


@function("Sum")
def _parse_sum(args, kwargs):
    if len(args) != 1:
        raise ValueError("Sum() takes exactly 1 argument")
    return Sum(args[0])


@function("Diag")
def _parse_diag(args, kwargs):
    if len(args) != 1:
        raise ValueError("Diag() takes exactly 1 argument")
    return Diag(args[0])


@function("Inv")
def _parse_inv(args, kwargs):
    if len(args) != 1:
        raise ValueError("Inv() takes exactly 1 argument")
    return Inv(args[0])


@function("Transpose")
def _parse_transpose(args, kwargs):
    if len(args) != 1:
        raise ValueError("Transpose() takes exactly 1 argument")
    return Transpose(args[0])
