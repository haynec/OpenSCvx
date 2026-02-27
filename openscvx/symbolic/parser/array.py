"""Parser handlers for array manipulation operations.

Handlers: Concat, Stack, Hstack, Vstack, Block
"""

from openscvx.symbolic.expr.array import Block, Concat, Hstack, Stack, Vstack
from openscvx.symbolic.parser._registry import function


@function("Concat")
def _parse_concat(args, kwargs):
    if len(args) < 1:
        raise ValueError("Concat() requires at least 1 argument")
    return Concat(*args)


@function("Stack")
def _parse_stack(args, kwargs):
    if len(args) < 1:
        raise ValueError("Stack() requires at least 1 argument")
    return Stack(list(args))


@function("Hstack")
def _parse_hstack(args, kwargs):
    if len(args) < 1:
        raise ValueError("Hstack() requires at least 1 argument")
    return Hstack(list(args))


@function("Vstack")
def _parse_vstack(args, kwargs):
    if len(args) < 1:
        raise ValueError("Vstack() requires at least 1 argument")
    return Vstack(list(args))


@function("Block")
def _parse_block(args, kwargs):
    if len(args) == 1 and isinstance(args[0], list):
        return Block(args[0])
    return Block(list(args))
