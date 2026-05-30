"""Shared utilities for the integrations subpackage."""


def _resolve_slice(arg, name: str) -> slice:
    """Accept either a State/Control or a slice and return the slice.

    Used inside BYOF dynamics factories so they can be constructed before
    ``Problem`` preprocessing assigns ``.slice`` to State/Control objects.
    The resolved slice is cached by the caller on first invocation.

    Args:
        arg: A ``State`` / ``Control`` object (reads ``.slice`` lazily) or
            a plain ``slice``.
        name: Human-readable name for the argument, used in error messages.

    Returns:
        The resolved ``slice`` into the unified ``x`` or ``u`` vector.

    Raises:
        ValueError: If ``arg`` has a ``.slice`` attribute but it is ``None``
            (preprocessing has not yet run).
        TypeError: If ``arg`` is neither a State/Control nor a ``slice``.
    """
    if hasattr(arg, "slice"):
        sl = arg.slice
        if sl is None:
            raise ValueError(
                f"{name} has no .slice yet — pass it after Problem construction has called "
                "preprocessing, or pass an explicit slice."
            )
        return sl
    if isinstance(arg, slice):
        return arg
    raise TypeError(f"{name} must be a State/Control or slice, got {type(arg).__name__}")
