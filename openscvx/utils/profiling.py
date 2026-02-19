from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import cProfile


def profiling_start(profiling_enabled: bool) -> "Optional[cProfile.Profile]":
    """Start profiling if enabled.

    Args:
        profiling_enabled: Whether to enable profiling.

    Returns:
        Profile object if enabled, None otherwise.
    """
    if profiling_enabled:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        return pr
    return None


def profiling_end(pr: "Optional[cProfile.Profile]", identifier: str):
    """Stop profiling and save results with timestamp.

    Args:
        pr: Profile object from profiling_start, or None.
        identifier: Identifier for the profiling session (e.g., "solve", "initialize").
    """
    if pr is not None:
        import os

        pr.disable()
        # Save results so it can be visualized with snakeviz
        os.makedirs("profiling", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pr.dump_stats(f"profiling/{timestamp}_{identifier}.prof")
