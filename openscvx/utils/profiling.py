import cProfile
import os
from datetime import datetime
from typing import Callable, Optional


def _create_session():
    """Create a new profiling session. Returns a function that generates .prof paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("profiling", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    counts = {}

    def path_for(identifier: str) -> str:
        count = counts.get(identifier, 0)
        counts[identifier] = count + 1
        suffix = f"-{count}" if count > 0 else ""
        return os.path.join(run_dir, f"{identifier}{suffix}.prof")

    return path_for


def profiling_start(
    profiling_enabled: bool, session: Optional[Callable[[str], str]] = None
) -> Optional[tuple[cProfile.Profile, Callable[[str], str]]]:
    """Start profiling if enabled.

    Args:
        profiling_enabled: Whether to enable profiling.
        session: Existing session to reuse. If None, a new session is created.

    Returns:
        (Profile, session) tuple if enabled, None otherwise.
    """
    if profiling_enabled:
        if session is None:
            session = _create_session()
        pr = cProfile.Profile()
        pr.enable()
        return (pr, session)
    return None


def profiling_end(ctx: Optional[tuple[cProfile.Profile, Callable[[str], str]]], identifier: str):
    """Stop profiling and save results.

    Args:
        ctx: (Profile, session) tuple from profiling_start, or None.
        identifier: Identifier for the profiling stage (e.g., "solve", "initialize").
    """
    if ctx is not None:
        pr, session = ctx
        pr.disable()
        pr.dump_stats(session(identifier))
