"""MkDocs hook: sync Viser's prebuilt web client into ``docs/assets/viser-client``.

The Viser wheel ships ``viser/client/build/`` (see ``viser._client_autobuild``). Copying it
before ``mkdocs build`` makes the home-page iframes resolve without committing the client.

If the bundled build is missing (unusual pip layout), print a hint to run
``viser-build-client --out-dir docs/assets/viser-client``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def on_pre_build(config, **kwargs) -> None:  # noqa: ARG001
    docs_dir = Path(config.docs_dir)
    dest = docs_dir / "assets" / "viser-client"

    try:
        import viser
    except ImportError:
        print(
            "[mkdocs] viser is not installed; skipping viser client sync "
            "(install the project or run viser-build-client --out-dir docs/assets/viser-client).",
            file=sys.stderr,
        )
        return

    src = Path(viser.__file__).resolve().parent / "client" / "build"
    if not (src / "index.html").is_file():
        print(
            f"[mkdocs] Viser client build not found at {src}. "
            "Run: viser-build-client --out-dir docs/assets/viser-client",
            file=sys.stderr,
        )
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    print(f"[mkdocs] Copied Viser web client to {dest}")
