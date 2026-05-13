"""MkDocs hook: ensure ``docs/assets/viser-client`` exists for embedded Viser iframes.

**Vendored client (preferred):** If ``docs/assets/viser-client/index.html`` is already present
(e.g. committed in git), this hook does nothing so docs builds and clones use the **same**
static files as the repository—no dependency on a local ``viser`` install.

**Fallback:** If that file is missing, copy from ``viser/client/build`` in the installed
``viser`` package (see ``viser._client_autobuild``), or install Viser and run::

    viser-build-client --out-dir docs/assets/viser-client

Then commit ``docs/assets/viser-client/`` so teammates and CI get working embeds.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def on_pre_build(config, **kwargs) -> None:  # noqa: ARG001
    docs_dir = Path(config.docs_dir)
    dest = docs_dir / "assets" / "viser-client"

    if (dest / "index.html").is_file():
        # Committed bundle: do not overwrite so git state and clone behavior stay predictable.
        return

    try:
        import viser
    except ImportError:
        print(
            "[mkdocs] docs/assets/viser-client/ is missing and viser is not installed. "
            "Run: viser-build-client --out-dir docs/assets/viser-client\n"
            "  then commit that directory so clones do not need Viser to build docs.",
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
