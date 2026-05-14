"""MkDocs hook: download Viser playback recordings from a GitHub Release.

The recordings are large (~90 MB for the Franka pick-and-place) and would bloat clones,
so they live as GitHub Release assets (tag: ``docs-assets-v1``) and are fetched into
``docs/assets/viser-recordings/`` at build time. The embedded Viser client then loads
them same-origin from the docs server, sidestepping the CORS issue with GitHub's
release-download 302 (github.com does not set ``Access-Control-Allow-Origin`` on the
redirect response).

Idempotent: skips files that already exist locally, so contributors only pay the
download cost once per recording.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

RELEASE_URL = (
    "https://github.com/OpenSCvx/OpenSCvx/releases/download/docs-assets-v1"
)
RECORDINGS = ("drone_racing.viser", "franka_fr3v2_pick_place.viser")


def on_pre_build(config, **kwargs) -> None:  # noqa: ARG001
    dest_dir = Path(config.docs_dir) / "assets" / "viser-recordings"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in RECORDINGS:
        dest = dest_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue

        url = f"{RELEASE_URL}/{name}"
        print(f"[mkdocs] Fetching {url} -> {dest}", file=sys.stderr)
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as exc:
            print(
                f"[mkdocs] Failed to download {url}: {exc}. "
                "Landing-page Viser embeds will not play until this file is present.",
                file=sys.stderr,
            )
