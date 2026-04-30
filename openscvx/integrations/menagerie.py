"""Helpers for loading models from the MuJoCo Menagerie submodule.

The MuJoCo Menagerie (google-deepmind/mujoco_menagerie) is an optional
dependency distributed as a git submodule under ``third_party/mujoco_menagerie``
in this repository.  Initialise it once with::

    git submodule update --init third_party/mujoco_menagerie

Then models can be loaded with a single call::

    from openscvx.integrations.menagerie import get_xml_path, load_mjmodel
    mj_model = load_mjmodel("skydio_x2")

Discovery order
---------------
1. ``MUJOCO_MENAGERIE_PATH`` environment variable.
2. ``<repo_root>/third_party/mujoco_menagerie`` (the git submodule).
3. ``~/mujoco_menagerie`` (common manual install location).

XML file selection (``prefer_mjx=True``)
-----------------------------------------
If the model directory contains an ``mjx_*.xml`` file it is preferred over the
standard XML because it has been tuned for MJX (reduced contacts, simplified
geoms, etc.).  Scene XML files (``scene*.xml``) are always excluded unless the
caller explicitly names the file.

Example
-------
Load the Skydio X2 and disable contacts for MJX differentiability::

    import mujoco
    import mujoco.mjx as mjx
    from openscvx.integrations.menagerie import load_mjmodel

    mj_model = load_mjmodel("skydio_x2")
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mjx_model = mjx.put_model(mj_model)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # mujoco imported lazily


def find_menagerie_root() -> Path | None:
    """Return the path to the mujoco_menagerie root directory, or ``None``.

    Checks, in order:
    1. ``MUJOCO_MENAGERIE_PATH`` environment variable.
    2. ``<repo_root>/third_party/mujoco_menagerie`` (git submodule).
    3. ``~/mujoco_menagerie``.
    """
    env = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if env:
        p = Path(env)
        if p.is_dir():
            return p

    # Locate the repo root: this file lives at openscvx/integrations/menagerie.py
    pkg_root = Path(__file__).resolve().parent.parent.parent
    submodule = pkg_root / "third_party" / "mujoco_menagerie"
    if submodule.is_dir() and any(submodule.iterdir()):
        return submodule

    home_clone = Path.home() / "mujoco_menagerie"
    if home_clone.is_dir():
        return home_clone

    return None


def get_model_dir(model_name: str) -> Path:
    """Return the directory for *model_name* inside the menagerie.

    Args:
        model_name: Name of the model directory (e.g. ``"skydio_x2"``).

    Raises:
        FileNotFoundError: If the menagerie root or the model cannot be found.
    """
    root = find_menagerie_root()
    if root is None:
        raise FileNotFoundError(
            "MuJoCo Menagerie not found.\n"
            "Initialise the submodule with:\n"
            "  git submodule update --init third_party/mujoco_menagerie\n"
            "or set the MUJOCO_MENAGERIE_PATH environment variable."
        )
    model_dir = root / model_name
    if not model_dir.is_dir():
        available = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {root}.\n"
            f"Available models: {available}"
        )
    return model_dir


def get_xml_path(model_name: str, prefer_mjx: bool = True) -> Path:
    """Return the path to the best XML file for *model_name*.

    Args:
        model_name: Name of the model directory (e.g. ``"skydio_x2"``).
        prefer_mjx: When ``True`` (default), an ``mjx_*.xml`` file is
            preferred if present, as it has been tuned for the MJX backend.
            Scene XMLs (``scene*.xml``) are excluded in all cases.

    Returns:
        Path to the selected XML file.

    Raises:
        FileNotFoundError: If no XML file can be found.
    """
    model_dir = get_model_dir(model_name)
    xmls = sorted(model_dir.glob("*.xml"))
    non_scene = [f for f in xmls if not f.name.lower().startswith("scene")]

    if prefer_mjx:
        mjx_xmls = [f for f in non_scene if "mjx" in f.stem.lower()]
        if mjx_xmls:
            return mjx_xmls[0]

    if non_scene:
        return non_scene[0]

    if xmls:
        return xmls[0]

    raise FileNotFoundError(f"No XML file found in {model_dir}")


def load_mjmodel(model_name: str, prefer_mjx: bool = True):
    """Load a ``mujoco.MjModel`` for *model_name* from the menagerie.

    Asset paths (meshes, textures) are resolved automatically by MuJoCo
    relative to the XML file, so the full mesh geometry is available without
    any extra configuration.

    Args:
        model_name: Name of the model directory (e.g. ``"skydio_x2"``).
        prefer_mjx: Prefer ``mjx_*.xml`` when present.

    Returns:
        A ``mujoco.MjModel`` instance.

    Raises:
        ImportError: If ``mujoco`` is not installed.
        FileNotFoundError: If the menagerie or model cannot be found.

    Example::

        import mujoco.mjx as mjx
        from openscvx.integrations.menagerie import load_mjmodel

        mj_model = load_mjmodel("skydio_x2")
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mjx_model = mjx.put_model(mj_model)
    """
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "mujoco is required to load menagerie models. "
            "Install with: pip install mujoco"
        ) from exc

    xml_path = get_xml_path(model_name, prefer_mjx=prefer_mjx)
    return mujoco.MjModel.from_xml_path(str(xml_path))


def list_models() -> list[str]:
    """Return the names of all models available in the menagerie.

    Returns an empty list if the menagerie is not found.
    """
    root = find_menagerie_root()
    if root is None:
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def get_asset_dir(model_name: str) -> Path:
    """Return the ``assets/`` sub-directory for *model_name*, if it exists.

    Useful for loading mesh and texture files for custom visualisers.

    Raises:
        FileNotFoundError: If no ``assets/`` directory exists.
    """
    assets = get_model_dir(model_name) / "assets"
    if not assets.is_dir():
        raise FileNotFoundError(f"No assets/ directory in {get_model_dir(model_name)}")
    return assets
