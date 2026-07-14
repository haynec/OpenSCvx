"""Generate the code reference pages and navigation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "openscvx"


def _module_summary(path: Path, dotted: str) -> str:
    """First line of a module's docstring, or a package-qualified fallback.

    The summary becomes the page's ``description`` meta tag so that every module
    in the reference gets a distinct, extractable one-liner instead of inheriting
    the site-wide tagline. ``dotted`` is the import path relative to ``openscvx``
    (e.g. ``symbolic.lowerers.jax.state``).
    """
    try:
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, OSError):
        docstring = None
    if docstring:
        first_line = docstring.strip().splitlines()[0].strip()
        if first_line:
            return first_line
    return f"API reference for openscvx.{dotted}, generated from source."

# Icon + one-line blurb for each subpackage (Material emoji shortcodes for MkDocs).
_REFERENCE_PACKAGES: dict[str, tuple[str, str]] = {
    "algorithms": (
        ":material-chart-timeline:",
        "SCP drivers, trust-region and augmented-Lagrangian style iteration.",
    ),
    "discretization": (
        ":material-grid:",
        "Time grids, collocation, and problem discretization.",
    ),
    "expert": (
        ":material-tune-variant:",
        "BYOF-style expert hooks: specs, lowering helpers, validation.",
    ),
    "init": (
        ":material-map-marker-path:",
        "Initial trajectory guesses: keyframes, interpolation, warm starts.",
    ),
    "integrations": (
        ":material-robot-industrial:",
        "MuJoCo MJX and related dynamics adapters.",
    ),
    "integrators": (
        ":material-sine-wave:",
        "Numerical integration for continuous-time dynamics.",
    ),
    "lowered": (
        ":material-code-json:",
        "Lowered / canonical constraint and cost representations.",
    ),
    "plotting": (
        ":material-chart-bell-curve:",
        "Figures, diagnostics, and optional Viser visualization.",
    ),
    "propagation": (
        ":material-transit-connection-variant:",
        "Trajectory rollout and segment propagation utilities.",
    ),
    "solvers": (
        ":material-lightning-bolt:",
        "Convex subsolvers and interfaces to external QP/LP backends.",
    ),
    "symbolic": (
        ":material-math-integral-box:",
        "Symbolic expressions, constraints, STL, Lie groups, and lowering to JAX/CVXPY.",
    ),
    "utils": (
        ":material-wrench-outline:",
        "Shared helpers used across the library.",
    ),
}

_REFERENCE_CORE_MODULES: tuple[tuple[str, str, str], ...] = (
    (
        "problem",
        ":material-file-cog:",
        "Define dynamics, costs, constraints, and assemble an optimization problem.",
    ),
    (
        "config",
        ":material-tune:",
        "Configuration objects for runs and solver integrations.",
    ),
    (
        "loader",
        ":material-book-open-variant:",
        "Load problems from YAML and serialized specs.",
    ),
)


def _reference_index_markdown(package_names: list[str]) -> str:
    """Build the Reference landing page (Markdown + Material grid cards)."""
    core_lines = []
    for mod, icon, blurb in _REFERENCE_CORE_MODULES:
        core_lines.append(f"- {icon} __[`openscvx.{mod}`]({mod}.md)__ — {blurb}")

    pkg_lines = []
    for name in package_names:
        icon, blurb = _REFERENCE_PACKAGES.get(
            name,
            (":material-package-variant:", "Python subpackage; see module index for contents."),
        )
        pkg_lines.append(f"- {icon} __[`openscvx.{name}`]({name}/index.md)__ — {blurb}")

    core_block = "\n".join(core_lines)
    pkg_block = "\n".join(pkg_lines)

    return f"""---
title: API reference
description: >-
  Browse the openscvx package: core modules and subpackages, all documented from source.
---

# API reference

This section documents **OpenSCvx** from live Python sources via
[mkdocstrings](https://mkdocstrings.github.io/). Use the left navigation to open any module, or
jump into a **core module** or **subpackage** below.

## Core modules

<div class="grid cards" markdown>

{core_block}

</div>

## Subpackages

<div class="grid cards" markdown>

{pkg_block}

</div>

---

Every page under **Reference** corresponds to a Python module under `openscvx/`. Internal APIs
prefixed with `_` are omitted from this reference.
"""


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("Reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __init__, __main__, __pycache__, and private modules (starting with _)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    # Skip if parts is empty (happens when __init__.py is at root level)
    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    dotted = ".".join(parts)
    identifier = ".".join(("openscvx",) + parts)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Front matter qualifies the page title by package path (leaf module names
        # such as ``state`` and ``base`` recur across the jax/cvxpy/latex lowerers)
        # and gives each module a distinct meta description.
        fd.write("---\n")
        fd.write(f"title: {dotted}\n")
        # json.dumps yields a double-quoted, escaped scalar that is valid YAML.
        fd.write(f"description: {json.dumps(_module_summary(path, dotted))}\n")
        fd.write("---\n\n")
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Discover top-level subpackages (directories with __init__.py), excluding tooling dirs
_package_names: list[str] = []
for child in sorted(src.iterdir()):
    if (
        child.is_dir()
        and (child / "__init__.py").is_file()
        and not child.name.startswith("_")
        and child.name not in ("__pycache__",)
    ):
        _package_names.append(child.name)

index_md = _reference_index_markdown(_package_names)
with mkdocs_gen_files.open(Path("Reference", "index.md"), "w") as fd:
    fd.write(index_md)
mkdocs_gen_files.set_edit_path(Path("Reference", "index.md"), Path("scripts/gen_ref_pages.py"))

# Write the navigation file for literate-nav (API landing page must be first so the Reference tab
# and section index open Reference/index.md — see mkdocs-literate-nav resolve_directories_in_nav).
# Important: no blank line between this item and the rest — a blank line makes Markdown emit a
# "loose" list (<li><p><a>), which mkdocs-literate-nav's parser rejects.
_REFERENCE_SUMMARY_HEAD = "* [API reference](index.md)\n"
with mkdocs_gen_files.open("Reference/SUMMARY.md", "w") as nav_file:
    nav_file.write(_REFERENCE_SUMMARY_HEAD)
    nav_file.writelines(nav.build_literate_nav())
