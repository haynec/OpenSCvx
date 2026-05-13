"""Generate the example pages and navigation."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
examples_dir = root / "examples"

# Files to skip (utility modules, not examples)
SKIP_FILES = {"plotting.py", "__init__.py"}

# Short intro per top-level folder under examples/ (Material emoji shortcodes).
_EXAMPLES_CATEGORY_BLURB: dict[str, tuple[str, str]] = {
    "abstract": (
        ":material-cube-scan:",
        "Small problems that highlight APIs: STL, impulsive dynamics, and toy costs.",
    ),
    "animations": (
        ":material-movie-open:",
        "Animation-oriented variants of arm, drone, and obstacle demos.",
    ),
    "arm": (
        ":material-robot-industrial:",
        "Manipulator motion, pick-and-place, viewpoint planning, and collisions.",
    ),
    "car": (
        ":material-car:",
        "Dubins and car-like models with obstacles, waypoints, and STL logic.",
    ),
    "drone": (
        ":material-quadcopter:",
        "Racing, obstacle avoidance, viewpoint constraints, and related utilities.",
    ),
    "mjx": (
        ":material-atom:",
        "MuJoCo MJX dynamics: cartpole, multi-body, and skydio-style models.",
    ),
    "mpc": (
        ":material-timer-cog:",
        "Discrete-time MPC-style and realtime-style double integrator setups.",
    ),
    "realtime": (
        ":material-speedometer:",
        "Receding-horizon and streaming demos built on shared base problems.",
    ),
    "rocket": (
        ":material-rocket-launch:",
        "Powered descent guidance sketches in 3-DOF and 6-DOF.",
    ),
    "spacecraft": (
        ":material-satellite-variant:",
        "Orbital transfers, halo orbits, and relative-motion prox-ops style problems.",
    ),
}

# Folder slug -> section title (avoid "Mjx"; keep acronyms readable). Emoji shortcodes belong in
# body text — not in ## headings — or pymdownx/Material may leave them literal and TOC permalinks
# look broken (e.g. ":material-physics: Mjx¶").
_CATEGORY_DISPLAY_NAME: dict[str, str] = {
    "mjx": "MJX",
    "mpc": "MPC",
}


def get_module_docstring(file_path: Path) -> str | None:
    """Extract the module-level docstring from a Python file."""
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree)
    except Exception:
        return None


def get_source_without_docstring(file_path: Path) -> str:
    """Get source code without the module-level docstring."""
    source_code = file_path.read_text()
    try:
        tree = ast.parse(source_code)
        docstring = ast.get_docstring(tree)

        if docstring and tree.body and isinstance(tree.body[0], ast.Expr):
            # Find the end of the docstring node
            docstring_node = tree.body[0].value
            # Get the line number where the docstring ends
            end_lineno = docstring_node.end_lineno

            # Split source into lines and skip the docstring lines
            lines = source_code.splitlines(keepends=True)
            # Rejoin from after the docstring, preserving remaining code
            return "".join(lines[end_lineno:])

        return source_code
    except Exception:
        return source_code


def format_title(name: str) -> str:
    """Convert a file name to a human-readable title."""
    title = name.replace("_", " ").replace("-", " ")
    words = title.split()
    formatted_words = []
    for word in words:
        if word.isupper() or word[0].isdigit():
            formatted_words.append(word)
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def _category_section_title(category: str) -> str:
    """Human-readable H2 title for a top-level examples/ folder."""
    if category in _CATEGORY_DISPLAY_NAME:
        return _CATEGORY_DISPLAY_NAME[category]
    return format_title(category)


def _docstring_teaser(docstring: str | None, max_len: int = 120) -> str:
    if not docstring:
        return "Runnable script; full source on the detail page."
    first = docstring.strip().split("\n", 1)[0].strip()
    if len(first) > max_len:
        return first[: max_len - 1].rstrip() + "…"
    return first


def _examples_index_markdown(
    grouped: dict[str, list[tuple[Path, str, str]]],
) -> str:
    """Build the Examples landing page (Markdown + Material grid cards)."""
    section_blocks: list[str] = []
    for category in sorted(grouped.keys()):
        icon, blurb = _EXAMPLES_CATEGORY_BLURB.get(
            category,
            (":material-folder-outline:", "Runnable scripts in this category."),
        )
        cards = []
        for doc_path, title, teaser in grouped[category]:
            rel = doc_path.as_posix()
            cards.append(f"- :material-file-code: __[{title}]({rel})__ — {teaser}")
        card_block = "\n".join(cards)
        heading = _category_section_title(category)
        section_blocks.append(
            f"## {heading}\n\n"
            f"{icon} {blurb}\n\n"
            f'<div class="grid cards" markdown>\n\n'
            f"{card_block}\n\n"
            f"</div>\n"
        )

    sections = "\n".join(section_blocks)

    return f"""---
title: Examples
description: >-
  Browse runnable trajectory optimization scripts from the repository, grouped by topic,
  with source listings on each page.
---

# Examples

This section lists **every example script** under `examples/` that ships with OpenSCvx (at least
one subdirectory deep). Each page shows the module docstring (if any) and the **full Python
source** so you can copy patterns into your own problems.

For **how to run** examples locally (editable install, command line), see the
[Examples overview](../examples.md) in *Getting started*.

---

{sections}

---

Paths mirror the repo layout: `examples/<category>/...`. Utility modules (for example plotting
helpers) are omitted from this index.
"""


# (category, doc_path, page_title, teaser) for the landing page
_index_entries: list[tuple[str, Path, str, str]] = []

for path in sorted(examples_dir.rglob("*.py")):
    # Skip files in the root examples/ directory and skip utility files
    rel_path = path.relative_to(examples_dir)
    if len(rel_path.parts) < 2:  # Must be in a subdirectory
        continue
    if path.name in SKIP_FILES or path.name.startswith("_"):
        continue

    # Create the documentation path
    module_path = rel_path.with_suffix("")
    doc_path = rel_path.with_suffix(".md")
    full_doc_path = Path("Examples", doc_path)

    parts = tuple(module_path.parts)

    # Build navigation entry with formatted titles
    nav_parts = tuple(format_title(part) for part in parts)
    nav[nav_parts] = doc_path.as_posix()

    # Get module docstring and source code
    docstring = get_module_docstring(path)
    source_code = get_source_without_docstring(path)
    page_title = format_title(path.stem)
    category = rel_path.parts[0]
    _index_entries.append((category, doc_path, page_title, _docstring_teaser(docstring)))

    # Generate the markdown content
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {page_title}\n\n")

        if docstring:
            fd.write(f"{docstring}\n\n")

        fd.write(f"**File:** `examples/{rel_path}`\n\n")
        fd.write("```python\n")
        fd.write(source_code)
        fd.write("\n```\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

_grouped: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
for category, doc_path, title, teaser in _index_entries:
    _grouped[category].append((doc_path, title, teaser))
for cat in _grouped:
    _grouped[cat].sort(key=lambda row: row[0].as_posix())

index_md = _examples_index_markdown(_grouped)
with mkdocs_gen_files.open(Path("Examples", "index.md"), "w") as fd:
    fd.write(index_md)
mkdocs_gen_files.set_edit_path(Path("Examples", "index.md"), Path("scripts/gen_example_pages.py"))

# Landing page first so the Examples tab and section index open Examples/index.md (same pattern as
# Reference/SUMMARY.md in gen_ref_pages.py).
_EXAMPLES_SUMMARY_HEAD = "* [Examples](index.md)\n"
with mkdocs_gen_files.open("Examples/SUMMARY.md", "w") as nav_file:
    nav_file.write(_EXAMPLES_SUMMARY_HEAD)
    nav_file.writelines(nav.build_literate_nav())
