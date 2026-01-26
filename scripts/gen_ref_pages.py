"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "openscvx"

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

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(("openscvx",) + parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation file for literate-nav
with mkdocs_gen_files.open("Reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
