"""Generate the example pages and navigation."""

import ast
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
examples_dir = root / "examples"

# Files to skip (utility modules, not examples)
SKIP_FILES = {"plotting.py", "__init__.py"}


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

    # Generate the markdown content
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        title = format_title(path.stem)
        fd.write(f"# {title}\n\n")

        if docstring:
            fd.write(f"{docstring}\n\n")

        fd.write(f"**File:** `examples/{rel_path}`\n\n")
        fd.write("```python\n")
        fd.write(source_code)
        fd.write("\n```\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation file for literate-nav
with mkdocs_gen_files.open("Examples/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
