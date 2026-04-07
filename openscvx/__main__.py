"""CLI entry point: ``openscvx path/to/problem.yaml``."""

import argparse
import json
import sys
from pathlib import Path


def _cmd_solve(args):
    """Solve a trajectory optimization problem from a YAML/JSON file."""
    path: Path = args.config
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        from openscvx.loader import load_yaml

        kwargs = load_yaml(path)
    elif suffix == ".json":
        from openscvx.loader import load_json

        kwargs = load_json(path)
    else:
        print(
            f"Error: unsupported file type {suffix!r} (expected .yaml, .yml, or .json)",
            file=sys.stderr,
        )
        sys.exit(1)

    from openscvx.problem import Problem

    settings = kwargs.pop("settings", None)
    problem = Problem(**kwargs)
    if settings:
        problem.settings.apply_dict(settings)
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    if args.output:
        result.save(args.output)
        print(f"Results saved to {args.output}")


def _cmd_schema(args):
    """Generate the JSON Schema for the YAML/JSON problem format."""
    from openscvx.loader import ProblemSpec

    schema = json.dumps(ProblemSpec.model_json_schema(), indent=2) + "\n"

    if args.output:
        args.output.write_text(schema)
        print(f"Schema written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(schema)


def main():
    # Backwards compat: bare `openscvx path/to/problem.yaml` (no subcommand).
    # Detect this before argparse sees the args, and prepend "solve".
    _SUBCOMMANDS = {"solve", "schema", "-h", "--help"}
    if len(sys.argv) > 1 and sys.argv[1] not in _SUBCOMMANDS:
        sys.argv.insert(1, "solve")

    parser = argparse.ArgumentParser(
        prog="openscvx",
        description="OpenSCvx trajectory optimization toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- solve ---
    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve a trajectory optimization problem from a YAML/JSON config file.",
    )
    solve_parser.add_argument(
        "config", type=Path, help="Path to a YAML or JSON problem definition file"
    )
    solve_parser.add_argument("-o", "--output", type=Path, help="Save results to a .npz file")
    solve_parser.set_defaults(func=_cmd_solve)

    # --- schema ---
    schema_parser = subparsers.add_parser(
        "schema",
        help="Print the JSON Schema for the YAML/JSON problem format.",
    )
    schema_parser.add_argument(
        "-o", "--output", type=Path, help="Write schema to a file instead of stdout"
    )
    schema_parser.set_defaults(func=_cmd_schema)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
