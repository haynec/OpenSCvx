"""CLI entry point: ``openscvx path/to/problem.yaml``."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="openscvx",
        description="Solve a trajectory optimization problem from a YAML/JSON config file.",
    )
    parser.add_argument("config", type=Path, help="Path to a YAML or JSON problem definition file")
    parser.add_argument("-o", "--output", type=Path, help="Save results to a .npz file")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
