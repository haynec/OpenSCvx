## Contributing to OpenSCvx

### Local Installation

To set up OpenSCvx for local development, first clone the repository and create a virtual environment:

> **Note:** These instructions use SSH URLs. If you haven't set up SSH keys with GitHub, see [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

```bash
git clone git@github.com:OpenSCvx/OpenSCvx.git
cd OpenSCvx
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then install the package in editable mode with development dependencies:

```bash
pip install -e ".[test]"
```

For additional features, you can install optional dependencies:
- `pip install -e ".[cvxpygen]"` - Code generation support
- `pip install -e ".[stl]"` - STL (Signal Temporal Logic) constraints

#### Using uv (Alternative)

If you prefer using [uv](https://github.com/astral-sh/uv) for faster dependency resolution:

```bash
git clone git@github.com:OpenSCvx/OpenSCvx.git
cd OpenSCvx
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[test]"
```

To contribute changes, you'll need to create a branch or fork. See [Forking and Merging](#forking-and-merging) below for details on our workflow and [Branch Naming Conventions](#branch-naming-conventions) for required naming patterns.

### Forking and Merging

All contributions must be submitted via pull request and approved by a maintainer before merging. This ensures code quality, maintains consistency across the codebase, and gives contributors feedback on their changes.

We currently use standard merge commits for pull requests. We may transition to squash merges in the future to maintain a cleaner commit history on `main`.

Contributors can either:
1. **Create a branch directly** on the OpenSCvx repository (for those with write access)
2. **Fork the repository** to their own GitHub account and submit pull requests from there (recommended for external contributors)

#### Fork Setup

To set up a fork for development:

1. Fork the repository on GitHub by clicking the "Fork" button on the [OpenSCvx repository page](https://github.com/OpenSCvx/OpenSCvx)

2. Clone your fork locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/OpenSCvx.git
   cd OpenSCvx
   ```

3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream git@github.com:OpenSCvx/OpenSCvx.git
   ```

4. Verify your remotes:
   ```bash
   git remote -v
   # origin    git@github.com:YOUR_USERNAME/OpenSCvx.git (fetch)
   # origin    git@github.com:YOUR_USERNAME/OpenSCvx.git (push)
   # upstream  git@github.com:OpenSCvx/OpenSCvx.git (fetch)
   # upstream  git@github.com:OpenSCvx/OpenSCvx.git (push)
   ```

5. Keep your fork up to date by periodically pulling from upstream:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

#### Branch Naming Conventions

We use [release-drafter](https://github.com/release-drafter/release-drafter) to automatically generate release notes based on pull request labels. These labels are auto-assigned based on your branch name prefix.
To ensure release-drafter can successfully sort pull requests we run a [branch-name-check](.github/workflows/branch-name.yml) workflow to ensure compatibility.

**Required branch prefixes:**

| Prefix | Label Applied | Release Category |
|--------|---------------|------------------|
| `feature/` | enhancement | Features |
| `fix/` | bug | Bug Fixes |
| `refactor/` | refactor | Maintenance |
| `chore/` | chore | Maintenance |
| `docs/` or `doc/` | documentation | Maintenance |

**Examples:**
- `feature/add-new-solver` - Adding a new solver option
- `fix/convergence-issue` - Fixing a bug
- `refactor/simplify-dynamics` - Code refactoring
- `chore/update-dependencies` - Maintenance tasks
- `docs/api-examples` - Documentation updates

See the release-drafter [workflow](.github/workflows/release-drafter.yml) and [configuration](.github/release-drafter.yml) for details.

### Development Guidelines

#### Use of LLM-Based Tools

While the development of OpenSCvx has been greatly assisted by LLM-based tools for coding tasks, it is up to the contributor to use them in a responsible manner. We welcome their use, _but with great power comes great responsibility_.

Our goal is to produce a high-quality codebase that is well organized, easy to modify, easy to understand, and, most importantly, easy to maintain. We will not accept so-called _AI-slop_. Contributors are expected to:

1. **Understand every line** of code they submit. You are responsible for it, not the LLM
2. **Review and refactor** generated code to match the project's style and patterns
3. **Write meaningful commit messages and keep commits small-ish** to break the changes into bite-sized chunks that can be understood based on the commit message.
4. **Test thoroughly** - LLMs can generate plausible-looking but subtly incorrect code. See point 1.

#### Keep It Simple

We value simplicity over cleverness. Before adding code, ask yourself:

- **[YAGNI](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)**: "You Aren't Gonna Need It" - don't add functionality until it's actually needed
- **[KISS](https://en.wikipedia.org/wiki/KISS_principle)**: "Keep It Simple, Stupid" - the simplest solution is usually the best one
- **[Feature creep](https://en.wikipedia.org/wiki/Feature_creep)**: the enemy of maintainable software. A 10-line function that's easy to understand beats a 50-line "flexible" abstraction that handles hypothetical future requirements. When in doubt, leave it out.

While we don't always follow these rules, we try.

#### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before submitting a PR, ensure your code passes:

```bash
ruff check .
ruff format --check .
```

To auto-fix issues:

```bash
ruff check --fix .
ruff format .
```

See the `[tool.ruff]` section in `pyproject.toml` for our configuration.

#### Testing

OpenSCvx uses `pytest` for automated testing. Tests run automatically via GitHub Actions:

- **[tests.yml](.github/workflows/tests.yml)**: Runs on every PR - a fast `unit` job, then gated `e2e` (full SCP solves) and `extras` (optional-dependency tests) jobs
- **[tests-examples.yml](.github/workflows/tests-examples.yml)**: Runs weekly - solves every example problem as a smoke test

##### Running Tests Locally

For faster feedback during development, run tests locally:

```bash
pytest -m "not e2e and not examples and not extras"    # Fast unit tests
pytest -m "e2e and not extras"                         # Full SCP-solve tests
pytest -m extras                                       # Optional-dependency tests
pytest -m examples                                     # Solve every example (slow)
pytest tests/e2e/test_brachistochrone.py -v            # Just the brachistochrone benchmark
```

##### Brachistochrone Problem

The [brachistochrone problem](https://en.wikipedia.org/wiki/Brachistochrone_curve) serves as our primary validation benchmark. This classic problem asks for the curve of fastest descent under gravity between two points, and has a known analytical solution: the cycloid. We use it as part of the unit-test suite because:

- **Analytical solution exists**: We can validate numerical results against the exact cycloid curve
- **Small and fast**: The problem converges quickly, making it ideal for CI/CD pipelines
- **Representative**: It exercises core features like free final time optimization and nonlinear dynamics

##### Test-Driven Development

While we do not strictly follow [test driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development), we aim to keep a representative test-suite where it makes sense to do so and leverage TDD when possible.

For example, the [symbolic-expression layer](openscvx/symbolic/__init__.py) is well covered by [tests](tests/symbolic/) which were written simultaneously with the symbolic layer.

The extensive test suite helps ensure that new features and refactors don't break existing functionality.

#### Documentation

We use [MkDocs](https://www.mkdocs.org/) with [Material](https://squidfunk.github.io/mkdocs-material/) theme for documentation, hosted at [openscvx.github.io/OpenSCvx](https://openscvx.github.io/OpenSCvx/).

API reference is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/). Please use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions and classes:

```python
def solve(self, max_iterations: int = 100) -> OptimizationResult:
    """Solve the trajectory optimization problem.

    Args:
        max_iterations: Maximum number of SCP iterations.

    Returns:
        The optimization result containing the trajectory and convergence info.

    Raises:
        ConvergenceError: If the solver fails to converge.
    """
```

To preview documentation locally, first install the dependencies:

```bash
pip install "mkdocs-material[imaging]" mkdocstrings-python mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

The `social` plugin (enabled in `mkdocs.yml`) uses **CairoSVG**, which loads the native **Cairo** library. A `pip`-only `cairosvg` on macOS often cannot find `libcairo` and will spam warnings during `mkdocs build` / `mkdocs serve`. Install Cairo into the same environment (for example `conda install -c conda-forge cairo cairosvg`) or install Cairo via Homebrew so the dynamic linker can resolve it. Uninstalling `cairosvg` does not silence the build: the plugin will warn on every page that imaging dependencies are missing.

Then serve the documentation locally:

```bash
mkdocs serve
```

To silence MkDocs 2.0 / ProperDocs notices when building with **`--strict`**, set **`NO_MKDOCS_2_WARNING=true`** (Material’s stderr banner) and, if you use ProperDocs, **`DISABLE_MKDOCS_2_WARNING=true`** for the pip-style WARNING; the documentation workflow sets both for CI.

The version dropdown uses Material’s **mike** integration and expects a `versions.json` at the site root. Plain `mkdocs serve` does not run `mike deploy`, so the repo includes a minimal **`docs/versions.json`** (copied into the build) to avoid a 404 in local previews. On **GitHub Pages**, CI replaces the published root with **`mike deploy`**, which writes the real multi-version manifest. To preview multiple versions locally the same way as production, use **`mike serve`** after you have a `gh-pages` branch with deployed versions.
