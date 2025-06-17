# Agent Guidelines for crosslearner

This repository uses GitHub Actions to lint and test every commit. The workflow
in `.github/workflows/ci.yml` installs dependencies, runs format checks, and
executes the test suite across multiple Python versions. Agents should mirror
these steps locally to avoid CI failures.

## Setup

1. Use **Python 3.10** or later.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black ruff
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   Replace the last command with the appropriate CUDA wheel if you have GPU
   support (`cu118`, etc.).
3. Working in a virtual environment is recommended to keep packages isolated.

## Linting

CI enforces two linters:

```bash
ruff check .
black --check .
```

Run these before committing. `black` uses its default 88-character line limit.
Use `black .` to automatically format files when needed. `ruff` will flag common
style issues—fix them prior to committing.

## Tests

Run the test suite with:

```bash
pytest --cov=crosslearner --cov-report=xml -q
```

All tests should pass locally. Please add or update tests whenever you modify
behavior or add new features.

## Additional Tips

- Keep commits focused and provide clear messages.
- Document new functions and modules with docstrings.
- Update `requirements.txt` and `pyproject.toml` if you add dependencies.
- Use relative imports within the `crosslearner` package.
- Summarize your changes in `CHANGELOG.md`.

Following these guidelines will keep the codebase consistent and reduce friction
when CI runs on your pull requests.
