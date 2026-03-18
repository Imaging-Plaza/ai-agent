# AGENTS.md

This file defines repository-wide agent guidance for contributors working in this codebase.

## Scope And Priority

- Use this file for shared workflow rules and contributor expectations.
- Use [.github/copilot-instructions.md](.github/copilot-instructions.md) for architecture details, data-flow patterns, and domain-specific constraints.
- If guidance conflicts, prioritize repository reality in this order:
  1. Executable code in [src/ai_agent/](src/ai_agent)
  2. Runtime/config files ([pyproject.toml](pyproject.toml), [config.yaml](config.yaml), [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json))
  3. Documentation pages

## Default Execution Context

Assume work is done inside the dev container unless a task says otherwise.

- Base environment: Debian Bookworm dev container
- Python: 3.12 (from dev container image)
- Package workflow: uv-managed virtual environment in .venv
- Default interpreter: .venv/bin/python

Preferred setup and install commands:

```bash
uv venv
uv pip install -e .
uv pip install -e ".[dev]"
```

## Command Truth

Use these commands as the current baseline:

```bash
ai_agent chat
ai_agent sync
pytest tests/
```

Notes:
- The CLI modes are defined in [src/ai_agent/cli.py](src/ai_agent/cli.py).
- If helper scripts (for example [justfile](justfile)) disagree with CLI behavior, align docs to real CLI behavior and then fix scripts in a follow-up change.

## Contributor Workflow Expectations

1. Confirm behavior from code before updating docs.
2. Keep module boundaries clear:
   - retrieval logic in `retriever/`
   - selection schemas/prompts in `generator/`
   - orchestration in `api/`
   - chat/tool orchestration in `agent/`
   - UI code in `ui/`
3. Prefer small, reviewable changes.
4. For user-facing changes, update [CHANGELOG.md](CHANGELOG.md).
5. For docs changes, keep [README.md](README.md), [docs/index.md](docs/index.md), and [docs/guide.md](docs/guide.md) in sync.

## Documentation Maintenance Rules

When adding or changing functionality:

1. Update architectural context if module boundaries change.
2. Update environment/command docs if startup, install, or test commands change.
3. Add or adjust examples when behavior changes.
4. Verify internal links in docs remain valid.

## Recommended Improvement Priorities

- Align task runners/scripts with current CLI contract (`chat`, `sync`).
- Expand tests around UI handlers and tool-call edge cases.
- Add lightweight retrieval quality regression checks.
- Add docs link validation in CI to prevent drift.


## Tests 

1. Run all tests compatible with pytest in ./tests 

## Developing flow

1. Check AGENTS.md
2. Follow the implementation 
3. Run tests
4. Run linting
5. CHANGELOG upload following keepachangelog format. 

## Related References

- [docs/guide.md](docs/guide.md)
- [docs/development/structure.md](docs/development/structure.md)
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [.github/copilot-instructions.md](.github/copilot-instructions.md)
