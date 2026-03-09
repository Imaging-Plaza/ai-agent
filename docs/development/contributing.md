# Contributing

Thank you for your interest in contributing to the AI Imaging Agent! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-agent.git
cd ai-agent
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Development Workflow

### Making Changes

1. **Make your changes** in the appropriate module
2. **Test your changes** (see [Testing](testing.md))
3. **Update documentation** if needed
4. **Update CHANGELOG.md** following [Keep a Changelog](https://keepachangelog.com/) format

<!-- ### Code Style

We use standard Python tools for code quality:

#### Black

Code formatting:

```bash
black src/ tests/
```

#### Ruff

Linting:

```bash
ruff check src/ tests/
``` -->

#### Type Checking

MyPy for type checking:

```bash
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_retrieval_pipeline.py

# Run with coverage
pytest --cov=ai_agent tests/
```

## Contribution Guidelines

### Code Quality

- **Follow PEP 8**: Use Black for formatting
- **Type hints**: Add type annotations to new functions
- **Docstrings**: Document all public functions and classes
- **Tests**: Add tests for new functionality
- **No warnings**: Fix any linter warnings before submitting

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add alternative search tool for agent

- Implement search_alternative tool
- Add retry logic for insufficient results
- Update agent prompt with tool description
```

**Format**:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

### Pull Requests

1. **Update CHANGELOG.md** under `[Unreleased]` section
2. **Write clear PR description** explaining changes
3. **Link related issues** if applicable
4. **Request review** from maintainers

**PR description template**:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
How to test these changes

## Checklist
- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Areas to Contribute

### High Priority

- **Catalog expansion**: Add more imaging tools
- **Demo integration**: Improve Gradio Space execution and add new spaces for tools
- **Format support**: Add new image format handlers

<!-- ### Good First Issues

Look for issues tagged `good-first-issue` on GitHub:

- Bug fixes
- Documentation improvements
- Test coverage expansion
- Example scripts -->

### Feature Requests

Before implementing new features:

1. **Check existing issues** for similar requests
2. **Open an issue** to discuss the feature
3. **Get feedback** from maintainers
4. **Implement** after discussion

## Documentation

### Writing Documentation

Documentation lives in `docs/` and uses MkDocs Material.

```bash
# Install MkDocs
pip install mkdocs-material

# Serve locally
mkdocs serve

# Open http://127.0.0.1:8000
```

### Documentation Style

- Use **clear headings** and structure
- Include **code examples** where relevant
- Add **warnings** and **tips** for important information
- Keep **language simple** and accessible

## Adding Tools to Catalog

### Process

1. **Create tool entry** in `dataset/catalog.jsonl`:

```json
{
  "@type": "SoftwareSourceCode",
  "name": "ToolName",
  "description": "Tool description",
  "url": "https://github.com/user/tool",
  "license": "Apache-2.0",
  "keywords": ["segmentation", "CT"],
  "supportingData": {
    "modalities": ["CT"],
    "dimensions": ["3D"],
    "formats": ["DICOM"],
    "demo_url": "https://huggingface.co/spaces/user/tool"
  }
}
```

2. **Validate entry**:

```bash
# Check JSON syntax
python -c "import json; print(json.loads('YOUR_JSON_HERE'))"
```

3. **Update checksum**:

```bash
shasum dataset/catalog.jsonl > dataset/catalog.jsonl.sha1
```

4. **Sync catalog**:

```bash
ai_agent sync
```

5. **Test retrieval**:

```bash
ai_agent chat
# Try queries that should return your new tool
```

### Tool Criteria

Tools should:

- ✅ Be relevant to imaging analysis
- ✅ Be actively maintained
- ✅ Have clear documentation
- ✅ Preferably have a runnable demo
- ✅ Be open-source or have free tier

## Reporting Issues

### Bug Reports

Include:

- **Description** of the bug
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment** (OS, Python version, etc.)
- **Logs** if available

### Feature Requests

Include:

- **Use case** for the feature
- **Proposed solution** (if you have one)
- **Alternatives considered**
- **Examples** of similar features

## Code Review Process

1. **Automated checks** run on PR (tests, linting)
2. **Maintainer review** provides feedback
3. **Address feedback** and update PR
4. **Approval** from maintainer(s)
5. **Merge** into main branch

## Release Process

Releases follow semantic versioning:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release date
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions deploys documentation

## Getting Help

- **GitHub Discussions**: Ask questions
- **GitHub Issues**: Report bugs

## Code of Conduct

Be respectful and professional:

- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what's best for the community

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Next Steps

- Review [Project Structure](structure.md)
- Learn about [Testing](testing.md)
- Read [Architecture Overview](../architecture/overview.md)
