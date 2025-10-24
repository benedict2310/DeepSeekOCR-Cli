# Contributing to DeepSeek-OCR Mac CLI

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/DeepSeekOCR-Cli.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment (see below)

## Development Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
# pip install pre-commit
# pre-commit install
```

## Development Workflow

### 1. Make Your Changes

- Write clear, concise code following Python best practices
- Add docstrings to functions and classes
- Keep functions focused and modular

### 2. Format Your Code

```bash
# Format with Black
black .

# Lint with Ruff
ruff check --fix .
```

### 3. Add Tests

All new features should include tests:

- **Unit tests**: Test individual functions in `tests/test_unit.py`
- **Integration tests**: Test end-to-end workflows in `tests/test_integration.py`
- Use mocks to avoid downloading the large model during tests

Example test:
```python
def test_new_feature(tmp_path):
    """Test description."""
    # Arrange
    test_file = tmp_path / "test.png"
    create_test_image(test_file)

    # Act
    result = your_function(test_file)

    # Assert
    assert result.exists()
    assert "expected" in result.read_text()
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term

# Run specific tests
pytest tests/test_unit.py::TestClassName::test_method_name -v
```

### 5. Check Code Quality

```bash
# Type checking
mypy deepseek_ocr_mac.py --ignore-missing-imports

# Security scan
bandit -r . -f json

# Check all at once
black --check . && ruff check . && pytest
```

## Commit Guidelines

### Commit Message Format

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(cli): add batch processing support

Add ability to process multiple files in parallel using multiprocessing.
Includes progress bar and error handling for failed files.

Closes #42
```

```
fix(pdf): correct page numbering in output files

Fixed issue where PDF pages were numbered incorrectly when processing
files with leading zeros in the filename.

Fixes #38
```

### Commit Best Practices

- Keep commits atomic (one logical change per commit)
- Write clear, descriptive commit messages
- Reference issue numbers when applicable
- Sign your commits (optional): `git commit -s`

## Pull Request Process

### Before Submitting

1. **Update documentation**: If you've changed functionality, update the README
2. **Add changelog entry**: Note your changes in a comment or CHANGELOG.md if it exists
3. **Ensure tests pass**: All CI checks must pass
4. **Resolve conflicts**: Rebase on the latest main/master branch

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe the tests you ran and how to reproduce them.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
```

## Code Style Guidelines

### Python Style

- Follow PEP 8 conventions
- Use Black for formatting (line length: 100)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Docstring Format

```python
def process_image(image_path: Path, output_dir: Path) -> str:
    """
    Process an image and extract text using OCR.

    Args:
        image_path: Path to the input image file
        output_dir: Directory where results should be saved

    Returns:
        Extracted text content as a string

    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image format is not supported
    """
    pass
```

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
from PIL import Image

# Local imports
from module import function
```

## Testing Guidelines

### Test Structure

- Use pytest for all tests
- Organize tests by functionality
- Use fixtures for common setup
- Mock external dependencies (model, network calls)

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths and edge cases
- Test error handling and validation

### Writing Good Tests

```python
class TestFeature:
    """Tests for feature X."""

    def test_happy_path(self):
        """Test normal, expected behavior."""
        pass

    def test_edge_case(self):
        """Test boundary conditions."""
        pass

    def test_error_handling(self):
        """Test error scenarios."""
        with pytest.raises(ValueError):
            function_that_should_fail()
```

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include type hints
- Document parameters, returns, and exceptions

### README Updates

- Update usage examples if you add new features
- Update installation instructions if dependencies change
- Add troubleshooting tips for common issues

### PRD Updates

- Update PRD.md if you change core functionality
- Document architectural decisions

## Issue Reporting

### Bug Reports

Include:
- Python version
- macOS version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Sample files (if relevant)

### Feature Requests

Include:
- Clear description of the feature
- Use cases and benefits
- Potential implementation approach
- Mockups or examples (if applicable)

## Code Review Process

### As a Contributor

- Be responsive to feedback
- Ask questions if feedback is unclear
- Make requested changes promptly
- Keep discussions professional and constructive

### As a Reviewer

- Be respectful and constructive
- Explain the reasoning behind suggestions
- Approve when requirements are met
- Test the changes locally if possible

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).

## Questions?

- Open an issue for questions about contributing
- Tag maintainers for urgent matters
- Check existing issues and PRs for similar questions

## Recognition

Contributors will be recognized in:
- README.md acknowledgments section
- Release notes for significant contributions
- GitHub contributors page

Thank you for contributing! 🎉
