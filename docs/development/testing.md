# Testing

The AI Imaging Agent uses pytest for testing. This guide covers running tests and writing new ones.

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_retrieval_pipeline.py

# Run specific test
pytest tests/test_retrieval_pipeline.py::test_basic_retrieval

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ai_agent --cov-report=html
```

### Test Categories

Tests are marked by category:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Organization

### Directory Structure

```
tests/
├── data/
│   ├── test_data.json         # Test cases
│   └── 0002.DCM               # Sample DICOM file
├── test_retrieval_pipeline.py # Retrieval tests
├── test_deepwiki_repo_info.py # Repo info tests
├── test_gpt4o_vision.py       # VLM tests (integration)
└── __pycache__/
```

### Test File Naming

- `test_*.py`: Test files
- `*_test.py`: Alternative naming (less common)

### Test Function Naming

```python
def test_basic_retrieval():
    """Test basic retrieval functionality."""
    pass

def test_edge_case_empty_query():
    """Test handling of empty query."""
    pass

def test_integration_full_pipeline():
    """Integration test for complete pipeline."""
    pass
```

## Writing Tests

### Unit Test Example

```python
import pytest
from ai_agent.retriever.vector_index import VectorIndex

def test_vector_index_search():
    """Test FAISS vector search."""
    # Arrange
    index = VectorIndex()
    index.load("artifacts/rag_index")
    
    query = "segment lungs CT"
    
    # Act
    results = index.search(query, k=5)
    
    # Assert
    assert len(results) == 5
    assert all(r['score'] > 0 for r in results)
    assert 'TotalSegmentator' in [r['name'] for r in results]
```

### Integration Test Example

```python
import pytest
from ai_agent.api.pipeline import RAGImagingPipeline

@pytest.mark.integration
def test_full_pipeline_with_image():
    """Integration test with real image and VLM call."""
    # Arrange
    pipeline = RAGImagingPipeline(
        catalog_path="dataset/catalog.jsonl",
        index_dir="artifacts/rag_index"
    )
    
    # Act
    result = pipeline.recommend(
        query="segment lungs",
        files=["tests/data/chest_ct.dcm"]
    )
    
    # Assert
    assert result.status == "complete"
    assert len(result.recommendations) > 0
    assert result.recommendations[0].accuracy_score > 70
```

### Parametrized Tests

```python
@pytest.mark.parametrize("query,expected_tool", [
    ("segment brain MRI", "FreeSurfer"),
    ("segment lungs CT", "TotalSegmentator"),
    ("classify chest X-ray", "CheXNet"),
])
def test_retrieval_for_queries(query, expected_tool):
    """Test retrieval returns expected tools for various queries."""
    index = VectorIndex()
    index.load("artifacts/rag_index")
    
    results = index.search(query, k=10)
    tool_names = [r['name'] for r in results]
    
    assert expected_tool in tool_names
```

### Fixtures

```python
import pytest

@pytest.fixture
def pipeline():
    """Provide initialized pipeline for tests."""
    return RAGImagingPipeline(
        catalog_path="dataset/catalog.jsonl",
        index_dir="artifacts/rag_index"
    )

@pytest.fixture
def sample_dicom():
    """Provide path to sample DICOM file."""
    return "tests/data/0002.DCM"

def test_with_fixtures(pipeline, sample_dicom):
    """Test using fixtures."""
    result = pipeline.recommend(
        query="analyze DICOM",
        files=[sample_dicom]
    )
    assert result is not None
```

## Mocking

### Mocking VLM Calls

To avoid API costs during testing:

```python
from unittest.mock import Mock, patch
import pytest

@pytest.fixture
def mock_vlm_response():
    """Mock VLM response."""
    return {
        "status": "complete",
        "recommendations": [
            {
                "rank": 1,
                "name": "TotalSegmentator",
                "accuracy_score": 95,
                "explanation": "Test explanation",
                "reason": "task_match"
            }
        ]
    }

def test_with_mocked_vlm(mock_vlm_response):
    """Test pipeline with mocked VLM."""
    with patch('ai_agent.agent.agent.Agent.run') as mock_run:
        mock_run.return_value = mock_vlm_response
        
        # Test code here
        result = pipeline.recommend(query="test", files=[])
        
        assert result["status"] == "complete"
```

### Mocking File Operations

```python
def test_file_validation():
    """Test file validation without real files."""
    with patch('os.path.getsize') as mock_size:
        mock_size.return_value = 1024 * 1024  # 1 MB
        
        from ai_agent.utils.file_validator import validate_file
        is_valid = validate_file("fake.dcm")
        
        assert is_valid
```

## Test Data

### Using Test Cases

Load test cases from JSON:

```python
import json

def load_test_cases():
    """Load test cases from data file."""
    with open("tests/data/test_data.json") as f:
        return json.load(f)

@pytest.mark.parametrize("test_case", load_test_cases())
def test_from_json(test_case):
    """Test using cases from JSON file."""
    query = test_case["query"]
    expected = test_case["expected_tool"]
    
    # Test logic here
    assert expected in results
```

### Sample Data Files

Keep sample files small:

- **DICOM**: Single slice, low resolution
- **NIfTI**: Small volume (e.g., 64×64×64)
- **Images**: PNG/JPG under 1 MB

## Coverage

### Measuring Coverage

```bash
# Run with coverage
pytest --cov=ai_agent

# Generate HTML report
pytest --cov=ai_agent --cov-report=html

# Open report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Coverage Goals

Aim for:

- **Overall**: >80%
- **Critical paths**: >90% (retrieval, agent, pipeline)
- **Utilities**: >70%

### Coverage Configuration

In `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/ai_agent"]
omit = ["tests/*", "*/migrations/*"]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- Pull requests
- Pushes to main

### CI Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=ai_agent
```

## Best Practices

### Do's

✅ **Test edge cases**: Empty inputs, invalid data, etc.  
✅ **Test error handling**: Verify exceptions are caught  
✅ **Use descriptive names**: `test_retrieval_with_empty_query` not `test1`  
✅ **Keep tests isolated**: Each test should be independent  
✅ **Use fixtures**: Avoid repeating setup code  
✅ **Mock expensive operations**: VLM calls, network requests

### Don'ts

❌ **Don't test implementation details**: Test behavior, not internal state  
❌ **Don't make tests depend on each other**: Each should run independently  
❌ **Don't commit large test files**: Keep test data small  
❌ **Don't skip error checking**: Test both success and failure paths  

## Performance Testing

### Benchmarking

Use pytest-benchmark:

```python
def test_retrieval_performance(benchmark):
    """Benchmark retrieval speed."""
    index = VectorIndex()
    index.load("artifacts/rag_index")
    
    result = benchmark(index.search, "segment lungs", k=10)
    
    assert len(result) == 10
```

### Profiling

```bash
# Profile tests
pytest --profile

# Generate SVG profile
pytest --profile-svg
```

## Debugging Tests

### Running in Debug Mode

```python
# Add to test
import pdb; pdb.set_trace()

# Run pytest
pytest tests/test_file.py
```

### Verbose Output

```bash
# Show print statements
pytest -s

# Very verbose
pytest -vv

# Show local variables on failure
pytest -l
```

### Running Single Test

```bash
# Run one test function
pytest tests/test_file.py::test_function_name -v
```

## Next Steps

- Review [Project Structure](structure.md)
- Read [Contributing Guide](contributing.md)
- Explore [Architecture](../architecture/overview.md)
