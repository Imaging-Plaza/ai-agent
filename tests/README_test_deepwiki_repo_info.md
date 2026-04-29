# DeepWiki Repo Info Test Suite

## Overview

`test_deepwiki_repo_info.py` provides comprehensive unit tests for the `repo_info` tool with DeepWiki integration and URL parsing.

## Test Coverage

### 1. DeepWiki Tool Tests (4 tests)
- **test_deepwiki_success**: Successful DeepWiki MCP SSE connection and content retrieval
- **test_deepwiki_timeout**: Handles timeout after 60s gracefully
- **test_deepwiki_empty_response**: Handles empty/None responses from DeepWiki
- **test_deepwiki_connection_error**: Handles connection errors with proper fallback

### 2. URL Coercion Tests (12 tests)
Tests the `coerce_github_url_or_none()` and `_coerce_owner_repo_ref()` functions:

**GitHub URL Formats (Accepted)**:
- `owner/repo` → `https://github.com/owner/repo`
- `https://github.com/owner/repo`
- `http://github.com/owner/repo`
- `github.com/owner/repo` → `https://github.com/owner/repo`
- `https://github.com/owner/repo.git` → `https://github.com/owner/repo`
- `https://github.com/owner/repo/tree/main` → `https://github.com/owner/repo#main`

**Non-GitHub URLs (Rejected)**:
- GitLab URLs → `None`
- BitBucket URLs → `None`
- Random URLs → `None`
- Empty strings → `None`

**Edge Cases**:
- Extracts owner, repo, and ref from various formats
- Handles special characters in repo names (hyphens, underscores, dots)
- FTP URLs with github.com are extracted via regex

### 3. Repo Info Tool Integration Tests (5 tests)
Tests the full `tool_repo_summary()` pipeline:

- **test_repo_info_deepwiki_success**: DeepWiki succeeds, returns `source="deepwiki"`
- **test_repo_info_deepwiki_failure_fallback_to_repocards**: DeepWiki fails, falls back to repocards
- **test_repo_info_deepwiki_exception_fallback**: DeepWiki exception triggers repocards fallback
- **test_repo_info_both_fail_error_response**: Both fail, returns `source="error"` with message
- **test_repo_info_truncation**: Content over 20K chars is truncated

### 4. Edge Cases Tests (5 tests)
- **test_repo_info_with_various_github_formats**: All valid GitHub URL formats work
- **test_gitlab_url_detection**: GitLab URLs properly rejected
- **test_other_urls_detection**: Non-GitHub/GitLab URLs properly rejected
- **test_empty_url**: Empty URL strings handled gracefully
- **test_url_with_special_characters**: Repo names with special chars work

### 5. Performance Tests (1 test)
- **test_deepwiki_timeout_duration**: Confirms timeout is enforced (~60s)

### 6. Integration Tests (1 test, skipped by default)
- **test_real_github_repo**: Real API call (set `INTEGRATION_TESTS=1` to run)

## Running Tests

```bash
# Run all tests
pytest tests/test_deepwiki_repo_info.py -v

# Run specific test
pytest tests/test_deepwiki_repo_info.py::test_deepwiki_success -v

# Run with integration tests
INTEGRATION_TESTS=1 pytest tests/test_deepwiki_repo_info.py -v

# Run only URL coercion tests
pytest tests/test_deepwiki_repo_info.py::TestURLCoercion -v

# Run with detailed output
pytest tests/test_deepwiki_repo_info.py -vv --tb=short
```

## Test Results

✅ **27 tests passed**  
⏭️ **1 test skipped** (integration test)  
⏱️ **~61 seconds** execution time

## Dependencies

- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- Mock modules for `retriever` and `api` (handled in test setup)

## Architecture

The tests use:
- **Unit test doubles** (mocks/patches) to avoid external dependencies
- **AsyncMock** for testing async functions
- **Fixtures** for reusable test data
- **Parametrized tests** (via loops) for URL format variations

## Key Behaviors Tested

1. ✅ **DeepWiki First**: Tries DeepWiki MCP before repocards
2. ✅ **Graceful Fallback**: Falls back to repocards on any DeepWiki error
3. ✅ **GitLab Rejection**: GitLab URLs are properly rejected (not supported)
4. ✅ **URL Normalization**: Various GitHub URL formats are normalized
5. ✅ **Timeout Handling**: 60s timeout prevents hanging
6. ✅ **Error Reporting**: Both sources failing returns error message

## Notes

- Tests use mocked modules to avoid importing the full RAG pipeline
- Integration test requires network access and GitHub API
- Timeout test takes ~60s due to actual timeout simulation
