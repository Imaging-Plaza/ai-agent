"""
Unit tests for the repo_info tool with DeepWiki integration.

Tests cover:
1. DeepWiki success case (GitHub repo)
2. DeepWiki failure/timeout -> fallback to repocards
3. GitLab repository URL handling
4. Non-GitHub/non-GitLab URL handling
5. Various URL format edge cases

Run with: pytest tests/test_deepwiki_repo_info.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, Mock

import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Mock the problematic imports before importing the modules
sys.modules["retriever"] = Mock()
sys.modules["retriever.software_doc"] = Mock()
sys.modules["retriever.reranker"] = Mock()
sys.modules["api"] = Mock()
sys.modules["api.pipeline"] = Mock()

from ai_agent.agent.tools.repo_info_tool import (
    RepoSummaryInput,
    RepoSummaryOutput,
    tool_repo_summary,
)
from ai_agent.agent.tools.deepwiki_tool import (
    DeepWikiContentsOutput,
    DeepWikiInput,
    get_wiki_contents,
)
from ai_agent.agent.utils import coerce_github_url_or_none, _coerce_owner_repo_ref

# ======================== Fixtures ========================


@pytest.fixture
def mock_deepwiki_success():
    """Mock successful DeepWiki response."""
    return DeepWikiContentsOutput(
        success=True,
        contents="# Test Repository\n\nThis is a test repository with documentation from DeepWiki.",
        truncated=False,
    )


@pytest.fixture
def mock_deepwiki_failure():
    """Mock failed DeepWiki response."""
    return DeepWikiContentsOutput(
        success=False,
        error="DeepWiki request timed out after 60s",
        truncated=False,
    )


@pytest.fixture
def mock_repocards_response():
    """Mock repocards.get_repo_info response."""
    return (
        "# Test Repository (via repocards)\n\nREADME content from repocards fallback."
    )


# ======================== DeepWiki Tool Tests ========================


@pytest.mark.asyncio
async def test_deepwiki_success():
    """Test successful DeepWiki wiki contents retrieval."""
    with patch("ai_agent.agent.tools.deepwiki_tool.MCPServerStreamableHTTP") as mock_server_class:
        # Setup mock server
        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=None)
        mock_server.direct_call_tool = AsyncMock(
            return_value=[
                "# Repository Documentation\n\nThis is test content from DeepWiki."
            ]
        )
        mock_server_class.return_value = mock_server

        # Test
        result = await get_wiki_contents(DeepWikiInput(url="owner/repo"))
        print(result)

        assert result.success is True
        assert result.contents is not None
        assert "Repository Documentation" in result.contents
        assert result.error is None


@pytest.mark.asyncio
async def test_deepwiki_timeout():
    """Test DeepWiki timeout handling."""
    with patch("ai_agent.agent.tools.deepwiki_tool.MCPServerStreamableHTTP") as mock_server_class:
        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=None)
        mock_server.direct_call_tool = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_server_class.return_value = mock_server

        result = await get_wiki_contents(DeepWikiInput(url="owner/repo"))

        assert result.success is False
        assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_deepwiki_empty_response():
    """Test DeepWiki returning empty/None content."""
    with patch("ai_agent.agent.tools.deepwiki_tool.MCPServerStreamableHTTP") as mock_server_class:
        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=None)
        mock_server.direct_call_tool = AsyncMock(return_value=[])
        mock_server_class.return_value = mock_server

        result = await get_wiki_contents(DeepWikiInput(url="owner/repo"))

        assert result.success is False
        assert "No content returned" in result.error


@pytest.mark.asyncio
async def test_deepwiki_connection_error():
    """Test DeepWiki connection errors."""
    with patch("ai_agent.agent.tools.deepwiki_tool.MCPServerStreamableHTTP") as mock_server_class:
        mock_server_class.side_effect = ConnectionError("Connection refused")

        result = await get_wiki_contents(DeepWikiInput(url="owner/repo"))

        assert result.success is False
        assert "Failed to connect" in result.error


# ======================== URL Coercion Tests ========================


class TestURLCoercion:
    """Test URL parsing and normalization."""

    def test_coerce_github_url_owner_repo_format(self):
        """Test owner/repo format."""
        result = coerce_github_url_or_none("owner/repo")
        assert result == "https://github.com/owner/repo"

    def test_coerce_github_url_full_https(self):
        """Test full HTTPS GitHub URL."""
        result = coerce_github_url_or_none("https://github.com/owner/repo")
        assert result == "https://github.com/owner/repo"

    def test_coerce_github_url_with_tree_ref(self):
        """Test GitHub URL with tree/ref."""
        result = coerce_github_url_or_none("https://github.com/owner/repo/tree/main")
        assert result == "https://github.com/owner/repo#main"

    def test_coerce_github_url_with_git_suffix(self):
        """Test GitHub URL with .git suffix."""
        result = coerce_github_url_or_none("https://github.com/owner/repo.git")
        assert result == "https://github.com/owner/repo"

    def test_coerce_github_url_without_scheme(self):
        """Test github.com URL without scheme."""
        result = coerce_github_url_or_none("github.com/owner/repo")
        assert result == "https://github.com/owner/repo"

    def test_coerce_github_url_gitlab_returns_none(self):
        """Test that GitLab URLs return None."""
        result = coerce_github_url_or_none("https://gitlab.com/owner/repo")
        assert result is None

    def test_coerce_github_url_random_url_returns_none(self):
        """Test that non-GitHub URLs return None."""
        result = coerce_github_url_or_none("https://example.com/some/path")
        assert result is None

    def test_coerce_github_url_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = coerce_github_url_or_none("")
        assert result is None

    def test_coerce_owner_repo_ref_extraction(self):
        """Test _coerce_owner_repo_ref extracts owner, repo, ref correctly."""
        owner, repo, ref = _coerce_owner_repo_ref("owner/repo")
        assert owner == "owner"
        assert repo == "repo"
        assert ref is None

    def test_coerce_owner_repo_ref_with_tree(self):
        """Test _coerce_owner_repo_ref with tree/branch."""
        owner, repo, ref = _coerce_owner_repo_ref(
            "https://github.com/owner/repo/tree/develop"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert ref == "develop"

    def test_coerce_owner_repo_ref_invalid_raises(self):
        """Test _coerce_owner_repo_ref raises on invalid input."""
        with pytest.raises(ValueError, match="BAD_REPO_URL"):
            _coerce_owner_repo_ref("not-a-valid-repo")

    def test_coerce_owner_repo_ref_gitlab_raises(self):
        """Test _coerce_owner_repo_ref raises on GitLab URL."""
        with pytest.raises(ValueError, match="BAD_REPO_URL"):
            _coerce_owner_repo_ref("https://gitlab.com/owner/repo")


# ======================== Repo Info Tool Tests ========================


@pytest.mark.asyncio
async def test_repo_info_deepwiki_success(mock_deepwiki_success):
    """Test repo_info tool with successful DeepWiki response."""
    with patch(
        "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
        new_callable=AsyncMock,
    ) as mock_deepwiki:
        mock_deepwiki.return_value = mock_deepwiki_success

        result = await tool_repo_summary(RepoSummaryInput(url="owner/repo"))

        assert isinstance(result, RepoSummaryOutput)
        assert result.source == "deepwiki"
        assert "DeepWiki" in result.summary
        assert result.truncated is False
        mock_deepwiki.assert_called_once()


@pytest.mark.asyncio
async def test_repo_info_deepwiki_failure_fallback_to_repocards(
    mock_deepwiki_failure, mock_repocards_response
):
    """Test repo_info tool falls back to repocards when DeepWiki fails."""
    with (
        patch(
            "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
            new_callable=AsyncMock,
        ) as mock_deepwiki,
        patch(
            "ai_agent.agent.tools.repo_info_tool.repocards.get_repo_info"
        ) as mock_repocards,
    ):
        mock_deepwiki.return_value = mock_deepwiki_failure
        mock_repocards.return_value = mock_repocards_response

        result = await tool_repo_summary(RepoSummaryInput(url="owner/repo"))

        assert isinstance(result, RepoSummaryOutput)
        assert result.source == "repocards"
        assert "repocards" in result.summary.lower()
        mock_deepwiki.assert_called_once()
        mock_repocards.assert_called_once()


@pytest.mark.asyncio
async def test_repo_info_deepwiki_exception_fallback():
    """Test repo_info tool handles DeepWiki exceptions and falls back."""
    with (
        patch(
            "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
            new_callable=AsyncMock,
        ) as mock_deepwiki,
        patch(
            "ai_agent.agent.tools.repo_info_tool.repocards.get_repo_info"
        ) as mock_repocards,
    ):
        mock_deepwiki.side_effect = Exception("DeepWiki connection error")
        mock_repocards.return_value = "# Fallback content"

        result = await tool_repo_summary(RepoSummaryInput(url="owner/repo"))

        assert result.source == "repocards"
        mock_repocards.assert_called_once()


@pytest.mark.asyncio
async def test_repo_info_both_fail_error_response():
    """Test repo_info tool when both DeepWiki and repocards fail."""
    with (
        patch(
            "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
            new_callable=AsyncMock,
        ) as mock_deepwiki,
        patch(
            "ai_agent.agent.tools.repo_info_tool.repocards.get_repo_info"
        ) as mock_repocards,
    ):
        mock_deepwiki.return_value = DeepWikiContentsOutput(
            success=False, error="DeepWiki failed"
        )
        mock_repocards.side_effect = Exception("Repocards API error")

        result = await tool_repo_summary(RepoSummaryInput(url="owner/repo"))

        assert result.source == "error"
        assert "Error" in result.summary
        assert "Failed to fetch repository information" in result.summary


@pytest.mark.asyncio
async def test_repo_info_truncation():
    """Test that repo_info properly handles content truncation."""
    # Create content longer than MAX_CHARS (20000)
    long_content = "x" * 25000

    with patch(
        "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
        new_callable=AsyncMock,
    ) as mock_deepwiki:
        mock_deepwiki.return_value = DeepWikiContentsOutput(
            success=True, contents=long_content, truncated=True
        )

        result = await tool_repo_summary(RepoSummaryInput(url="owner/repo"))

        assert result.truncated is True
        assert result.source == "deepwiki"


# ======================== Integration Tests ========================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("INTEGRATION_TESTS"),
    reason="Skipping integration tests (set INTEGRATION_TESTS=1 to run)",
)
async def test_real_github_repo():
    """Integration test with a real GitHub repository (requires network)."""
    result = await tool_repo_summary(
        RepoSummaryInput(url="https://github.com/python/cpython")
    )

    assert result.source in ["deepwiki", "repocards"]
    assert len(result.summary) > 0
    assert result.summary != ""


# ======================== Edge Cases ========================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_repo_info_with_various_github_formats(self):
        """Test repo_info handles various GitHub URL formats."""
        test_cases = [
            "owner/repo",
            "https://github.com/owner/repo",
            "http://github.com/owner/repo",
            "github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "https://github.com/owner/repo/tree/main",
        ]

        with patch(
            "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
            new_callable=AsyncMock,
        ) as mock_deepwiki:
            mock_deepwiki.return_value = DeepWikiContentsOutput(
                success=True, contents="# Test", truncated=False
            )

            for url in test_cases:
                result = await tool_repo_summary(RepoSummaryInput(url=url))
                assert result.source == "deepwiki"
                assert result.summary is not None

    def test_gitlab_url_detection(self):
        """Test that GitLab URLs are properly detected as non-GitHub."""
        gitlab_urls = [
            "https://gitlab.com/owner/repo",
            "gitlab.com/owner/repo",
            "https://gitlab.example.com/owner/repo",
        ]

        for url in gitlab_urls:
            result = coerce_github_url_or_none(url)
            assert result is None, f"Expected None for {url}, got {result}"

    def test_other_urls_detection(self):
        """Test that non-GitHub/non-GitLab URLs return None."""
        other_urls = [
            "https://bitbucket.org/owner/repo",
            "https://example.com/some/path",
            "https://docs.github.com/",
            "not-a-url-at-all",
            "http://localhost:3000/repo",
        ]

        for url in other_urls:
            result = coerce_github_url_or_none(url)
            assert result is None, f"Expected None for {url}, got {result}"

    @pytest.mark.asyncio
    async def test_empty_url(self):
        """Test handling of empty URL string."""
        with (
            patch(
                "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
                new_callable=AsyncMock,
            ) as mock_deepwiki,
            patch(
                "ai_agent.agent.tools.repo_info_tool.repocards.get_repo_info"
            ) as mock_repocards,
        ):
            mock_deepwiki.side_effect = ValueError("BAD_REPO_URL")
            mock_repocards.side_effect = Exception("Invalid URL")

            result = await tool_repo_summary(RepoSummaryInput(url=""))

            assert result.source == "error"

    @pytest.mark.asyncio
    async def test_url_with_special_characters(self):
        """Test handling of URLs with special characters in repo name."""
        with patch(
            "ai_agent.agent.tools.repo_info_tool.get_wiki_contents",
            new_callable=AsyncMock,
        ) as mock_deepwiki:
            mock_deepwiki.return_value = DeepWikiContentsOutput(
                success=True, contents="# Test", truncated=False
            )

            # Test with hyphens, underscores, dots
            result = await tool_repo_summary(
                RepoSummaryInput(url="owner/repo-name_with.special")
            )
            assert result.source == "deepwiki"


# ======================== Performance Tests ========================


@pytest.mark.asyncio
async def test_deepwiki_timeout_duration():
    """Test that DeepWiki timeout is properly configured."""
    import time

    with patch("ai_agent.agent.tools.deepwiki_tool.MCPServerStreamableHTTP") as mock_server_class:
        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=None)

        # Simulate a slow response
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(100)  # Longer than timeout

        mock_server.direct_call_tool = slow_call
        mock_server_class.return_value = mock_server

        start = time.time()
        result = await get_wiki_contents(DeepWikiInput(url="owner/repo"))
        duration = time.time() - start

        # Should timeout and not take 100 seconds
        assert duration < 65  # Timeout is 60s plus some buffer
        assert result.success is False
        assert "timed out" in result.error.lower()
