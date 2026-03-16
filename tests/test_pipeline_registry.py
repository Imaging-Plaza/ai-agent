from __future__ import annotations

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.core import pipeline_registry


class _DummyPipeline:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir).resolve()


@pytest.fixture(autouse=True)
def _reset_registry_state():
    pipeline_registry.reset_pipeline()
    yield
    pipeline_registry.reset_pipeline()


@pytest.fixture
def _use_dummy_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_registry, "_PIPELINE_CLASS", _DummyPipeline)


def test_pipeline_singleton_same_dir(_use_dummy_pipeline, tmp_path):

    p1 = pipeline_registry.get_pipeline(index_dir=str(tmp_path))
    p2 = pipeline_registry.get_pipeline(index_dir=str(tmp_path))

    assert p1 is p2


def test_pipeline_recreated_for_different_dir(_use_dummy_pipeline, tmp_path):

    p1 = pipeline_registry.get_pipeline(index_dir=str(tmp_path / "a"))
    p2 = pipeline_registry.get_pipeline(index_dir=str(tmp_path / "b"))

    assert p1 is not p2
    assert p2.index_dir == (tmp_path / "b").resolve()
