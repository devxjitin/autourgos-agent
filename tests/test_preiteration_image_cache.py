"""
Coverage for autourgos_agent._preiteration's image-cache LRU/eviction and
thread safety -- previously zero test coverage (autourgos-audit-report.md).
"""
import os
import tempfile
import threading
import time

import pytest

from autourgos_agent import _preiteration
from autourgos_agent._preiteration import (
    PreIterationMiddleware,
    _image_cache,
    _preprocess_image,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


@pytest.fixture(autouse=True)
def clean_image_cache():
    """Snapshot/restore the module-level _image_cache so tests don't leak
    entries (and their backing temp files) into each other."""
    saved = dict(_image_cache)
    _image_cache.clear()
    yield
    for _, (_, path) in _image_cache.items():
        try:
            os.remove(path)
        except OSError:
            pass
    _image_cache.clear()
    _image_cache.update(saved)


def _make_png(tmp_path, name="src.png", size=(1000, 1000), color=(255, 0, 0)):
    path = os.path.join(tmp_path, name)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def test_preprocess_image_caches_result_for_unchanged_mtime(tmp_path):
    src = _make_png(str(tmp_path))
    import logging
    logger = logging.getLogger("test")

    first = _preprocess_image(src, "low", logger)
    second = _preprocess_image(src, "low", logger)

    assert first == second
    assert os.path.exists(first)
    assert len(_image_cache) == 1


def test_preprocess_image_invalidates_cache_on_mtime_change(tmp_path):
    src = _make_png(str(tmp_path))
    import logging
    logger = logging.getLogger("test")

    first = _preprocess_image(src, "low", logger)
    assert os.path.exists(first)

    # Bump the source file's mtime (simulating a re-taken screenshot at the
    # same path) -- must invalidate the cache entry and remove the stale
    # temp file rather than accumulating it forever.
    future = time.time() + 5
    os.utime(src, (future, future))

    second = _preprocess_image(src, "low", logger)

    assert second != first
    assert os.path.exists(second)
    assert not os.path.exists(first)
    assert len(_image_cache) == 1


def test_image_cache_evicts_oldest_entry_beyond_max(tmp_path, monkeypatch):
    monkeypatch.setattr(_preiteration, "_IMAGE_CACHE_MAX_ENTRIES", 2)
    import logging
    logger = logging.getLogger("test")

    paths = [_make_png(str(tmp_path), name=f"img{i}.png", color=(i * 10, 0, 0)) for i in range(3)]
    produced = [_preprocess_image(p, "low", logger) for p in paths]

    # Cache capped at 2 entries -- the oldest (first image processed) was
    # evicted, and its temp file removed from disk, not just its dict entry.
    assert len(_image_cache) == 2
    assert not os.path.exists(produced[0])
    assert os.path.exists(produced[1])
    assert os.path.exists(produced[2])


def test_preprocess_image_is_thread_safe_under_concurrent_calls(tmp_path):
    src = _make_png(str(tmp_path))
    import logging
    logger = logging.getLogger("test")

    results = []
    errors = []

    def worker():
        try:
            results.append(_preprocess_image(src, "low", logger))
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(results) == 16
    # All concurrent calls for the same (path, quality) key converge on one
    # cache entry -- not 16 independently-created, leaked temp files.
    assert len(_image_cache) == 1
    assert len(set(results)) == 1
    assert os.path.exists(results[0])


def test_preiteration_middleware_cleanup_preserves_still_cached_file(tmp_path):
    """A temp file this run created but that's still live in the shared
    cache must survive on_agent_end's cleanup -- only files evicted or
    superseded should actually be removed (see get_injection_kwargs'
    cache-sharing docstring)."""
    src = _make_png(str(tmp_path))
    middleware = PreIterationMiddleware(files=src, image_quality="low")

    middleware.on_iteration_start(1, agent=None)
    injected = middleware.get_injection_kwargs()
    processed_path = injected["files"][0]
    assert os.path.exists(processed_path)

    middleware.on_agent_end("done", agent=None)

    # Still referenced by the shared _image_cache -- not deleted.
    assert os.path.exists(processed_path)


def test_preiteration_middleware_cleanup_removes_evicted_temp_file(tmp_path, monkeypatch):
    """Once a run's temp file is no longer the live cache entry (e.g. the
    source changed and a new one superseded it), cleanup must actually
    remove it instead of leaking it forever."""
    src = _make_png(str(tmp_path))
    middleware = PreIterationMiddleware(files=src, image_quality="low")

    middleware.on_iteration_start(1, agent=None)
    injected = middleware.get_injection_kwargs()
    processed_path = injected["files"][0]

    # Simulate the cache entry being superseded by something else entirely
    # (e.g. a different quality run, or eviction) before this run ends.
    _image_cache.clear()

    middleware.on_agent_error(Exception("boom"), agent=None)

    assert not os.path.exists(processed_path)
