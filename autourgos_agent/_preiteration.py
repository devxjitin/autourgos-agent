"""
_preiteration.py -- SEQUENTIAL, PARALLEL, and the inline pre-iteration runtime.

Ported from the (now retired) standalone autourgos-preiteration package,
then folded into the agent loop itself (see _PreIterationRuntime below).
Run callbacks and inject files (screenshots, docs) before each agent
iteration -- sync and async hooks, sequential and parallel execution, and
automatic image compression to reduce LLM token costs.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import inspect
import logging
import os
import tempfile
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

__all__ = ["SEQUENTIAL", "PARALLEL", "is_async_callable"]

# _NullPreIteration/_PreIterationRuntime (bottom of this file) are internal
# -- Agent(pre_iteration_callback=..., pre_iteration_files=...) builds one
# directly, the same way Agent(history=...)/Agent(summarize_every=...)
# build _HistoryRecorder/their inline summarizer state instead of going
# through the CallbackHandler/middleware bus. Features native to this
# package (pre-iteration injection, history, summarization) are plain
# Agent() constructor kwargs, not middleware -- the middleware bus
# (CallbackHandler/middleware=[...]) is reserved for third-party
# extensions (autourgos-hcix, autourgos-skills, your own code), not for
# this package's own built-in features.


# ── helpers ────────────────────────────────────────────────────────────────

def _run_coroutine_sync(coro: Any) -> Any:
    """
    Run a coroutine to completion from synchronous code, safe whether or
    not the calling thread already has a running event loop.

    ``_PreIterationRuntime.before_iteration`` is a plain sync method, but
    it can end up called on a thread that already has a running loop (a
    caller invoking it directly instead of through
    ``abefore_iteration``'s worker-thread offload). Running the coroutine
    on an isolated thread with its own fresh loop sidesteps deadlocking
    that thread against its own loop in that case.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def is_async_callable(obj: Any) -> bool:
    if obj is None:
        return False
    if inspect.iscoroutinefunction(obj):
        return True
    if hasattr(obj, "__call__") and inspect.iscoroutinefunction(obj.__call__):
        return True
    if hasattr(obj, "_is_async") and getattr(obj, "_is_async"):
        return True
    return False


# ── SEQUENTIAL ────────────────────────────────────────────────────────────

class SEQUENTIAL:
    """
    Run multiple pre-iteration hooks one after another.

    Supports both sync and async callables. If any hook is async the whole
    chain becomes async and must be awaited.

    Example
    -------
    ::

        from autourgos_agent import Agent, SEQUENTIAL

        def capture_screen(iteration: int) -> None:
            take_screenshot(f"step_{iteration}.png")

        def log_step(iteration: int) -> None:
            print(f"Iteration {iteration} starting")

        agent = Agent(
            llm=my_llm,
            pre_iteration_callback=SEQUENTIAL[capture_screen, log_step],
        )
    """

    def __init__(self, *funcs: Callable[[int], Any]) -> None:
        self.funcs     = [f for f in funcs if f is not None]
        self._is_async = any(is_async_callable(f) for f in self.funcs)

    def __class_getitem__(cls, item: Any) -> "SEQUENTIAL":
        if not isinstance(item, tuple):
            item = (item,)
        return cls(*item)

    def __call__(self, iteration: int) -> Any:
        if self._is_async:
            return self._run_async(iteration)
        for func in self.funcs:
            func(iteration)
        return None

    async def _run_async(self, iteration: int) -> None:
        for func in self.funcs:
            if is_async_callable(func):
                await func(iteration)
            else:
                res = func(iteration)
                if inspect.iscoroutine(res):
                    await res


# ── PARALLEL ──────────────────────────────────────────────────────────────

class PARALLEL:
    """
    Run multiple pre-iteration hooks at the same time.

    Sync hooks run in a ``ThreadPoolExecutor``; async hooks run as
    ``asyncio`` tasks. Results (and errors) are gathered before the
    agent continues.

    Example
    -------
    ::

        from autourgos_agent import Agent, PARALLEL

        agent = Agent(
            llm=my_llm,
            pre_iteration_callback=PARALLEL[capture_screen, refresh_cache, ping_health_check],
        )
    """

    def __init__(self, *funcs: Callable[[int], Any]) -> None:
        self.funcs     = [f for f in funcs if f is not None]
        self._is_async = any(is_async_callable(f) for f in self.funcs)

    def __class_getitem__(cls, item: Any) -> "PARALLEL":
        if not isinstance(item, tuple):
            item = (item,)
        return cls(*item)

    def __call__(self, iteration: int) -> Any:
        if self._is_async:
            return self._run_async(iteration)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self.funcs))
        ) as executor:
            futures = [executor.submit(func, iteration) for func in self.funcs]
            concurrent.futures.wait(futures)
            for fut in futures:
                fut.result()  # re-raises any exceptions
        return None

    async def _run_async(self, iteration: int) -> None:
        loop = asyncio.get_running_loop()

        tasks = []
        for func in self.funcs:
            if is_async_callable(func):
                tasks.append(asyncio.ensure_future(func(iteration)))
            else:
                tasks.append(loop.run_in_executor(None, func, iteration))
        if tasks:
            await asyncio.gather(*tasks)


# ── image helpers ─────────────────────────────────────────────────────────

_IMAGE_QUALITY_TIERS: Dict[str, Tuple] = {
    "low":    (512,  60,   "low"),
    "medium": (768,  70,   "auto"),
    "high":   (None, None, "high"),
    "auto":   (None, None, "auto"),
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# (source_path, quality_key) -> (mtime, preprocessed_path)
#
# Keyed WITHOUT mtime so there is at most one live entry per (path, quality):
# when the source file's mtime changes (e.g. a screenshot re-taken every
# iteration), the stale entry's temp file is deleted immediately below
# instead of being left to accumulate forever.
_image_cache: "collections.OrderedDict[Tuple[str, str], Tuple[float, str]]" = collections.OrderedDict()
_image_cache_lock = threading.Lock()
# Caps distinct (path, quality) entries -- dynamic files= callables that
# generate a new source path every iteration would otherwise grow this
# process-lifetime global cache and its backing temp files without bound.
_IMAGE_CACHE_MAX_ENTRIES = 256


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTENSIONS


def _detail_for(image_quality: Union[str, int]) -> Optional[str]:
    if isinstance(image_quality, int):
        return "low" if image_quality <= 512 else "auto"
    tier = _IMAGE_QUALITY_TIERS.get(str(image_quality).lower())
    return tier[2] if tier else "auto"


def _preprocess_image(
    path: str,
    image_quality: Union[str, int],
    logger: logging.Logger,
    created_files: Optional[List[str]] = None,
) -> str:
    """Resize and re-encode an image based on image_quality. Results are cached."""
    quality_key = str(image_quality)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return path

    cache_key = (path, quality_key)
    with _image_cache_lock:
        cached = _image_cache.get(cache_key)
        if cached is not None:
            _image_cache.move_to_end(cache_key)
    if cached is not None:
        cached_mtime, cached_path = cached
        if cached_mtime == mtime and os.path.exists(cached_path):
            return cached_path

    if isinstance(image_quality, int):
        jpeg_quality = max(1, min(100, image_quality))
        max_dim      = 512 if image_quality <= 512 else None
    else:
        tier = _IMAGE_QUALITY_TIERS.get(
            str(image_quality).lower(), _IMAGE_QUALITY_TIERS["auto"]
        )
        max_dim, jpeg_quality, _ = tier

    if max_dim is None and jpeg_quality is None:
        return path

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        logger.warning(
            "Pillow is not installed — image resize/recompress skipped. "
            "Install with: pip install 'autourgos-agent[images]'. "
            "The image_detail hint is still applied for OpenAI token savings."
        )
        return path

    try:
        with Image.open(path) as src:
            img = src.convert("RGB")
        if max_dim is not None:
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".jpg", prefix="autourgos_img_", delete=False
        )
        tmp.close()
        img.save(tmp.name, format="JPEG", quality=jpeg_quality, optimize=True)
        evicted: List[str] = []
        discard_own = False
        with _image_cache_lock:
            stale = _image_cache.get(cache_key)
            if stale is not None and stale[0] == mtime:
                # A concurrent call for the exact same (path, quality, mtime)
                # already won this cache slot while we were processing
                # (Image.open/resize/save runs outside this lock, so two
                # threads can race here) -- use its result instead of
                # overwriting it, so a caller that already received and
                # started using the winning path never has it deleted out
                # from under it. Our own duplicate output is discarded below.
                winner_path = stale[1]
                _image_cache.move_to_end(cache_key)
                discard_own = True
            else:
                _image_cache[cache_key] = (mtime, tmp.name)
                _image_cache.move_to_end(cache_key)
                winner_path = tmp.name
            while len(_image_cache) > _IMAGE_CACHE_MAX_ENTRIES:
                _, (_, evicted_path) = _image_cache.popitem(last=False)
                evicted.append(evicted_path)
        if discard_own:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        elif stale is not None and stale[1] != tmp.name:
            try:
                os.remove(stale[1])
            except OSError:
                pass
        for evicted_path in evicted:
            try:
                os.remove(evicted_path)
            except OSError:
                pass
        if created_files is not None and not discard_own:
            created_files.append(tmp.name)
        return winner_path
    except Exception as exc:
        logger.warning(
            f"Image preprocessing failed for {path!r}: {exc}. Using original."
        )
        return path


# ── inline runtime (Agent(pre_iteration_callback=..., pre_iteration_files=...)) ──

class _NullPreIteration:
    """No-op stand-in used when Agent() is built without pre_iteration_callback=/
    pre_iteration_files= -- every call is a cheap no-op, zero overhead."""

    def before_iteration(self, iteration: int, agent: Any = None) -> Optional[Dict[str, Any]]:
        return None

    async def abefore_iteration(self, iteration: int, agent: Any = None) -> Optional[Dict[str, Any]]:
        return None

    def cleanup(self) -> None:
        pass


_NULL_PREITERATION = _NullPreIteration()


class _PreIterationRuntime:
    """
    Inline (non-middleware) per-iteration callback/file injection, built
    directly by Agent(pre_iteration_callback=..., pre_iteration_files=...,
    image_quality=...) -- mirrors AgentLoopMixin's _maybe_summarize/
    _history pattern: this only ever applies to the ONE Agent instance it's
    configured on, so it needs no cross-instance-sharing machinery -- flat
    instance state is safe since Agent's own _run_lock guarantees only one
    run is ever active per instance at a time.
    """

    def __init__(
        self,
        callback: Optional[Callable[[int], Union[None, Awaitable[None]]]],
        files: Optional[
            Union[str, List[str], Callable[[int], Optional[Union[str, List[str]]]]]
        ],
        image_quality: Union[str, int],
    ) -> None:
        if isinstance(image_quality, int):
            if not (1 <= image_quality <= 100):
                raise ValueError("image_quality as int must be between 1 and 100.")
        elif str(image_quality).lower() not in _IMAGE_QUALITY_TIERS:
            raise ValueError(
                f"image_quality must be one of {list(_IMAGE_QUALITY_TIERS)} or int 1-100, "
                f"got {image_quality!r}."
            )
        self.callback = callback
        self._files = files
        self.image_quality = image_quality
        self.logger = logging.getLogger(__name__)
        self._created_temp_files: List[str] = []

    def before_iteration(self, iteration: int, agent: Any = None) -> Optional[Dict[str, Any]]:
        if self.callback:
            try:
                res = self.callback(iteration)
                if inspect.iscoroutine(res):
                    _run_coroutine_sync(res)
            except Exception as exc:
                self.logger.error(
                    f"Error in pre-iteration callback at iteration {iteration}: {exc}"
                )

        if self._files is None:
            return None
        resolved = self._files(iteration) if callable(self._files) else self._files
        if not resolved:
            return None

        raw: List[str] = []
        if isinstance(resolved, list):
            raw = [f for f in resolved if f and os.path.exists(f)]
        elif isinstance(resolved, str) and os.path.exists(resolved):
            raw = [resolved]
        if not raw:
            return None

        processed = [
            _preprocess_image(
                f, self.image_quality, self.logger, created_files=self._created_temp_files,
            )
            if _is_image(f) else f
            for f in raw
        ]
        result: Dict[str, Any] = {"files": processed}
        detail = _detail_for(self.image_quality)
        if detail is not None:
            result["image_detail"] = detail

        narrate_logger = getattr(agent, "logger", None)
        if narrate_logger:
            narrate_logger.middleware(
                "PreIteration", f"Injected {len(processed)} file(s) before iteration {iteration}.",
            )
        return result

    async def abefore_iteration(self, iteration: int, agent: Any = None) -> Optional[Dict[str, Any]]:
        """Async twin of before_iteration -- offloads to a worker thread
        (matching _amaybe_summarize's identical pattern) instead of
        blocking the event loop for the callback/image-preprocessing work."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.before_iteration, iteration, agent)

    def cleanup(self) -> None:
        """Remove temp files created during this run, skipping any file
        still referenced by the shared _image_cache (so a later run with
        the same unchanged image can still hit the cache and reuse it
        without reprocessing)."""
        if not self._created_temp_files:
            return
        with _image_cache_lock:
            live_cached_paths = {v[1] for v in _image_cache.values()}
        for tmp_path in self._created_temp_files:
            if tmp_path in live_cached_paths:
                continue
            try:
                os.remove(tmp_path)
            except Exception:
                self.logger.debug(
                    "Could not remove temp file %r during cleanup", tmp_path, exc_info=True,
                )
        self._created_temp_files = []
