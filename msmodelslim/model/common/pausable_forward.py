#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Turn a model's **native** ``forward`` into a *pausable* inference process.

Motivation
----------
``PipelineInterface.generate_model_forward`` must return a
``Generator[ProcessRequest, Any, None]`` that walks the model block by block so
that a calibration processor can observe (and replace) each block's output.

Hand-writing such a generator is error prone: you end up re-implementing the
model's inter-block logic (rotary embeddings, attention masks, multi-modal
feature fusion, deepstack injection, ...). :func:`pausable_forward` avoids this
by running a caller-provided ``forward_func`` on a background thread and pausing
the forward, via pre-hooks, right before each matched block of interest.

Control flow (single calibration sample)::

    gen = pausable_forward(forward_func, bindings)
    req = next(gen)                 # worker paused before the first matched block
    out = run_processors(req)       # main thread calibrates the block
    req = gen.send(out)             # inject ``out`` as that block's output, resume

When ``send(out)`` is called the paused block's ``forward`` is short-circuited to
return ``out`` (aligned to the block's device/dtype). Because the *native*
forward continues afterwards, the next block receives ``out`` through the real
model code path -- exactly as it would during ordinary inference.

Two independent controls per matched point
------------------------------------------
Each ``bindings[root][point_spec]`` entry is a :class:`PointConfig`:

* ``handler`` -- a side-effect callback invoked on the main thread right before
  the block yields (or, for non-yielding points, right before it runs natively).
  Its return value is ignored; typical uses are materialising a ``meta``
  placeholder's weights in place, MOE unpacking a fused expert container in
  place, logging, or offloading a layer back to ``meta`` after use.
* ``should_yield`` -- a predicate deciding whether the point suspends the
  forward and hands a :class:`~msmodelslim.core.base.protocol.ProcessRequest`
  to the consumer (``send`` value becomes the block's output) or lets the block
  run natively.

Residency contract
------------------
The whole ``forward`` runs in one shot, so **every module matched by a point
spec must already exist in its root's module tree when the generator starts** (a
*growing* ``ModuleList`` cannot be iterated -- ``append`` mid-forward raises
``RuntimeError: dictionary changed size during iteration``).

Lazy, memory-efficient loading is still possible with the *placeholder + in
place* pattern: build the full-size ``ModuleList`` up front with cheap ``meta``
placeholders (stable size), then materialise each block's weights **in place**
via a ``handler`` right before it is processed, and offload it again afterwards.
Because ``send()`` short-circuits the block's ``forward``, the worker never
executes the block's weights -- they only need to be real while the processor
calibrates the block on the main thread.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.utils.exception import InvalidModelError, ModelslimTimeoutError
from msmodelslim.utils.logging import get_logger

# The pause handshake carries NO wall-clock deadline: a calibration consumer may
# legitimately spend minutes on one block (AWQ scale search, smoothing), so a
# fixed net turns "still working" into a spurious ModelslimTimeoutError.
# Liveness is detected structurally instead -- the worker always publishes
# _SENTINEL from its ``finally``, and a paused worker leaves through _stopping
# on the next poll.
# Poll interval used while waiting for a consumer decision. Small so teardown
# can wake a paused worker promptly even if the worker is mid-handshake.
_RESUME_POLL_S = 0.1
# Teardown still joins the worker with a bound: a thread wedged inside a kernel
# call would otherwise hang process exit. A stop-path bound, not a work-path
# deadline.
_JOIN_TIMEOUT_S = 30.0

_SENTINEL = object()
_NO_SKIP = object()
_FORWARD_PATCH_STATE = "_pausable_forward_patch_state"
_PATCH_LOCK = threading.RLock()

# Skip values are stored per worker-thread so several generators (one per calib
# sample) can drive the same modules concurrently without clobbering each other.
_SKIP_LOCAL = threading.local()


class _StopForward(Exception):
    """Raised inside the worker to unwind the native forward when we stop early.

    Once the consumer stops pulling blocks (generator closed / all needed blocks
    seen), there is no point running the model tail (final norm / lm_head): it
    wastes compute and may hit dtype/device mismatches now that layers have been
    offloaded. We unwind the paused worker cleanly instead of resuming it.
    """


@dataclass
class _ForwardPatchState:
    original: Callable[..., Any]
    owners: Set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Public matching / callback types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathSpec:
    """Glob matched against a module's qualified path (relative to its root).

    ``fnmatch`` semantics (``*`` ``?`` ``[seq]``); ``*`` spans dots, so
    ``"language_model.layers.*"`` also matches deeper submodules. A bare
    ``str`` key in ``bindings`` is shorthand for ``PathSpec``.
    """

    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern.strip():
            raise ValueError("PathSpec.pattern must be a non-empty string")


@dataclass(frozen=True)
class ClassSpec:
    """Glob matched against a module's class name.

    ``fnmatch`` semantics (``*`` ``?`` ``[seq]``), e.g. ``"*DecoderLayer"``.
    Class names are matched by the module's ``type(module).__name__``, so
    several structurally identical blocks share one spec without caring about
    their individual paths.
    """

    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern.strip():
            raise ValueError("ClassSpec.pattern must be a non-empty string")


# A point spec is one of:
#   * an ``nn.Module`` subclass -- matched by ``isinstance`` (covers subclasses);
#   * a :class:`PathSpec` -- glob-matched against the module's qualified path;
#   * a :class:`ClassSpec` -- glob-matched against the module's class name;
#   * a bare ``str`` -- shorthand for :class:`PathSpec`.
PointSpec = Union[Type[nn.Module], PathSpec, ClassSpec, str]


@dataclass(frozen=True)
class PausePoint:
    """Live inputs captured at a matched point (before the block runs)."""

    root: nn.Module
    name: str
    module: nn.Module
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    occurrence: int


Handler = Callable[[PausePoint], None]


def _always_yield(_point: PausePoint) -> bool:
    return True


@dataclass(frozen=True)
class PointConfig:
    """Per-point controls: an optional side-effect callback and the yield policy.

    ``handler`` runs on the main thread; its return value is ignored. Blocks
    matched with ``should_yield`` true suspend the forward and hand the
    consumer a :class:`ProcessRequest` (whose ``send`` value becomes that
    block's output). Blocks with ``should_yield`` false run natively (after the
    ``handler`` has had a chance to prepare them).
    """

    handler: Optional[Handler] = None
    should_yield: Callable[[PausePoint], bool] = _always_yield

    def __post_init__(self) -> None:
        if self.handler is not None and not callable(self.handler):
            raise ValueError("PointConfig.handler must be callable or None")
        if not callable(self.should_yield):
            raise ValueError("PointConfig.should_yield must be callable")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class _RootBindings:
    root: nn.Module
    specs: List[Tuple[PointSpec, PointConfig]]


class _Engine:
    """Runs ``forward_func`` on a worker thread, pausing at every matched point.

    Instances are single-use: call :meth:`drive` exactly once.
    """

    def __init__(
        self,
        forward_func: Callable[[], Any],
        bindings: Dict[nn.Module, Dict[PointSpec, PointConfig]],
    ):
        if not callable(forward_func):
            raise ValueError("forward_func must be callable")
        if not isinstance(bindings, dict) or not bindings:
            raise InvalidModelError("bindings must be a non-empty dict of module -> point configs")
        blocks = []
        for root, spec_configs in bindings.items():
            if not isinstance(root, nn.Module):
                raise InvalidModelError("bindings keys must be nn.Module instances")
            if not isinstance(spec_configs, dict) or not spec_configs:
                raise InvalidModelError(f"bindings for module {root.__class__.__name__} must be a non-empty dict")
            for spec, config in spec_configs.items():
                if isinstance(spec, str) and not spec.strip():
                    raise InvalidModelError("point spec strings must be non-empty path patterns")
                if not isinstance(spec, (str, PathSpec, ClassSpec, type)):
                    raise InvalidModelError(
                        f"invalid point spec {spec!r}: expected nn.Module, PathSpec, ClassSpec or str"
                    )
                if isinstance(spec, type) and not issubclass(spec, nn.Module):
                    raise InvalidModelError(f"point spec {spec!r} must be an nn.Module subclass")
                if not isinstance(config, PointConfig):
                    raise InvalidModelError(f"point config for {spec!r} must be a PointConfig")
            blocks.append(_RootBindings(root, list(spec_configs.items())))
        self._roots = blocks
        self._forward = forward_func

        self._queue: "queue.Queue[Tuple[PausePoint, PointConfig]]" = queue.Queue()
        self._resume = threading.Event()
        self._handles: List[RemovableHandle] = []
        self._patched_modules: List[nn.Module] = []
        self._feedback: Any = None
        self._replace_feedback = False
        self._occurrences: Dict[Tuple[int, str], int] = {}
        self._worker: Optional[threading.Thread] = None
        self._worker_ident: Optional[int] = None
        self._worker_error: Optional[BaseException] = None
        self._worker_output: Any = None
        self._grad_enabled = True
        self._running = False
        self._stopping = False

    # -- public entry -------------------------------------------------------

    def drive(self) -> Generator[ProcessRequest, Any, None]:
        """Return the pausable-forward generator (the engine's main loop).

        When the forward completes, the engine's :attr:`_worker_output` (the
        return value of the caller's ``forward_func``) is handed back as the
        generator's ``StopIteration.value``, so the final inference result
        (e.g. logits) is not lost.
        """
        try:
            self._start()
            while True:
                # Blocking, never bounded: the worker publishes ``_SENTINEL``
                # from its ``finally`` on every exit path, so this returns
                # even when the worker dies.
                item = self._queue.get()
                if item is _SENTINEL:
                    if self._worker_error is not None:
                        raise self._worker_error
                    break
                point, config = item

                if config.handler is not None:
                    config.handler(point)
                if config.should_yield(point):
                    request = ProcessRequest(
                        name=point.name,
                        module=point.module,
                        args=point.args,
                        kwargs=point.kwargs,
                    )
                    decision = yield request
                    self._resume_for(decision)
                else:
                    self._resume_native()
            return self._worker_output
        finally:
            self._teardown()

    # -- worker -------------------------------------------------------------

    def _worker_main(self) -> None:
        self._worker_ident = threading.get_ident()
        try:
            with torch.set_grad_enabled(self._grad_enabled):
                self._worker_output = self._forward()
        except _StopForward:
            pass  # intentional early unwind; not an error
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
            self._worker_error = exc
            get_logger().error("Forward failed in pausable_forward: %s", exc)
        finally:
            self._queue.put(_SENTINEL)
            self._worker_ident = None

    # -- lifecycle ----------------------------------------------------------

    def _start(self) -> None:
        self._running = True
        self._worker_error = None
        self._stopping = False
        self._occurrences.clear()
        for root_bindings in self._roots:
            for name, module in root_bindings.root.named_modules():
                if not name:
                    continue
                if self._match(root_bindings, name, module) is None:
                    continue
                self._attach_hook(root_bindings, name, module)
        self._grad_enabled = torch.is_grad_enabled()
        self._resume.set()
        self._worker = threading.Thread(
            target=self._worker_main,
            name="PausableForward",
            daemon=True,
        )
        self._worker.start()

    def _teardown(self) -> None:
        # Signal a paused worker to unwind instead of running the model tail.
        self._stopping = True
        self._resume.set()
        worker_timeout = False
        if self._worker is not None:
            if self._worker is not threading.current_thread():
                self._worker.join(timeout=_JOIN_TIMEOUT_S)
                worker_timeout = self._worker.is_alive()
            self._worker = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module in self._patched_modules:
            self._release_patch(module)
        self._patched_modules.clear()
        self._feedback = None
        self._replace_feedback = False
        self._running = False
        if worker_timeout:
            raise ModelslimTimeoutError(f"pausable-forward worker did not stop within join_timeout_s={_JOIN_TIMEOUT_S}")

    def _resume_for(self, decision: Any) -> None:
        """Resume a paused block from a consumer decision (``None`` = native)."""
        if decision is None:
            self._resume_native()
            return
        self._feedback = decision
        self._replace_feedback = True
        self._resume.set()

    def _resume_native(self) -> None:
        """Resume a paused block, letting its (possibly modified) forward run."""
        self._feedback = None
        self._replace_feedback = False
        self._resume.set()

    # -- hooks --------------------------------------------------------------

    def _match(self, root_bindings: _RootBindings, name: str, module: nn.Module) -> Optional[PointConfig]:
        for spec, config in root_bindings.specs:
            if isinstance(spec, str):
                if fnmatchcase(name, spec):
                    return config
            elif isinstance(spec, PathSpec):
                if fnmatchcase(name, spec.pattern):
                    return config
            elif isinstance(spec, ClassSpec):
                if fnmatchcase(module.__class__.__name__, spec.pattern):
                    return config
            elif isinstance(spec, type) and issubclass(spec, nn.Module):
                if isinstance(module, spec):
                    return config
        return None

    def _attach_hook(self, root_bindings: _RootBindings, name: str, module: nn.Module) -> None:
        self._patch_forward(module)
        handle = module.register_forward_pre_hook(
            self._make_hook(root_bindings, name),
            with_kwargs=True,
        )
        self._handles.append(handle)

    def _patch_forward(self, module: nn.Module) -> None:
        with _PATCH_LOCK:
            state: Optional[_ForwardPatchState] = getattr(module, _FORWARD_PATCH_STATE, None)
            if state is None:
                original = module.forward

                def forward(*f_args: Any, **f_kwargs: Any) -> Any:
                    skip = _pop_skip(module)
                    if skip is not _NO_SKIP:
                        return skip
                    return original(*f_args, **f_kwargs)

                state = _ForwardPatchState(original=original)
                module.forward = forward
                setattr(module, _FORWARD_PATCH_STATE, state)
            state.owners.add(id(self))
            self._patched_modules.append(module)

    def _release_patch(self, module: nn.Module) -> None:
        with _PATCH_LOCK:
            state: Optional[_ForwardPatchState] = getattr(module, _FORWARD_PATCH_STATE, None)
            if state is None:
                return
            state.owners.discard(id(self))
            if state.owners:
                return
            module.forward = state.original
            delattr(module, _FORWARD_PATCH_STATE)

    def _make_hook(self, root_bindings: _RootBindings, name: str):
        def hook(
            module: nn.Module,
            hook_args: Tuple[Any, ...],
            hook_kwargs: Dict[str, Any],
        ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            if threading.get_ident() != self._worker_ident:
                return hook_args, hook_kwargs
            # The matcher ran while registering hooks; look the config up again
            # here so occurrence counting stays local to the worker.
            config = self._match(root_bindings, name, module)
            if config is None:
                return hook_args, hook_kwargs

            occurrence = self._occurrences.get((id(root_bindings.root), name), 0)
            self._occurrences[(id(root_bindings.root), name)] = occurrence + 1
            point = PausePoint(
                root=root_bindings.root,
                name=name,
                module=module,
                args=tuple(hook_args),
                kwargs=dict(hook_kwargs),
                occurrence=occurrence,
            )
            # Clear the latch *before* announcing so the main thread can never
            # resume us before we start waiting (its set() happens only after it
            # observes this point, i.e. strictly after put()).
            self._resume.clear()
            self._queue.put((point, config))
            while not self._resume.wait(timeout=_RESUME_POLL_S):
                if self._stopping:
                    raise _StopForward()
                # No deadline: wait for the consumer however long it needs.

            if self._stopping:
                raise _StopForward()

            feedback = self._feedback
            self._feedback = None
            if self._replace_feedback:
                # Align to the block *input* (the residual stream), not the
                # block's own params: the module may already be offloaded (e.g.
                # to ``meta``) by the time we resume, whereas the input tensor
                # always reflects the device/dtype the forward is running on.
                reference = _reference_tensor(point.args, point.kwargs)
                _set_skip(module, _align_output(reference, feedback))
            self._replace_feedback = False
            return point.args, point.kwargs

        return hook


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def pausable_forward(
    forward_func: Callable[[], Any],
    bindings: Dict[nn.Module, Dict[PointSpec, PointConfig]],
) -> Generator[ProcessRequest, Any, None]:
    """Run ``forward_func`` and pause the forward at every matched point.

    ``forward_func`` decides **how** the forward runs and where its inputs come
    from (a closure over the caller's tensors); it may assemble any number of
    modules into a single inference. ``bindings`` maps each module to insert
    hooks into (the "root") to an ordered ``{point spec: PointConfig}`` map.

    Yields a :class:`ProcessRequest` at each point whose config says
    ``should_yield``. ``send(value)`` feeds ``value`` back as that block's
    output (aligned and short-circuited into the forward); ``send(None)`` (the
    initial prime, or an explicit later decision) lets the block run natively.

    When the forward finishes, its final result is delivered as the
    generator's return value -- read it from ``StopIteration.value``::

        gen = pausable_forward(forward_func, bindings)
        request = next(gen)
        try:
            while True:
                out = run_processors(request)
                request = gen.send(out)
        except StopIteration as done:
            final_output = done.value   # forward_func's return (e.g. logits)

    Examples::

        gen = pausable_forward(
            forward_func=lambda: model.model(input_ids=sample["input_ids"], ...),
            bindings={
                model.model: {
                    ClassSpec("*DecoderLayer"): PointConfig(
                        handler=lambda pt: adapter.materialize(pt.module),
                    ),
                    PathSpec("lm_head"): PointConfig(should_yield=lambda pt: False),
                },
            },
        )
    """
    return _Engine(forward_func, bindings).drive()


def _reference_tensor(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """First tensor among the block inputs -- defines the residual-stream placement."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
    return None


def _align_output(reference: Optional[torch.Tensor], value: Any) -> Any:
    if reference is None:
        return value

    def align(t: torch.Tensor) -> torch.Tensor:
        # Keep integer/bool tensors' dtype; only realign floating dtype.
        dtype = reference.dtype if t.is_floating_point() and reference.is_floating_point() else t.dtype
        return t.to(device=reference.device, dtype=dtype)

    def align_tree(node: Any) -> Any:
        if isinstance(node, torch.Tensor):
            return align(node)
        if isinstance(node, (tuple, list)):
            return type(node)(align_tree(x) for x in node)
        return node

    return align_tree(value)


def _set_skip(module: nn.Module, value: Any) -> None:
    store: Dict[int, Any] = getattr(_SKIP_LOCAL, "outputs", None)
    if store is None:
        store = {}
        _SKIP_LOCAL.outputs = store
    store[id(module)] = value


def _pop_skip(module: nn.Module) -> Any:
    store: Dict[int, Any] = getattr(_SKIP_LOCAL, "outputs", None)
    if not store:
        return _NO_SKIP
    return store.pop(id(module), _NO_SKIP)
