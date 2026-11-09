# -*- coding: UTF-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

Precheck framework: status models, rule declaration, and execution.
"""

from __future__ import annotations

import collections.abc
import inspect
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    Collection,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from msmodelslim.utils.config import CONFIG_FILE_PATH
from msmodelslim.utils.exception import ModelslimError
from msmodelslim.utils.logging import get_logger


class PrecheckStatus(str, Enum):
    PASS = "PASS"  # nosec B105  # precheck status name, not a credential
    SKIP = "SKIP"
    FAIL = "FAIL"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PrecheckResult:
    group: str
    check_id: str
    status: PrecheckStatus
    reason: str = ""
    suggestion: str = ""
    skippable: bool = False
    config_skipped: bool = False
    error: Optional[BaseException] = None


@dataclass(frozen=True)
class PrecheckReport:
    results: tuple[PrecheckResult, ...]

    @property
    def failed(self) -> bool:
        return any(result.status == PrecheckStatus.FAIL for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "check_id": result.check_id,
                    "group": result.group,
                    "status": result.status.value,
                    "reason": result.reason,
                    "suggestion": result.suggestion,
                }
                for result in self.results
            ]
        }


# Reserved by bind(); check functions must use different parameter names.
_RESERVED_BIND_KWARGS = ("rule_id",)


def _func_name(func: Any) -> str:
    """Return a readable name for error messages."""
    return getattr(func, "__name__", None) or repr(func)


def _default_rule_id(func: Any) -> str:
    """Derive a stable rule id from the callable name."""
    target = func
    while isinstance(target, partial):
        target = target.func
    name = getattr(target, "__name__", None)
    if not name:
        raise TypeError("cannot derive rule_id from %r; specify rule_id=... in @precheck_rule" % (func,))
    return name


class PrecheckRule:
    """A precheck rule with group, id, skip policy, and executable logic.

    Rules can be declared with :func:`precheck_rule` and bound before registration,
    or implemented by subclassing and overriding :meth:`check`.

    ``True``, ``None``, and strings produce PASS; ``False`` produces SKIP;
    exceptions produce FAIL. Optional rules are marked with ``skippable=True``.
    """

    def __init__(
        self,
        check: Optional[Callable[[], Any]] = None,
        *,
        func: Optional[Callable[..., Any]] = None,
        group: Optional[str] = None,
        rule_id: Optional[str] = None,
        skippable: bool = False,
    ) -> None:
        self._check = check
        self._func = func
        self.id = rule_id if rule_id is not None else (_default_rule_id(func) if func is not None else None)
        self.group = group if group is not None else self.id
        self.skippable = skippable

    @property
    def func(self) -> Optional[Callable[..., Any]]:
        """Original function for decorated rules, otherwise ``None``."""
        return self._func

    @property
    def __signature__(self) -> inspect.Signature:
        """Expose the original signature until the rule is bound."""
        if self._func is not None:
            return inspect.signature(self._func)
        return inspect.Signature()

    @property
    def needs_binding(self) -> bool:
        """Return whether a decorated rule still has required arguments."""
        if self._check is not None or self._func is None:
            return False
        required = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        return any(
            param.default is param.empty and param.kind in required
            for param in inspect.signature(self._func).parameters.values()
        )

    def bind(self, *args: Any, rule_id: Optional[str] = None, **kwargs: Any) -> PrecheckRule:
        """Bind runtime arguments and return an executable rule."""
        if self._func is None:
            raise TypeError("only function rules support bind(); subclass rules should take arguments in __init__")
        return PrecheckRule(
            check=_validate_and_bind(self._func, *args, **kwargs),
            group=self.group,
            rule_id=rule_id if rule_id is not None else self.id,
            skippable=self.skippable,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a rule directly using its original function signature."""
        if self._check is not None:
            if args or kwargs:
                raise TypeError("precheck rule %s is already bound; call it without arguments or use check()" % self.id)
            return self.check()
        if self._func is None:
            raise TypeError("subclass rules have no function signature; use check()")
        return _validate_and_bind(self._func, *args, **kwargs)()

    def check(self) -> Any:
        """Run the configured check."""
        if self._check is not None:
            return self._check()
        if self._func is not None:
            return self._func()
        raise NotImplementedError("PrecheckRule.check must be overridden by a subclass or built from a check function")

    def run(self, enabled: bool) -> PrecheckResult:
        """Run an enabled rule, or return a configured SKIP result."""
        if self.skippable and not enabled:
            return PrecheckResult(
                self.group,
                self.id,
                PrecheckStatus.SKIP,
                reason="removed from quant_precheck_item in %s" % CONFIG_FILE_PATH,
                suggestion="add %s to quant_precheck_item in %s to re-enable this check"
                % (self.group, CONFIG_FILE_PATH),
                skippable=True,
                config_skipped=True,
            )
        try:
            passed = self.check()
        except Exception as exc:
            return _exception_result(self, exc)
        # False means SKIP; strings are PASS messages; True and None are silent PASS.
        status = PrecheckStatus.SKIP if passed is False else PrecheckStatus.PASS
        reason = passed if isinstance(passed, str) else ""
        return PrecheckResult(self.group, self.id, status, reason, skippable=self.skippable)


def _matches_annotation(value: Any, annotation: Any) -> bool:
    """Best-effort runtime check of ``value`` against a type annotation.

    Supports plain types, ``Any``, ``Union``/``Optional`` and generic
    containers. Unrecognized annotation forms do not block registration.
    """
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return isinstance(value, annotation) if isinstance(annotation, type) else True
    if origin is Union:
        return any(_matches_annotation(value, option) for option in get_args(annotation))
    if origin in (list, set, frozenset):
        if not isinstance(value, origin):
            return False
        item_types = get_args(annotation)
        if not item_types:
            return True
        return all(_matches_annotation(item, item_types[0]) for item in value)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        item_types = get_args(annotation)
        if not item_types:
            return True
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return all(_matches_annotation(item, item_types[0]) for item in value)
        return len(value) == len(item_types) and all(
            _matches_annotation(item, item_type) for item, item_type in zip(value, item_types)
        )
    if origin is dict or (isinstance(origin, type) and issubclass(origin, collections.abc.Mapping)):
        if not isinstance(value, collections.abc.Mapping):
            return False
        item_types = get_args(annotation)
        if len(item_types) != 2:
            return True
        key_type, value_type = item_types
        return all(
            _matches_annotation(key, key_type) and _matches_annotation(item, value_type) for key, item in value.items()
        )
    if isinstance(origin, type) and issubclass(origin, collections.abc.Callable):
        return callable(value)
    if isinstance(origin, type):
        return isinstance(value, origin)
    return True


def _validate_and_bind(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """Validate call arguments against the signature and bind them."""
    signature = inspect.signature(func)
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError("check function %s argument count mismatch: %s" % (_func_name(func), exc)) from exc
    bound.apply_defaults()

    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}
    for name, value in bound.arguments.items():
        annotation = type_hints.get(name)
        if annotation is None:
            continue
        if not _matches_annotation(value, annotation):
            raise TypeError(
                "check function %s argument %s type mismatch: expected %s, got %s"
                % (_func_name(func), name, annotation, type(value).__name__)
            )
    return partial(func, *args, **kwargs)


def _exception_result(rule: PrecheckRule, exc: Exception) -> PrecheckResult:
    # Keep ModelslimError's raw message separate from its multiline action.
    reason = Exception.__str__(exc) if isinstance(exc, ModelslimError) else str(exc)
    suggestion = getattr(exc, "action", "")
    if reason == suggestion:
        reason = ""
    return PrecheckResult(
        rule.group,
        rule.id,
        PrecheckStatus.FAIL,
        reason,
        suggestion,
        skippable=rule.skippable,
        error=exc,
    )


def precheck_rule(
    func: Optional[Callable[..., Any]] = None,
    /,
    *,
    group: Optional[str] = None,
    rule_id: Optional[str] = None,
    skippable: bool = False,
) -> Union[PrecheckRule, Callable[[Callable[..., Any]], PrecheckRule]]:
    """Declare a function as a :class:`PrecheckRule`.

    ::

        @precheck_rule(group="disk", rule_id="disk", skippable=True)
        def check_disk_space(model_path, save_path) -> Union[bool, str]:
            ...

        manager.register(check_disk_space.bind(model_path, save_path))
    """

    def decorator(check_func: Callable[..., Any]) -> PrecheckRule:
        # Reject collisions with bind(rule_id=...) while the rule is declared.
        params = inspect.signature(check_func).parameters
        for reserved in _RESERVED_BIND_KWARGS:
            if reserved in params:
                raise TypeError(
                    "check function %s uses the reserved parameter name %s; rename it (reserved for PrecheckRule.bind)"
                    % (_func_name(check_func), reserved)
                )
        return PrecheckRule(func=check_func, group=group, rule_id=rule_id, skippable=skippable)

    return decorator(func) if func is not None else decorator


class PrecheckManager:
    """Register precheck rules and execute them in order."""

    def __init__(self):
        self._rules: list[PrecheckRule] = []

    def register(self, rule: PrecheckRule, /) -> None:
        """Register a :class:`PrecheckRule`."""
        if not isinstance(rule, PrecheckRule):
            raise TypeError(
                "register expects a PrecheckRule; decorate the check function with @precheck_rule "
                "and bind(...) runtime arguments first"
            )
        if rule.needs_binding:
            raise TypeError("precheck rule %s needs arguments; call bind(...) before register" % rule.id)
        if rule.func is not None and inspect.signature(rule.func).parameters:
            # Default-only parameters are executable but often indicate a missing bind().
            get_logger().warning(
                "precheck rule %s registered without bind(); it will run with the function defaults", rule.id
            )
        if any(registered.id == rule.id for registered in self._rules):
            get_logger().warning("Duplicate precheck rule id: %s", rule.id)
            return
        self._rules.append(rule)

    def run(self, configured_items: Collection[str] = ()) -> PrecheckReport:
        enabled = set(configured_items)
        results = [
            rule.run(enabled=(not rule.skippable) or rule.id in enabled or rule.group in enabled)
            for rule in self._rules
        ]
        return PrecheckReport(tuple(results))


# Reporting is application policy: each caller renders a PrecheckReport in the way
# its own users expect (log lines, exception type, exit behaviour), so the
# framework only produces data and never prints or raises by itself.
