#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Unit tests for msmodelslim.model.step3_7_flash.loader.

Mirrors the structure of qwen3_vl_moe's adapter loader test coverage:
- ADAPTER_CLASS_PATH contract (module path : class name).
- Inheritance from BaseModelAdapterLoader.
- precheck() behaviour (dependency requirement discovery, failure bookkeeping).
- load() behaviour (resolve adapter class, instantiate it with model_path / model_type / trust_remote_code).

Each test follows the project naming convention:
    test_<function>_given_<input>_when_<condition>_then_<expected_outcome>
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader
from msmodelslim.model.step3_7_flash.loader import Step3_7FlashAdapterLoader
from msmodelslim.utils.exception import VersionError


LOADER_PATH = "msmodelslim.model.step3_7_flash.loader"
# All precheck / load helpers are defined in base_loader.py and accessed
# via the parent class — patch them at their source module, not at the
# thin Step3_7FlashAdapterLoader re-export shim.
BASE_LOADER_PATH = "msmodelslim.model.plugin_factory.base_loader"


class TestStep3_7FlashAdapterLoaderContract(unittest.TestCase):
    """Verify the static contract of Step3_7FlashAdapterLoader itself."""

    def test_class_given_loader_then_subclass_of_base_model_adapter_loader(self):
        """The loader class must inherit from BaseModelAdapterLoader so the
        plugin factory can call precheck() / load() on it uniformly.
        """
        self.assertTrue(issubclass(Step3_7FlashAdapterLoader, BaseModelAdapterLoader))

    def test_class_given_loader_then_has_distinct_class_name(self):
        """The class must be exported under its own name (not aliased to the
        base class), so setuptools entry-point strings resolve unambiguously.
        """
        self.assertEqual(Step3_7FlashAdapterLoader.__name__, "Step3_7FlashAdapterLoader")

    def test_adapter_class_path_given_default_value_then_matches_module_dot_class_format(self):
        """ADAPTER_CLASS_PATH must follow the 'module.path:ClassName' format
        that BaseModelAdapterLoader.load() splits on.
        """
        path = Step3_7FlashAdapterLoader.ADAPTER_CLASS_PATH
        self.assertIsInstance(path, str)
        self.assertIn(":", path)
        module_part, class_part = path.split(":", 1)
        self.assertTrue(module_part.startswith("msmodelslim."))
        self.assertTrue(class_part)  # non-empty after the colon

    def test_adapter_class_path_given_default_value_then_points_to_step3_7_flash_model_adapter(self):
        """The module part must point at our model_adapter submodule."""
        module_part, _ = Step3_7FlashAdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
        self.assertEqual(module_part, "msmodelslim.model.step3_7_flash.model_adapter")

    def test_adapter_class_path_given_default_value_then_targets_class_step3_7_flash_model_adapter(self):
        """The class part must match the adapter class exported by model_adapter.py."""
        _, class_part = Step3_7FlashAdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
        self.assertEqual(class_part, "Step3_7FlashModelAdapter")

    def test_adapter_class_path_given_default_value_then_resolves_to_importable_class(self):
        """End-to-end: the ADAPTER_CLASS_PATH must point at a class that
        actually exists in the live Python module graph.
        """
        module_part, class_part = Step3_7FlashAdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)
        import importlib

        adapter_module = importlib.import_module(module_part)
        adapter_class = getattr(adapter_module, class_part)
        self.assertTrue(callable(adapter_class))


class TestStep3_7FlashAdapterLoaderInstance(unittest.TestCase):
    """Behavioural tests against a live Step3_7FlashAdapterLoader() instance."""

    def setUp(self):
        self.loader = Step3_7FlashAdapterLoader()
        self.model_path = Path(tempfile.mkdtemp())
        self.model_type = "step3_7_flash"

    def test_init_given_default_then_inherits_base_state(self):
        """__init__ (inherited) must set _is_match=True and an empty
        requirement dict, matching BaseModelAdapterLoader's contract.
        """
        self.assertTrue(self.loader._is_match)
        self.assertEqual(self.loader._requirements, {})
        self.assertEqual(self.loader._failed_requirements, {})

    def test_get_loader_requirements_given_default_then_returns_dict(self):
        """The inherited get_loader_requirements must return a dict (possibly
        empty) without raising — exact contents come from get_require_packages.
        """
        reqs = self.loader.get_loader_requirements()
        self.assertIsInstance(reqs, dict)

    # ----- precheck -----

    def test_precheck_given_empty_requirements_when_called_then_keeps_is_match_true(self):
        """When dependency requirements are met (or empty), precheck must leave
        _is_match=True so BaseModelAdapterLoader.load() takes the normal path.
        """
        with patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value={}):
            self.loader.precheck(model_type=self.model_type, model_path=self.model_path)

        self.assertTrue(self.loader._is_match)
        self.assertEqual(self.loader._failed_requirements, {})

    def test_precheck_given_failing_version_check_when_called_then_records_failure(self):
        """If a listed package fails version check, precheck must flip
        _is_match to False and remember the failed package.

        check_requirements only catches VersionError (not generic Exception),
        so we must raise VersionError from the dependency checker mock.
        """
        failing_requirements = {"transformers": "==99.99.99"}

        with (
            patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value=failing_requirements),
            patch(
                f"{BASE_LOADER_PATH}.DependencyChecker._check_single",
                side_effect=VersionError("transformers==99.99.99"),
            ),
        ):
            self.loader.precheck(model_type=self.model_type, model_path=self.model_path)

        self.assertFalse(self.loader._is_match)
        # The failed package is remembered for downstream messaging.
        self.assertIn("transformers", self.loader._failed_requirements)

    def test_precheck_given_empty_path_when_called_then_does_not_raise(self):
        """precheck must not blow up if model_path is empty / non-existent —
        BaseModelAdapterLoader.precheck() only uses it for logging context.
        """
        with patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value={}):
            try:
                self.loader.precheck(model_type=self.model_type, model_path=Path("/"))
            except Exception as e:  # pragma: no cover - defensive guard
                self.fail(f"precheck raised unexpectedly: {e}")

    def test_precheck_given_model_type_when_called_then_uses_correct_plugin_name(self):
        """The plugin_name derived from model_type must flow through to
        DependencyChecker.set_plugin — verify by intercepting the call.
        """
        captured = {}

        def fake_set_plugin(plugin_name, requirements):
            captured["plugin_name"] = plugin_name
            captured["requirements"] = requirements

        with (
            patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value={}),
            patch(f"{BASE_LOADER_PATH}.DependencyChecker.set_plugin", side_effect=fake_set_plugin),
        ):
            self.loader.precheck(model_type=self.model_type, model_path=self.model_path)

        self.assertEqual(
            captured["plugin_name"],
            f"msmodelslim.model_adapter.plugins:{self.model_type}",
        )

    # ----- load -----

    def test_load_given_mock_adapter_when_dependencies_match_then_resolves_class_and_calls_it(self):
        """When _is_match=True and get_require_packages returns {}, load()
        must import the module referenced in ADAPTER_CLASS_PATH, look up the
        class attribute, call it with the right kwargs, and return the result.

        We mock import_module and ADAPTER_CLASS_PATH so the test stays
        decoupled from the real Step3_7FlashModelAdapter (which would try
        to read a non-existent config.json from the empty tmpdir).
        """
        # ``Cls`` is the stand-in for ``Step3_7FlashModelAdapter``; calling
        # it returns another MagicMock (load() returns whatever that yields).
        adapter_factory = MagicMock(name="AdapterFactory")
        constructed_instance = MagicMock(name="ConstructedAdapter")
        adapter_factory.return_value = constructed_instance
        mock_module = MagicMock(Cls=adapter_factory)

        with (
            patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value={}),
            patch(f"{BASE_LOADER_PATH}.import_module") as import_module_mock,
            patch.object(Step3_7FlashAdapterLoader, "ADAPTER_CLASS_PATH", new="dummy.mod:Cls"),
        ):
            import_module_mock.return_value = mock_module
            self.loader._is_match = True
            result = self.loader.load(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=True,
            )

        # import_module is called once to resolve the adapter module.
        import_module_mock.assert_called_once_with("dummy.mod")
        # The class returned from the module was instantiated with our kwargs.
        adapter_factory.assert_called_once_with(
            model_type=self.model_type,
            model_path=self.model_path,
            trust_remote_code=True,
        )
        # load() returns whatever the class's constructor returns.
        self.assertIs(result, constructed_instance)

    def test_load_given_mock_adapter_when_dependencies_match_then_passes_through_trust_remote_code(self):
        """trust_remote_code=True/False must reach the adapter's __init__."""
        adapter_factory = MagicMock(name="AdapterFactory")
        mock_module = MagicMock(Cls=adapter_factory)

        with (
            patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value={}),
            patch(f"{BASE_LOADER_PATH}.import_module") as import_module_mock,
            patch.object(Step3_7FlashAdapterLoader, "ADAPTER_CLASS_PATH", new="dummy.mod:Cls"),
        ):
            import_module_mock.return_value = mock_module
            self.loader._is_match = True
            self.loader.load(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=False,
            )

        # The adapter class was called with the trust_remote_code kw
        adapter_factory.assert_called_once_with(
            model_type=self.model_type,
            model_path=self.model_path,
            trust_remote_code=False,
        )

    def test_load_given_malformed_class_path_when_called_then_raises_unsupported_error(self):
        """An ADAPTER_CLASS_PATH without a colon must raise UnsupportedError
        — base loader contract: 'module.path:ClassName' is mandatory.
        """
        with patch.object(Step3_7FlashAdapterLoader, "ADAPTER_CLASS_PATH", new="nocolon"):
            from msmodelslim.utils.exception import UnsupportedError

            with self.assertRaises(UnsupportedError):
                self.loader.load(
                    model_type=self.model_type,
                    model_path=self.model_path,
                )

    def test_load_given_real_class_path_when_adapter_init_mocked_then_returns_real_instance(self):
        """End-to-end: with the real ADAPTER_CLASS_PATH and a mocked
        Step3_7FlashModelAdapter.__init__, the loader must still return a
        real instance whose pedigree / type match what the adapter advertises.
        """
        from msmodelslim.model.step3_7_flash import model_adapter as model_adapter_mod

        with (
            patch(f"{BASE_LOADER_PATH}.get_require_packages", return_value={}),
            patch.object(
                model_adapter_mod.Step3_7FlashModelAdapter,
                "__init__",
                return_value=None,
            ),
        ):
            self.loader._is_match = True
            result = self.loader.load(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=False,
            )

        self.assertIsInstance(result, model_adapter_mod.Step3_7FlashModelAdapter)


if __name__ == "__main__":
    unittest.main()
