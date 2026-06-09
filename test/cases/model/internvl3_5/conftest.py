# pylint: disable=no-member
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock


_saved_state = {}


def _setup_mocks():
    def _save_and_set(key, value):
        _saved_state[key] = sys.modules.get(key, None)
        sys.modules[key] = value

    if "torchvision" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        torchvision.__spec__ = types.ModuleType("torchvision_spec")
        torchvision.__spec__.name = "torchvision"
        torchvision.__spec__.origin = "mock"
        torchvision.__path__ = []

        torchvision.transforms = types.ModuleType("torchvision.transforms")
        torchvision.transforms.__spec__ = types.ModuleType("torchvision.transforms_spec")
        torchvision.transforms.__spec__.name = "torchvision.transforms"
        torchvision.transforms.__spec__.origin = "mock"
        torchvision.transforms.Compose = MagicMock
        torchvision.transforms.Resize = MagicMock
        torchvision.transforms.ToTensor = MagicMock
        torchvision.transforms.Normalize = MagicMock
        torchvision.transforms.CenterCrop = MagicMock
        torchvision.transforms.Lambda = MagicMock
        torchvision.transforms.RandomResizedCrop = MagicMock
        torchvision.transforms.InterpolationMode = SimpleNamespace(
            NEAREST=0,
            BILINEAR=1,
            BICUBIC=2,
            LANCZOS=3,
            BOX=4,
            HAMMING=5,
        )

        torchvision.io = types.ModuleType("torchvision.io")
        torchvision.io.__spec__ = types.ModuleType("torchvision.io_spec")
        torchvision.io.__spec__.name = "torchvision.io"
        torchvision.io.__spec__.origin = "mock"

        _save_and_set("torchvision", torchvision)
        _save_and_set("torchvision.transforms", torchvision.transforms)
        _save_and_set("torchvision.io", torchvision.io)

    if "torch_npu" not in sys.modules:
        torch_npu_spec = types.ModuleType(name="torch_npu")
        mock_torch_npu = Mock(__spec__=torch_npu_spec)
        _save_and_set("torch_npu", mock_torch_npu)

    import torch

    _saved_npu = getattr(torch, "npu", None)
    if not hasattr(torch, "npu") or isinstance(getattr(torch, "npu", None), Mock):
        mock_npu = Mock()
        mock_npu.is_available = Mock(return_value=False)
        mock_npu.current_device = Mock(return_value=0)
        mock_npu.get_device_name = Mock(return_value="Ascend910")
        torch.npu = mock_npu
        _saved_state["torch.npu"] = _saved_npu

    if "transformers.masking_utils" not in sys.modules:
        masking_utils = types.ModuleType("transformers.masking_utils")
        masking_utils.create_causal_mask = MagicMock()
        _save_and_set("transformers.masking_utils", masking_utils)
        import transformers

        _saved_state["transformers.masking_utils_attr"] = getattr(transformers, "masking_utils", None)
        setattr(transformers, "masking_utils", masking_utils)


def _teardown_mocks():
    for key, original in _saved_state.items():
        if key.endswith("_attr"):
            continue
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original

    if "transformers.masking_utils_attr" in _saved_state:
        try:
            import transformers

            setattr(transformers, "masking_utils", _saved_state["transformers.masking_utils_attr"])
        except ImportError:
            pass

    if "torch.npu" in _saved_state:
        try:
            import torch

            if _saved_state["torch.npu"] is None:
                if hasattr(torch, "npu"):
                    delattr(torch, "npu")
            else:
                torch.npu = _saved_state["torch.npu"]
        except ImportError:
            pass

    _saved_state.clear()


def pytest_configure(config):
    this_conftest = __file__
    if "internvl3_5" in this_conftest:
        _setup_mocks()


def pytest_unconfigure(config):
    if _saved_state:
        _teardown_mocks()
