from msmodelslim.model.internvl3_5.loader import InternVL3_5AdapterLoader


def test_adapter_class_path_given_loader_when_called_then_return_expected():
    assert (
        InternVL3_5AdapterLoader.ADAPTER_CLASS_PATH
        == "msmodelslim.model.internvl3_5.model_adapter:InternVL3_5ModelAdapter"
    )


def test_loader_instantiation_given_default_when_called_then_succeed():
    loader = InternVL3_5AdapterLoader()
    assert loader.ADAPTER_CLASS_PATH == "msmodelslim.model.internvl3_5.model_adapter:InternVL3_5ModelAdapter"
