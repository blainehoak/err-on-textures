import torchvision as tv
import glob


def load_model_by_name(model_name, threat_model=None):
    """
    Load a model and its corresponding transforms/preprocessing pipeline.

    Handles models from torchvision, RobustBench, and Hugging Face.

    Args:
        model_name (str): The name of the model to load.
        threat_model (str, optional): Required for RobustBench models. Specifies the threat model.

    Returns:
        model: The loaded model.
        model_transforms: The preprocessing pipeline for the model.
    """
    try:
        # Try to load a torchvision model
        model, model_transforms = load_torchvision_model(model_name)
    except ValueError:
        try:
            model, model_transforms = load_robustbench_model(model_name, threat_model)
        except ValueError:
            try:
                model, model_transforms = load_timm_model(model_name)
            except ValueError:
                model, model_transforms = load_huggingface_model(model_name)
    model.eval()
    return model, model_transforms


def load_timm_model(model_name):
    from timm import create_model
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    # loading any timm model
    model = create_model(model_name, pretrained=True)

    # processing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def load_torchvision_model(model_name):
    weights = tv.models.get_model_weights(model_name).DEFAULT
    model_transforms = weights.transforms()
    model = getattr(tv.models, model_name)(weights=weights)
    return model, model_transforms


def load_huggingface_model(model_name):
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, HFTransformAdapter(processor)


def load_robustbench_model(model_name, threat_model=None):
    import robustbench

    if threat_model is None:
        model_options = glob.glob(f"models/imagenet/*/{model_name}.pt")
        threat_model = (
            model_options[0].split("/")[-2] if len(model_options) > 0 else None
        )
    model = robustbench.utils.load_model(
        model_name, dataset="imagenet", threat_model=threat_model
    )
    if threat_model == "Linf":
        preproc_name = robustbench.model_zoo.imagenet.linf[model_name]["preprocessing"]
    elif threat_model == "corruptions":
        preproc_name = robustbench.model_zoo.imagenet.common_corruptions[model_name][
            "preprocessing"
        ]
    model_transforms = robustbench.data.PREPROCESSINGS[preproc_name]
    return model, model_transforms


class HFTransformAdapter:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        proc_image = self.processor(image, return_tensors="pt")
        return proc_image["pixel_values"].squeeze(0)
