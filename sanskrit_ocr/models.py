import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class BaselineCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128),
                                        nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, classes, pretrained=False):
    if name == "cnn":
        return BaselineCNN(classes)
    if name != "transfer":
        raise ValueError(f"Unknown model {name}")
    # No silent fallback: a failed weight download must stop transfer training.
    try:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    except Exception as error:
        raise RuntimeError("Pretrained MobileNetV3 weights unavailable. Enable the download or populate TORCH_HOME/hub/checkpoints/mobilenet_v3_small-047dcff4.pth. No random-weight transfer experiment was run.") from error
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    return model


def set_transfer_stage(model, fine_tune_last_blocks=0):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if fine_tune_last_blocks:
        for block in list(model.features.children())[-fine_tune_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def train_mode(model, name):
    model.train()
    if name == "transfer":
        # Freeze running statistics even in fine-tuning: DHCD is unlike ImageNet.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"schema_version": 1, "state_dict": state, **metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    # These project checkpoints contain tensors and primitive metadata only.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    from .labels import validate_mapping
    validate_mapping(checkpoint["mapping"])
    model = build_model(checkpoint["model_name"], len(checkpoint["mapping"]["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
