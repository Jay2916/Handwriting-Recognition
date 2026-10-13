import torch

from .common import device_for
from .models import load_checkpoint
from .preprocess import image_tensor, prepare_image
from .photo import prepare_photo


class Predictor:
    def __init__(self, checkpoint_path, device="auto"):
        self.device = device_for(device)
        self.model, self.metadata = load_checkpoint(checkpoint_path, self.device)

    def _input_image(self, image, input_mode):
        if input_mode == "auto":
            input_mode = "dataset" if image.size == (32, 32) else "photo"
        if input_mode in ("photo", "cropped_photo"):
            return prepare_photo(image, self.metadata["preprocessing"]["size"],
                                 self.metadata["preprocessing"]["target_polarity"],
                                 isolate=input_mode == "photo")
        if input_mode != "dataset":
            raise ValueError("Unsupported input mode")
        return image

    def predict(self, image, input_mode="dataset"):
        image = self._input_image(image, input_mode)
        tensor = image_tensor(image, self.metadata["preprocessing"], self.metadata["model_name"],
                              transfer_size=self.metadata["transfer_size"])
        with torch.inference_mode():
            probabilities = self.model(tensor.unsqueeze(0).to(self.device)).softmax(1)[0].cpu()
        values, indices = probabilities.topk(min(3, len(probabilities)))
        predictions = [{**self.metadata["mapping"]["classes"][i], "confidence": float(value)}
                       for i, value in zip(indices.tolist(), values.tolist())]
        return {"prediction": predictions[0], "top_three": predictions,
                "confidence_note": "Softmax confidence is not calibrated and is not a guarantee of correctness.",
                "experiment_scope": self.metadata["experiment_scope"], "input_mode": input_mode}

    def preview(self, image, input_mode="dataset"):
        image = self._input_image(image, input_mode)
        return prepare_image(image, self.metadata["preprocessing"])
