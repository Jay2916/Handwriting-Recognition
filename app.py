from pathlib import Path
import pickle

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from sanskrit_ocr.common import load_config
from sanskrit_ocr.predict import Predictor
from sanskrit_ocr.crop_widget import visual_crop

st.set_page_config(page_title="Devanagari character recognition", page_icon="क")
st.title("Handwritten Devanagari character recognition")
st.write("Upload one isolated character. This prototype does not segment pages or recognize complete Sanskrit text.")
st.caption("Softmax confidence is not calibrated and is not a guarantee of correctness.")
st.caption("Supported: 33 consonant letters, क्ष/त्र/ज्ञ, and digits. Independent vowels such as ऋ and vowel-sign combinations such as रु are outside this model's classes.")

config_path = st.sidebar.text_input("Configuration file", str(Path(__file__).parent / "config.json"))
try:
    config = load_config(config_path)
except (OSError, ValueError, KeyError) as error:
    st.error(f"Cannot read configuration: {error}")
    st.stop()
checkpoints = sorted((Path(config["output_dir"]) / "checkpoints").glob("*.pt"))
# Keep smoke checkpoints out of the practical model selector.
checkpoints = [p for p in checkpoints if not p.stem.startswith("smoke_")]
if not checkpoints:
    st.warning("No trained checkpoints found. Run the training command in the README first.")
    st.stop()
preferred = config.get("default_checkpoint", "cnn.pt")
default_index = next((i for i, path in enumerate(checkpoints) if path.name == preferred), 0)
selected = st.selectbox("Available model", checkpoints, index=default_index,
                        format_func=lambda path: "MobileNetV3 Small (final)" if path.name == "transfer_final.pt" else path.stem)


@st.cache_resource
def load_predictor(path, modified, device):
    return Predictor(path, device)


try:
    predictor = load_predictor(str(selected), selected.stat().st_mtime_ns, config["training"]["device"])
except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
    st.error(f"Cannot load selected checkpoint: {error}")
    st.stop()
if predictor.metadata["experiment_scope"] == "pilot_subset":
    st.info("This model comes from a limited pilot experiment. See the measured report for its performance.")
if "best_epoch" in predictor.metadata:
    st.caption(f"Selected checkpoint: epoch {predictor.metadata['best_epoch']}")
upload = st.file_uploader("Character image", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"])
input_label = st.selectbox("Image type", ["Automatic", "Photograph", "Prepared image"])
input_mode = {"Automatic": "auto", "Photograph": "photo", "Prepared image": "dataset"}[input_label]
if upload is not None:
    try:
        with Image.open(upload) as opened:
            if opened.width * opened.height > 16_000_000:
                raise ValueError("Please use an image smaller than 16 megapixels")
            opened.load()
            image = ImageOps.exif_transpose(opened).copy()
        original, processed = st.columns(2)
        original.image(image, caption="Original", width=240)
        if input_mode != "dataset" and st.checkbox("Crop around the character"):
            bounds = visual_crop(image)
            if bounds is None:
                st.info("Select the whole character in the image above, then click Apply crop.")
                st.stop()
            image = image.crop(bounds)
            st.image(image, caption="Selected area", width=240)
            input_mode = "cropped_photo"
        processed.image(predictor.preview(image, input_mode), caption="Preprocessed character", width=240)
        result = predictor.predict(image, input_mode)
        top = result["prediction"]
        st.subheader(f"Predicted character: {top['character']}")
        st.write(f"Softmax confidence: {top['confidence']:.2%}")
        st.code(" ".join(top["codepoints"]), language=None)
        st.dataframe([{"Character": row["character"], "Dataset label": row["label"],
                       "Confidence": f"{row['confidence']:.2%}", "Unicode": " ".join(row["codepoints"])}
                      for row in result["top_three"]], hide_index=True)
        st.download_button("Download Unicode prediction", top["character"], file_name="prediction.txt", mime="text/plain; charset=utf-8")
    except UnidentifiedImageError:
        st.error("Cannot process this image. Please upload a valid PNG, JPEG, BMP, WebP, or TIFF file.")
    except (Image.DecompressionBombError, OSError, ValueError, RuntimeError) as error:
        st.error(f"Cannot process this image: {error}")
