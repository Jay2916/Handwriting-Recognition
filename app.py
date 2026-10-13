import io
import pickle
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from sanskrit_ocr.common import load_config
from sanskrit_ocr.crop_widget import visual_crop
from sanskrit_ocr.predict import Predictor

st.set_page_config(page_title="Sanskrit OCR", page_icon="क", layout="wide")
st.markdown("""<style>
.stMainBlockContainer {max-width:1120px;padding-top:2rem;}
[data-testid="stMetricValue"] {font-size:5rem;line-height:1.25;}
[data-testid="stMetricLabel"] {font-size:1rem;}
</style>""", unsafe_allow_html=True)
st.title("Sanskrit OCR")
st.caption("Upload one handwritten Devanagari character. Drag around it to refine the result.")

try:
    config = load_config(Path(__file__).parent / "config.json")
except (OSError, ValueError, KeyError) as error:
    st.error(f"Cannot read configuration: {error}")
    st.stop()

checkpoints = sorted((Path(config["output_dir"]) / "checkpoints").glob("*.pt"))
checkpoints = [path for path in checkpoints if not path.stem.startswith("smoke_")]
if not checkpoints:
    st.warning("No trained checkpoints found. Recognition is not ready yet.")
    st.stop()


def model_label(path):
    if path.name == "transfer_final.pt":
        return "Final model · MobileNetV3"
    if path.name == "cnn.pt":
        return "Fast model · CNN"
    return path.stem.replace("_", " ")


preferred = config.get("default_checkpoint", "cnn.pt")
default_index = next((i for i, path in enumerate(checkpoints) if path.name == preferred), 0)
with st.expander("Recognition options", expanded=False):
    model_column, format_column = st.columns(2)
    selected = model_column.selectbox("Model", checkpoints, index=default_index, format_func=model_label)
    image_type = format_column.selectbox("Image type", ["Automatic", "Photograph", "Prepared image"])
    st.caption("Automatic cleans up photos and preserves prepared 32×32 character images.")
input_mode = {"Automatic": "auto", "Photograph": "photo", "Prepared image": "dataset"}[image_type]


@st.cache_resource
def load_predictor(path, modified, device):
    return Predictor(path, device)


@st.cache_data(show_spinner=False, max_entries=16)
def recognize(image_bytes, path, modified, device, mode):
    predictor = load_predictor(path, modified, device)
    with Image.open(io.BytesIO(image_bytes)) as image:
        preview = predictor.preview(image, mode)
        result = predictor.predict(image, mode)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return result, buffer.getvalue()


editor_column, result_column = st.columns([1.3, 1], gap="large")
result = None
with editor_column:
    with st.container(border=True):
        st.subheader("Your image")
        upload = st.file_uploader("Upload a character image", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
                                  label_visibility="collapsed")
        if upload is None:
            st.caption("A clear, close photo works best. Keep the whole character and its headline visible.")
        else:
            try:
                with Image.open(upload) as opened:
                    if opened.width * opened.height > 16_000_000:
                        raise ValueError("Please use an image smaller than 16 megapixels")
                    opened.load()
                    image = ImageOps.exif_transpose(opened).copy()
                bounds = visual_crop(image)
                selected_image = image.crop(bounds) if bounds else image
                # Prepared black-background images stay on the original pipeline.
                mode = input_mode
                if bounds and input_mode != "dataset":
                    mode = "dataset" if input_mode == "auto" and image.size == (32, 32) else "cropped_photo"
                buffer = io.BytesIO()
                selected_image.save(buffer, format="PNG")
                with st.spinner("Reading the character…"):
                    result, preview_bytes = recognize(buffer.getvalue(), str(selected), selected.stat().st_mtime_ns,
                                                      config["training"]["device"], mode)
            except UnidentifiedImageError:
                st.error("Cannot process this image. Please upload a valid image file.")
                result = None
            except (Image.DecompressionBombError, OSError, ValueError, RuntimeError, KeyError,
                    EOFError, pickle.UnpicklingError) as error:
                st.error(f"Cannot process this image: {error}")
                result = None

with result_column:
    with st.container(border=True):
        if upload is None or result is None:
            st.subheader("Your result")
            st.caption("The recognized character will appear here.")
        else:
            top = result["prediction"]
            st.metric("Recognized character", top["character"])
            st.caption(f"Model confidence · {top['confidence']:.1%}")
            st.download_button("Download text", top["character"], file_name="prediction.txt",
                               mime="text/plain; charset=utf-8", width="stretch")
            st.caption("Model confidence is not a guarantee of correctness.")
            with st.expander("Other possibilities"):
                candidates = st.columns(len(result["top_three"]))
                for column, row in zip(candidates, result["top_three"]):
                    column.subheader(row["character"])
                    column.caption(f"{row['confidence']:.1%}")
            with st.expander("Image seen by the model"):
                st.image(preview_bytes, width=180)
                st.caption("Check that cleanup kept the whole character. If not, refine the crop or change Image type.")
            with st.expander("Technical details"):
                st.code(" ".join(top["codepoints"]), language=None)
                predictor = load_predictor(str(selected), selected.stat().st_mtime_ns, config["training"]["device"])
                epoch = predictor.metadata.get("best_epoch")
                if epoch is not None:
                    st.caption(f"Saved checkpoint · epoch {epoch}")
                if result["experiment_scope"] == "pilot_subset":
                    st.caption("This checkpoint comes from a limited pilot experiment.")

with st.expander("What can this recognize?"):
    st.write("One isolated character: 33 consonant letters, क्ष / त्र / ज्ञ, or a Devanagari digit.")
    st.caption("Independent vowels such as ऋ, vowel signs such as रु, and complete words or pages are outside this model's classes.")
    st.caption("Avoid shadows and notebook lines where possible. Heavy contrast editing can damage strokes.")
