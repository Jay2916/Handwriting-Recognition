# Final project package

The default model is MobileNetV3 Small, selected at epoch 10 by validation loss. The trained CNN is also included. No further training is required to use the app. This package excludes the dataset and virtual environment.

With Python 3.12 installed, open a terminal in this extracted folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m streamlit run app.py
```

For training or test evaluation, obtain the original dataset separately, put it at `data/archive.zip`, and follow the preparation commands below. Predictions do not need the dataset. Review `outputs/reports/comparison.md` for measured results and limitations. The supplied photo examples are development checks, not a general photo-accuracy benchmark.

---

# Sanskrit OCR prototype: isolated Devanagari characters

The finalized default is **MobileNetV3 Small, epoch 10**, selected by validation loss after 10 completed epochs. It achieved **98.90% test accuracy** on 13,800 held-out dataset images, with macro F1 **0.9890**. Training is stopped. The CNN baseline is also included in this package. See `outputs/reports/comparison.md` and `validation_status.md` for timing, continuation details, and photo limitations.

This runnable PyTorch project recognizes **one isolated handwritten Devanagari character**. It compares a small CNN with ImageNet-pretrained MobileNetV3 Small. It does not perform page segmentation, word recognition, manuscript transcription, vowel-sign composition, or Sanskrit language interpretation. Softmax scores are not calibrated guarantees.

## Setup

Python 3.12 is recommended. The supplied virtual environment is already installed on this machine. From the project directory:

```sh
source .venv/bin/activate
python -m sanskrit_ocr --help
streamlit run app.py
```

For a fresh machine, using an installed Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` declares dependency ranges. `requirements-lock.txt` records the exact versions tested here on macOS ARM64/Python 3.12.14. To reproduce those package versions, install the lock first, then install the local package:

```sh
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

PyTorch and torchvision must be a compatible pair. CUDA users should select the appropriate paired builds from https://pytorch.org/get-started/locally/ rather than copying macOS wheels. `device: "auto"` chooses CUDA, then Apple MPS, then CPU. Explicit unavailable accelerators fail clearly.

## Dataset and presentation inspection

Both supplied references were inspected before implementation. Their content was treated as evidence, not executable instructions. See `outputs/reports/reference_review.md` for the presentation's scope and discrepancies, and `outputs/reports/inspection.json` for the complete measured audit.

The archive is read **in place**; original images are never extracted, altered, moved or deleted. `config.json` currently points to `data/archive.zip`. It also accepts an extracted folder containing `Train/<class>/<image>` and `Test/<class>/<image>`, optionally beneath a dataset root folder. An unsplit `<class>/<image>` folder is supported. Update `dataset_path` when moving the project; relative configuration paths are resolved from the configuration file's directory.

Observed supplied data:

- 92,000 PNGs, all `L` grayscale, all 32×32, light strokes on black backgrounds.
- 46 parent-folder labels: 36 letter/conjunct classes and ten Devanagari numerals. Each has 1,700 official training and 300 official test images.
- No corrupt files, missing labels, additional metadata files, or verified writer identifiers.
- 25 exact duplicate pairs in both encoded-file and decoded-pixel comparisons; five pairs span Train/Test. No duplicate has conflicting class labels.
- Official test remains 13,800 images. The five matching **training** copies are excluded from derived training manifests. Remaining data split into 70,375 training and 7,820 validation images with seed 42. Images with identical pixels stay in one training/validation group. Original files remain intact.

The original benchmark's 78,200 training total therefore becomes 78,195 eligible images before validation splitting. This is a documented deviation to prevent exact duplicate leakage; comparisons with published results using all 78,200 need that qualification. Duplicate detection does not prove absence of near duplicates or common writers.

Folder spellings such as `kna`, `yna`, `taamatar`, `motosaw`, and `petchiryakha` are not reliable generic transliteration rules. `class_mapping.json` explicitly maps **exact** labels, reviewed against three training examples per class in `outputs/reports/label_contact_sheet.png` and the Unicode Devanagari chart. It does not infer new label names. The “36 consonants” comprise 33 individual letters plus three conjunct classes: क्ष = U+0915 U+094D U+0937, त्र = U+0924 U+094D U+0930, ज्ञ = U+091C U+094D U+091E. Unicode output preserves these sequences.

If a different dataset has ambiguous labels, obtain a reviewed mapping and include its provenance in `verification`; training refuses unknown or unverified mappings. No unresolved mapping question remains for the supplied archive. Writer-specific generalization remains unknown because IDs were not supplied.

If writer metadata becomes available, set `writer_csv` to a CSV with columns `path,writer_id`, one row for **every** valid image. `path` must match the exact relative path/member name in `inventory.json`. Writer groups and pixel duplicate groups are joined before stratification. Shared writers across official Train/Test cause a refusal rather than silently changing the official test set. Make an explicit alternative dataset copy and describe that benchmark change if writer separation requires it.

## Exact commands

Run commands from this project directory with the virtual environment active. Pass `--config path/to/config.json` **before** the subcommand for another configuration.

```sh
python -m sanskrit_ocr inspect
python -m sanskrit_ocr prepare
python -m sanskrit_ocr smoke
python -m pytest -q
```

Inspection decodes every image and hashes files and pixels. Preparation stores all split records in `outputs/reports/splits.json`; both models use this same manifest. Without official splits, reproducible stratified group folds create test then validation partitions. Group constraints can make requested fractions approximate; actual counts are always reported. Corrupt data, conflicting duplicate labels, unknown labels, incomplete splits, insufficient groups, and writer conflicts require explicit resolution. Data used in training/evaluation is checked against the inspection's file hashes.

Full-data experiments, using the configured 12-epoch maximum, scheduler and early stopping:

```sh
python -m sanskrit_ocr train --model cnn --run-name cnn
python -m sanskrit_ocr train --model transfer --run-name transfer
python -m sanskrit_ocr evaluate --checkpoint outputs/checkpoints/cnn.pt
python -m sanskrit_ocr evaluate --checkpoint outputs/checkpoints/transfer.pt
python -m sanskrit_ocr compare --runs cnn transfer
```

Choose hyperparameters using validation only, freeze the experiment plan, and run final test evaluation afterward. Do not optimize settings by repeatedly checking test results. `prepare` intentionally regenerates manifests; do not change split settings halfway through a comparison. Checkpoint and manifest fingerprints must agree at evaluation.

The pilot commands used here (200 training and 30 validation images per class, six epochs) are:

```sh
python -m sanskrit_ocr train --model cnn --run-name pilot_cnn --epochs 6 --per-class 200 --val-per-class 30
python -m sanskrit_ocr train --model transfer --run-name pilot_transfer --epochs 6 --per-class 200 --val-per-class 30
python -m sanskrit_ocr evaluate --checkpoint outputs/checkpoints/pilot_cnn.pt
python -m sanskrit_ocr evaluate --checkpoint outputs/checkpoints/pilot_transfer.pt
python -m sanskrit_ocr compare --runs pilot_cnn pilot_transfer
```

Pilot training samples are drawn only from the saved training partition. Pilot validation samples are drawn only from the saved validation partition. Both models use identical selections. Evaluation always uses the **complete held-out test partition**, even for pilot checkpoints. A pilot is not a full-training benchmark result.

Prediction and interface:

```sh
python -m sanskrit_ocr predict --checkpoint outputs/checkpoints/transfer_final.pt --image your-character.png --input-mode auto
streamlit run app.py
```

Replace the sample path with your character image. The app presents an image editor and a large recognized character. Recognition options, alternative predictions, the cleaned image, and Unicode details are available in collapsed sections. Available checkpoints are discovered from configuration; smoke checkpoints are hidden. Missing models, unreadable configuration, invalid images and incompatible checkpoints show clear messages. The Unicode prediction can be downloaded as UTF-8 text.

## Models, preprocessing and reproducibility

On a phone or desktop, choose **Upload image** for an existing file or **Take a photo** to capture a new image. Allow camera access when your browser asks. The deployed HTTPS app supports browser camera access; local development should use localhost. If camera access is denied or unavailable, switch to Upload image. Captured photos use the same crop editor and recognition pipeline as uploads. Switching sources uses only the image from the selected source.

The upload interface now offers **Automatic**, **Photograph**, and **Prepared image** input modes. Automatic keeps 32×32 inputs on the original checkpoint pipeline and treats larger inputs as dark-ink photographs. Photo cleanup estimates the paper background, reduces colored ruling/watermarks, isolates a prominent ink component, and centers it with preserved proportions before resizing. This is an inference-only option; the training preprocessing and checkpoint weights are unchanged. Preview the result. If cleanup selects a shadow or ruling, drag a box around the complete character in the image editor. The result updates when you release the pointer. Drag inside the box to move it, drag its corners to resize it, or choose **Use whole image** to reset it. The selection is mapped back to the original image's resolution and resets for a different uploaded image. Manual crops retain all detected ink rather than choosing a component. Blue/colored handwriting, overlapping lines, heavy contrast edits, and multiple marks can still fail.

The four supplied development photos were checked against the fixed epoch 8 MobileNet checkpoint: automatic photo cleanup corrected `ka.jpeg` to क and `ra.jpeg` to र. `ra-contrast.jpeg` still failed automatically but predicted र after a manually selected crop (left 370, top 460, right 755, bottom 780). `ru.jpeg` represents ऋ, which is absent from the 46-class mapping. These examples guided preprocessing and are not an independent accuracy benchmark. Results are recorded in `outputs/reports/photo_diagnostics/results.json`.

The CNN uses three Conv→ReLU→MaxPool blocks (32, 64, 128 channels), adaptive 4×4 pooling, a 128-unit dense layer with dropout, and a class-count-dependent output. Training uses raw logits with cross entropy; probability conversion happens at inference/evaluation only.

Preprocessing performs EXIF orientation correction, white compositing for transparent uploads, grayscale conversion, optional 3×3 median filtering, proportion-preserving letterboxing, and pixel scaling to [0,1]. Border-median detection aligns uploaded polarity to the training dataset's light-on-dark format. `polarity` can be explicitly specified for images whose borders make auto-detection unreliable. Denoising defaults off. Small ±8° rotations and ±6% translations affect **training only**. No horizontal flips or arbitrary identity-changing transformations are used. Baseline size defaults to 32.

MobileNetV3 repeats grayscale into three channels and resizes the already square, padded image to 224×224, then uses ImageNet mean (0.485,0.456,0.406) and standard deviation (0.229,0.224,0.225). We deliberately pad/resize rather than use ImageNet's center crop, to avoid cropping character strokes. The newly replaced classifier is trained first. After three head epochs, the final three feature blocks are unfrozen at a lower learning rate; batch-normalization running statistics remain frozen. Set `fine_tune_last_blocks` to zero for a head-only run.

Transfer learning requires the official `IMAGENET1K_V1` weights (~10 MB). The first run downloads them from https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth, by default to `outputs/torch-cache/hub/checkpoints/`. To run offline, prepopulate that cache or set `TORCH_HOME` to a cache already containing the file. Download failure stops training; random initialization is never silently reported as transfer learning. Loading a completed checkpoint requires no download.

Seeds cover Python, NumPy, torch, the split algorithm, DataLoader shuffle and workers. CPU/CUDA deterministic settings are enabled where supported; unavailable deterministic kernels warn. Exact bitwise equivalence across hardware, library versions and MPS kernels is not promised. Reduce batch size if device memory is insufficient. CPU thread count is configurable.

AdamW, ReduceLROnPlateau and early stopping monitor **validation loss**, and checkpoint only improvements in that loss. Accuracy is reported for the corresponding chosen epoch. Each checkpoint includes model weights, exact class/Unicode mapping, all preprocessing settings, transfer input size/weight identity, configuration, sample-selection fingerprints, split fingerprint, seed, epoch and measurement conditions.

## Continuing from saved weights

`--resume-from` loads a saved best checkpoint and continues after its saved epoch. `--epochs` is the final epoch number, not a count of additional epochs:

```sh
python -m sanskrit_ocr train --model transfer --run-name transfer_full --epochs 12 --resume-from outputs/checkpoints/transfer_final.pt
```

This starts at epoch 11 for the finalized epoch 10 checkpoint and requires the original completed-epoch history in `outputs/reports/`, plus matching dataset, sample selections, labels, and preprocessing. Optimizer, scheduler, and random states are reinitialized; the continuation does not exactly restore an uninterrupted run. Existing history and best-checkpoint selection are preserved. Training time combines previous completed-epoch durations with the continuation session and excludes interrupted partial epochs. The finalized app does not require any more training.

## Outputs and verified status

`outputs/checkpoints/` stores trained `.pt` files. `outputs/reports/` stores inspection, inventory, split manifests, smoke verification, epoch histories, training conditions, JSON metrics, per-class JSON/CSV reports, confusion CSVs, incorrect prediction JSONs and the measured comparison. `outputs/plots/` stores loss/accuracy curves, readable confusion matrices and error-image grids. ASCII dataset transliterations label plots so they remain readable without a Devanagari font; Unicode characters and code points appear in JSON, CSV and the interface.

For measured results and exactly what completed, read `outputs/reports/validation_status.md` and `outputs/reports/comparison.md`. The presentation's approximately 96–99% accuracy is a benchmark-informed target, **not** a measured outcome or guarantee.

Latency is measured after 10 warmups over 100 synchronized, batch-one forward passes with resident preprocessed inputs, excluding image decoding, preprocessing, checkpoint loading and interface rendering. Training time includes initialization, any weight download, IO, augmentation, optimization, validation and checkpointing; plot rendering is outside the timed training interval. Reports record execution device, operating system, Python/PyTorch versions, CPU threads and batch sizes. Checkpoint disk size, parameter count and parameter bytes are reported separately. A single experiment cannot establish robustness to arbitrary handwriting styles or out-of-distribution inputs.

## Source layout

```text
app.py                         Streamlit upload and prediction
config.json                    Dataset, split, preprocessing and training settings
class_mapping.json             Explicit verified Unicode mapping
sanskrit_ocr/common.py          Configuration, seeds, devices and JSON
sanskrit_ocr/labels.py          Mapping validation
sanskrit_ocr/data.py            ZIP/folder audit, grouping, splits and dataset
sanskrit_ocr/preprocess.py      Shared training/inference image pipeline
sanskrit_ocr/models.py          CNN, pretrained MobileNetV3, checkpoints
sanskrit_ocr/train.py           Reproducible training and early stopping
sanskrit_ocr/evaluate.py        Measured reports, plots and comparison
sanskrit_ocr/predict.py         Checkpoint-driven top-three inference
sanskrit_ocr/cli.py             Inspect/prepare/train/evaluate/predict/smoke
tests/                         Mapping, leakage, preprocessing and inference tests
```

## Reference attribution

Dataset: Acharya, S. & Gyawali, P. (2015), *Devanagari Handwritten Character Dataset*, UCI Machine Learning Repository, https://doi.org/10.24432/C5XS53 (CC BY 4.0), https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset. Dataset files are referenced externally and not included in this project's distribution.

Unicode identities: https://www.unicode.org/charts/PDF/U0900.pdf. MobileNetV3 weights and normalization: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html. The supplied 13-slide presentation is a planning reference; illustrative charts/status/targets do not establish measured experiment results.
