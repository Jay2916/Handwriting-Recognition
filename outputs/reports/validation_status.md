# Final validation status

The project is finalized with `outputs/checkpoints/transfer_final.pt` as the default application model. Further training has been stopped at the user's request after 10 completed epochs.

## Final MobileNetV3 Small model

- Trained on all 70,375 eligible training images; 7,820 validation images.
- The checkpoint at epoch 10 was selected using the lowest validation loss (0.038105), before final test evaluation. Its validation accuracy is 98.7852%.
- Epochs 1–8 completed in one session. An interruption occurred during epoch 9. Training continued from the saved epoch 8 weights for epochs 9–10, with optimizer, scheduler, and random state reinitialized. Another partially completed epoch was stopped at the user's request. This is a continuation from learned weights, not an exact uninterrupted run.
- Test evaluation used all 13,800 official held-out images with the original checkpoint preprocessing, not the new photo cleanup mode.
- Test accuracy: 98.8986%; macro precision: 0.989052; macro recall: 0.988986; macro F1: 0.988969.
- Recorded training time: 20517.6 seconds, summed completed-epoch durations only. It excludes initialization, between-epoch checkpoint IO, and interrupted partial epochs. It is not end-to-end wall time.
- Test evaluation: 232.4 seconds. Median model-only CPU inference: 30.914 ms, excluding image decoding, preprocessing, model loading, and UI rendering.

## Other models

The full-data CNN remains available. Its test accuracy is 98.5362% and macro F1 is 0.9853755. See `comparison.md` for the matching split/sample fingerprints and the differing training schedules and timing scope. The interrupted MobileNet pilot is historical; its test metrics were never measured and are not used in the final comparison.

## Application verification and limits

- The complete project test suite passed: 32 tests.
- The visual crop control was checked in a browser: dragging around the contrast-edited र and applying the crop produced a र prediction. Crop coordinates map back to the original image size.
- The final epoch 10 model predicted क for ka.jpeg, र for ra.jpeg, and र for ra-contrast.jpeg after a manual crop, using the photo cleanup mode. These three supplied examples helped develop preprocessing; they are not an independent photo-accuracy benchmark. See `photo_diagnostics/final_model_results.json`.
- ru.jpeg represents ऋ, an independent vowel outside the 46-class mapping. Unsupported characters are not automatically detected or rejected; the classifier will still choose a supported class.
- This model recognizes isolated characters, not words or pages. Photo cleanup may fail on colored ink, overlapping ruling, shadows, heavy contrast edits, or multiple marks. Softmax scores are uncalibrated.
- Dataset inspection and duplicate removal are unchanged: 92,000 original PNGs, 46 classes; five training copies overlapping the official test set were excluded. No writer identifiers were supplied, so writer-independent generalization is unverified.
