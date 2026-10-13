# Measured model comparison

Scope: full_dataset. Same saved splits and training/validation sample selections.

| Model | Epochs | Validation accuracy | Test accuracy | Macro P / R / F1 | Training s | Checkpoint MiB | Latency median ms |
|---|---:|---:|---:|---|---:|---:|---:|
| cnn | 12 | 98.6061% | 98.5362% | 0.9856 / 0.9854 / 0.9854 | 929.7 | 1.39 | 0.35 |
| transfer_final | 10 | 98.7852% | 98.8986% | 0.9891 / 0.9890 / 0.9890 | 20517.6 | 6.09 | 30.91 |

The 96–99% presentation target is an aspiration, not a measured result or a promise.
All test metrics use the entire held-out test set. Test results must not be used to select hyperparameters.
Latency: 10 warmups, 100 synchronized batch-one forward passes, resident preprocessed input. See JSON for hardware and versions.
Training timing scope varies by run; see the notes below. Interrupted partial epochs are excluded from the finalized transfer timing.
This experiment measures isolated Devanagari characters, not Sanskrit words, sentences, manuscripts, writer generalization or confidence calibration.

- cnn: Full training-session wall time, including initialization and checkpoint IO.
- transfer_final: Sum of completed epoch durations across both sessions; excludes interrupted partial epochs, initialization, and between-epoch checkpoint IO.

The transfer run continued from epoch 8 weights with reinitialized optimizer, scheduler, and random state. Different architectures and training schedules were used; this is a practical comparison on matching data, not an identical-budget architecture experiment.
