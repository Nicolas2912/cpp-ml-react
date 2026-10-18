# C++ engine

Build with `make`; run `OMP_NUM_THREADS=1 make test_all`. The actual CLI entry point is `main_server.cpp`.

Use C++11, four-space indentation, local headers before standard headers, and `UpperCamelCase` classes. Keep generated files untracked. Preserve deterministic neural-network seeding, finite-value validation, and double-precision serialization. Cover changes to training math or saved parameters with numerical tests and the real API integration suite.

`lr_train` reads X and Y as two CSV lines. `nn_train_predict <layers> <rate> <epochs> [sigmoid|tanh|relu] [weight_decay=<0..1>] [patience=<epochs>] [optimizer=sgd|adam] [batch=<size>]` reads those lines plus optional evaluation X, validation X, and validation Y lines. The activation defaults to sigmoid and is stored in the serialized model; the options default to per-sample SGD without weight decay or early stopping, which reproduces the original engine bit for bit. Progress lines are `epoch=<n>,mse=<m>` with `,val_mse=<v>` when validation data is given; results add `epochs_run`, and with validation `best_epoch` and `stopped_early`. `nn_predict` reads a serialized model line and an X CSV line. Standard output is machine-readable `key=value` data; diagnostics belong on standard error.

Keep optimizer updates in separate statements (see `apply_gradients`) so clang does not contract them into FMAs. If you change optimizer math, regenerate the PyTorch reference values with `tests/reference/optimizer_reference.py`.
