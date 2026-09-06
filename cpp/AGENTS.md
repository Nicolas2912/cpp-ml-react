# C++ engine

Build with `make`; run `OMP_NUM_THREADS=1 make test_all`. The actual CLI entry point is `main_server.cpp`.

Use C++11, four-space indentation, local headers before standard headers, and `UpperCamelCase` classes. Keep generated files untracked. Preserve deterministic neural-network seeding, finite-value validation, and double-precision serialization. Cover changes to training math or saved parameters with numerical tests and the real API integration suite.

`lr_train` reads X and Y as two CSV lines. `nn_train_predict <layers> <rate> <epochs>` reads those lines plus optional evaluation X. `nn_predict` reads a serialized model line and an X CSV line. Standard output is machine-readable `key=value` data; diagnostics belong on standard error.
