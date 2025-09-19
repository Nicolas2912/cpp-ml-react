# Repository Guidelines

## Project Structure & Module Organization
Source code lives in the repository root: `main_server.cpp` exposes the CLI entry point, `linear_regression.cpp` and `neural_network.cpp` implement the ML primitives, and headers (`*.h`) define their public surface. `main.cpp` is a sandbox driver for manual experimentation; keep it lightweight or move experiments into `build/` if they grow. Generated artifacts (`*.o`, `linear_regression_app`, `neuronal_network.exe`, `build/`) should stay untracked so incremental C++ rebuilds remain fast.

## Build, Test, and Development Commands
Use the GNU Make workflow that ships with the project:
- `make` — compiles all sources with `g++`, C++11, OpenMP, and produces `linear_regression_app`.
- `make clean` — removes compiled objects and binaries (Windows-compatible via `del`).
- `./linear_regression_app lr_train < data.txt` — trains the analytical linear regressor using two comma-separated lines (X then y) piped from `data.txt`.
- `./linear_regression_app nn_train_predict 1-5-1 0.05 1000 < data.txt` — trains the neural net with layer signature `1-5-1`; tune learning rate and epochs per experiment.

## Coding Style & Naming Conventions
Write modern C++11 with 4-space indentation and no tabs. Favor `UpperCamelCase` for classes (`LinearRegression`, `NeuralNetwork`) and `snake_case` for variables and free utilities (`learning_rate`, `parse_vector`). Keep local headers (e.g., `"linear_regression.h"`) grouped before standard headers, mirroring existing files. Parallel sections rely on OpenMP pragmas; vet new parallel code for determinism. Run `clang-format` if available, using LLVM style as a baseline.

## Testing Guidelines
There is no formal test harness yet—validate changes by exercising the CLI modes. Store sample datasets outside version control (for example, under `build/fixtures/` and list that path in `.gitignore`) and rerun both `lr_train` and `nn_train_predict` flows with representative inputs, checking metrics such as `mse` and `training_time_ms`. When introducing stochastic behavior, set deterministic seeds or document the expected variability to keep outputs debuggable.

## Commit & Pull Request Guidelines
Base commit messages on the existing short imperative style ("Add", "Fix", "Update"), keep the summary line under ~60 characters, and expand details in the body if needed. For pull requests, describe the problem, the approach, and validation steps; link issues or TODO references explicitly. Include sample command output or before/after metrics whenever the change affects model quality or performance.
