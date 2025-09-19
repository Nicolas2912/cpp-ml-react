# Repository Guidelines

## Project Structure & Module Organization
The app is split across `client/` (React UI and tests under `src/`), `server/` (Express/WebSocket bridge in `server.js`), and `cpp/` (training engine with `linear_regression.cpp`, `neural_network.cpp`, and `Makefile`). The compiled executable `cpp/linear_regression_app` is what the server spawns; keep generated binaries and `.o` files inside `cpp/` or `build/`. Shared assets such as sample datasets should live under `build/fixtures/` and stay untracked.

## Build, Test, and Development Commands
- `make -C cpp` — builds the C++ engine with OpenMP support when available.
- `make -C cpp clean` — removes `*.o` and local binaries.
- `npm install && npm start` inside `server/` — installs dependencies and runs the API on `localhost:3001`.
- `npm install && npm start` inside `client/` — launches the CRA dev server with hot reload at `localhost:3000`.
- `npm run build` (client) — produces a production bundle in `client/build/` for static hosting.

## Coding Style & Naming Conventions
Use 2-space indentation in the React codebase and keep component files in PascalCase (`LossChart.js`, `NNVisualizer.js`). Favor descriptive state setters such as `setLossHistory`. Server code uses 4-space indentation with semicolons and `camelCase` helpers (`parseCppLine`, `broadcast`). C++ modules follow 4-space indentation, `UpperCamelCase` classes, and `snake_case` free functions; include headers locally using double quotes and group standard headers afterwards. Stick with double quotes in JS/JSX to match existing files.

## Testing Guidelines
Run `npm test -- --watchAll=false` in `client/` to execute the Jest/Testing Library suite before pushing UI changes. There is no automated server harness; validate API updates by `node server.js` and exercising endpoints via the React app or `curl http://localhost:3001/api/lr_train`. For C++, smoke-test both modes: `./linear_regression_app lr_train < data/sample.txt` and `./linear_regression_app nn_train_predict 1-4-1 0.01 1000 < data/sample.txt`, capturing metrics for regression comparisons. Document any expected nondeterminism (e.g., randomized datasets).

## Commit & Pull Request Guidelines
Keep commit subjects short and imperative (`Fix LR button wiring`, `Update client build`). When changes span multiple layers, split commits by concern (engine vs. UI) so diffs stay reviewable. Pull requests should explain the problem, summarize the approach, and list validation steps (commands run, screenshots of new charts, sample metrics). Link related issues and mention configuration notes whenever contributors must rebuild C++ binaries or reinstall npm packages.
