# Repository guide

- `client/`: React with Vite and Vitest. Use `npm test` and `npm run build` here.
- `server/`: Express/WebSocket job API. `npm test` starts real local servers and C++ subprocesses; build the engine first.
- `cpp/`: C++11 models. `make -C cpp` builds the engine; `OMP_NUM_THREADS=1 make -C cpp test_all` runs its tests.
- Keep dataset splitting and original-unit metrics in `server/data.js`. Fit scalers on training data only.
- Keep model inference in C++; preserve full-precision model serialization and session ownership.
- Use 2-space indentation in frontend JS and 4 spaces in server/C++; prefer double quotes.
- Generated binaries, object files, build output, and node_modules must remain untracked.
- For UI changes, verify training, model comparison, prediction, and cancellation in a real browser at desktop and mobile widths.
