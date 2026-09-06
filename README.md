# ML playground

Compare a C++ linear regressor and a small neural network on the same dataset. The React interface shows their fitted curves, held-out test scores, and predictions for new inputs.

## Run locally

Requires Node.js 22.12+ (Node 24 LTS recommended), npm, make, and a C++11 compiler. GCC uses OpenMP; Clang can build sequentially when libomp is unavailable.

```sh
make -C cpp
npm --prefix server ci
npm --prefix client ci
```

Start these in separate terminals:

```sh
npm --prefix server start
npm --prefix client start
```

Open http://127.0.0.1:3000. Vite proxies `/api` and `/ws` to the API on port 3001. Both servers bind to loopback. Use the Vite origin for browser requests.

## What the comparison measures

- The server shuffles pairs with seed 42 and holds out 20% (rounded, at least two points). Both models train on the other 80%.
- NN normalization uses **training values only**. Training loss, train MSE, and test MSE are all returned in original Y units squared.
- Lower test MSE means better predictions on this particular split. A low training error alone does not establish generalization. Repeated tuning against one test split can overfit that split.
- Test R² compares predictions with the test-set mean: 1 is perfect, 0 matches that baseline, and negative values are worse. It is undefined (`null`) for constant test targets.
- Compute time includes training and engine evaluation/output, excluding process startup and network latency. It is not a rigorous benchmark.
- NN hidden layers use sigmoid activation; the output is linear. Training uses stochastic gradient descent with fixed initialization/shuffling seeds. Repeated runs with the same build and inputs are reproducible; timings and cross-platform floating-point results may vary.
- “Predict both” runs C++ inference using the actual trained parameters. NN weights round-trip at full double precision. New predictions are neither interpolation nor endpoint clamping.

## Structure

| Location                            | Purpose                                                   |
| ----------------------------------- | --------------------------------------------------------- |
| `client/src/App.jsx`                | Dataset, settings, and comparison workflow                |
| `client/src/hooks/useComparison.js` | Connection lifecycle and job updates                      |
| `client/src/components/`            | Fit/loss charts, score table, prediction form             |
| `server/server.js`                  | HTTP/WebSocket endpoints and session ownership            |
| `server/jobs.js`                    | Training, evaluation, model retention, cancellation       |
| `server/engine.js`                  | Bounded subprocess execution and streamed output parsing  |
| `server/data.js`                    | Validation, splitting, scaling, metrics                   |
| `cpp/main_server.cpp`               | CLI protocol                                              |
| `cpp/neural_network.*`              | Network training, inference, serialization                |
| `cpp/linear_regression.*`           | Analytical regression and gradient-descent implementation |

## Sessions and limits

A WebSocket connection receives an opaque session token. HTTP requests send it as `Authorization: Bearer <token>`. Jobs have unique IDs; only the owning session can read, cancel, or predict with them. Progress goes to that connection only.

Models live in server memory, with a 30-minute TTL and at most three retained runs per session. Disconnecting or restarting the server discards them; the UI reconnects and asks you to train again. This is a local playground, without accounts or disk model storage.

Limits: 10–1,000 input pairs; finite values within ±1,000,000; one input/output neuron; up to four hidden layers of 32 neurons; 1–10,000 epochs; learning rate in `(0, 1]`. A parameter × point × epoch budget rejects overly large runs. Each engine process has a 30-second timeout and a 2 MB output limit. The server allows four simultaneous compute operations, one training job per session, 32 connections, and 64 retained jobs globally. Disconnect and cancellation terminate associated subprocesses.

API:

| Method / path                | Purpose                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `GET /api/health`            | API liveness                                                                    |
| `POST /api/jobs`             | Train both models; body: `{x, y, layers, learningRate, epochs}`; returns `{id}` |
| `GET /api/jobs/:id`          | Recover the latest status, loss, and results                                    |
| `POST /api/jobs/:id/cancel`  | Cancel an active run                                                            |
| `POST /api/jobs/:id/predict` | Evaluate both models; body: `{x}`                                               |

WebSocket `/ws` events: `session`, `progress`, `completed`, `failed`, `cancelled`. Job events include `id`. The former `/api/lr_*` and `/api/nn_train_predict` HTTP routes have been replaced by this unified job API.

## Verify

```sh
OMP_NUM_THREADS=1 make -C cpp test_all
npm --prefix server test
npm --prefix client test
npm --prefix client run build
```

Server integration tests launch real C++ processes and HTTP/WebSocket clients. They cover inference, metric scaling, repeatability, session isolation, validation, cancellation, expiry, capacity, malformed output, launch failure, and timeout. UI tests cover training progress, results, prediction, errors, cancellation, invalid data, stale updates, and disconnects. CI runs these checks and the production build.

Browser check: choose Curve, compare models, inspect the held-out scores, predict at a new X, toggle a model line, change the dataset, and cancel a long run. Repeat at a narrow mobile width. Test loss of the API connection and recovery.

The production bundle is written to `client/build`. A deployment would need same-origin `/api` and `/ws` reverse proxying to the Node server; the build alone does not provide the backend.
