const { randomUUID } = require("node:crypto");
const { runEngine, numbers } = require("./engine");
const { splitData, scaler, scale, unscale, metrics } = require("./data");

// Errors the client caused, such as asking a run for a model it never trained.
const requestError = (message) =>
    Object.assign(new Error(message), { status: 400 });

class Jobs {
    constructor({
        timeoutMs = 30000,
        command,
        ttlMs = 30 * 60 * 1000,
        maxActive = 4,
    } = {}) {
        this.jobs = new Map();
        this.options = { timeoutMs, command };
        this.ttlMs = ttlMs;
        this.maxActive = maxActive;
    }

    sweep() {
        for (const [id, job] of this.jobs) {
            if (Date.now() - job.created > this.ttlMs) {
                job.controller.abort();
                this.jobs.delete(id);
            }
        }
    }

    create(owner, data, notify) {
        this.sweep();
        const all = [...this.jobs.values()];
        if (
            all.filter((job) => job.status === "running" || job.predicting)
                .length >= this.maxActive
        )
            throw new Error("All training slots are busy. Try again shortly.");
        if (all.some((job) => job.owner === owner && job.status === "running"))
            throw new Error("Finish or cancel your current run first.");
        const previous = all.filter((job) => job.owner === owner);
        if (previous.length >= 3) {
            // Keep the latest completed LR and NN when evicting history. A cancelled
            // or failed comparison can still hold an LR result the client never adopted.
            const latest = [...previous]
                .reverse()
                .filter((item) => item.status === "completed");
            const keep = new Set([
                latest.find((item) => item.result.lr)?.id,
                latest.find((item) => item.result.nn)?.id,
            ]);
            const evicted =
                previous.find((item) => !keep.has(item.id)) || previous[0];
            evicted.controller.abort();
            this.jobs.delete(evicted.id);
        }
        if (this.jobs.size >= 64)
            throw new Error("The server is full. Try again later.");
        const job = {
            id: randomUUID(),
            owner,
            created: Date.now(),
            status: "running",
            data,
            loss: [],
            controller: new AbortController(),
            notify,
        };
        this.jobs.set(job.id, job);
        // Respond with the ID before work can generate socket events.
        setImmediate(() => this.train(job));
        return job;
    }

    owned(id, owner) {
        this.sweep();
        const job = this.jobs.get(id);
        return job?.owner === owner ? job : null;
    }

    snapshot(job) {
        return {
            id: job.id,
            status: job.status,
            loss: job.loss,
            result: job.result,
            error: job.error,
        };
    }

    async train(job) {
        const {
            x,
            y,
            layers,
            learningRate,
            epochs,
            activation = "sigmoid",
            weightDecay = 0,
            patience = 0,
            optimizer = "sgd",
            batchSize = 1,
            model = "both",
        } = job.data;
        const split = splitData(x, y);
        const trainX = split.train.map((point) => point.x);
        const trainY = split.train.map((point) => point.y);
        const fitX = split.fit.map((point) => point.x);
        const fitY = split.fit.map((point) => point.y);
        const validationX = split.validation.map((point) => point.x);
        // The network only learns from the fit points, so its scalers come from them alone.
        const sx = scaler(fitX);
        const sy = scaler(fitY);
        const min = Math.min(...x),
            max = Math.max(...x);
        const gridX = Array.from(
            { length: 101 },
            (_, i) => min + ((max - min) * i) / 100,
        );
        const testX = split.test.map((point) => point.x);
        const evaluationX = [...testX, ...validationX, ...gridX];
        const opts = { ...this.options, signal: job.controller.signal };
        try {
            job.model = { sx, sy };
            job.result = {
                split,
                settings: {
                    layers,
                    learningRate,
                    epochs,
                    activation,
                    weightDecay,
                    patience,
                    optimizer,
                    batchSize,
                    seed: 42,
                    model,
                },
                lr: null,
                nn: null,
            };
            if (model !== "nn") {
                const lr = await runEngine(
                    ["lr_train"],
                    `${trainX}\n${trainY}\n`,
                    opts,
                );
                const [slope] = numbers(lr.slope, 1),
                    [intercept] = numbers(lr.intercept, 1);
                const predict = (values) =>
                    values.map((value) => slope * value + intercept);
                job.model.slope = slope;
                job.model.intercept = intercept;
                job.result.lr = {
                    train: metrics(split.train, predict(trainX)),
                    test: metrics(
                        split.test,
                        predict(split.test.map((p) => p.x)),
                    ),
                    timeMs: numbers(lr.training_time_ms, 1)[0],
                    slope,
                    intercept,
                    curve: gridX.map((value) => ({
                        x: value,
                        y: slope * value + intercept,
                    })),
                };
            }
            if (model !== "lr") {
                const nn = await runEngine(
                    [
                        "nn_train_predict",
                        layers,
                        String(learningRate),
                        String(epochs),
                        activation,
                        `weight_decay=${weightDecay}`,
                        `patience=${patience}`,
                        `optimizer=${optimizer}`,
                        `batch=${batchSize}`,
                    ],
                    [
                        fitX.map((value) => scale(value, sx)),
                        fitY.map((value) => scale(value, sy)),
                        evaluationX.map((value) => scale(value, sx)),
                        validationX.map((value) => scale(value, sx)),
                        split.validation.map((point) => scale(point.y, sy)),
                    ]
                        .map((values) => `${values}\n`)
                        .join(""),
                    {
                        ...opts,
                        onLoss: (point) => {
                            if (job.status !== "running") return;
                            // Scaling Y by its range scales squared errors by range².
                            const original = {
                                epoch: point.epoch,
                                mse: point.mse * sy.range ** 2,
                                ...(point.valMse === undefined
                                    ? {}
                                    : { valMse: point.valMse * sy.range ** 2 }),
                            };
                            if (
                                !Number.isFinite(original.mse) ||
                                (original.valMse !== undefined &&
                                    !Number.isFinite(original.valMse))
                            )
                                throw new Error(
                                    "Training diverged. Lower the learning rate.",
                                );
                            job.loss.push(original);
                            job.notify({
                                type: "progress",
                                id: job.id,
                                ...original,
                            });
                        },
                    },
                );
                const fitPredictions = numbers(
                    nn.nn_predictions,
                    fitX.length,
                ).map((value) => unscale(value, sy));
                const predictions = numbers(
                    nn.eval_predictions,
                    evaluationX.length,
                ).map((value) => unscale(value, sy));
                if (
                    !nn.model ||
                    predictions.some((value) => !Number.isFinite(value))
                )
                    throw new Error("The engine returned invalid model data.");
                job.model.serialized = nn.model;
                const gridStart = testX.length + validationX.length;
                const validated = validationX.length > 0;
                const [epochsRun] = numbers(nn.epochs_run, 1);
                job.result.nn = {
                    train: metrics(split.fit, fitPredictions),
                    validation: validated
                        ? metrics(
                              split.validation,
                              predictions.slice(testX.length, gridStart),
                          )
                        : null,
                    test: metrics(
                        split.test,
                        predictions.slice(0, testX.length),
                    ),
                    epochsRun,
                    bestEpoch: validated ? numbers(nn.best_epoch, 1)[0] : null,
                    stoppedEarly: validated && nn.stopped_early === "1",
                    restoredBest: patience > 0,
                    timeMs: numbers(nn.training_time_ms, 1)[0],
                    curve: gridX.map((value, i) => ({
                        x: value,
                        y: predictions[gridStart + i],
                    })),
                };
            }
            if (job.controller.signal.aborted)
                throw new Error("Training cancelled.");
            job.status = "completed";
            job.notify({ type: "completed", ...this.snapshot(job) });
        } catch (error) {
            if (job.status === "cancelled") return;
            job.status = job.controller.signal.aborted ? "cancelled" : "failed";
            job.error = error.message;
            job.notify({ type: job.status, ...this.snapshot(job) });
        }
    }

    cancel(job) {
        if (job.status === "running") {
            job.status = "cancelled";
            job.controller.abort();
            job.notify({ type: "cancelled", ...this.snapshot(job) });
        }
    }

    removeOwner(owner) {
        for (const [id, job] of this.jobs) {
            if (job.owner === owner) {
                job.controller.abort();
                this.jobs.delete(id);
            }
        }
    }

    async predict(job, x, model) {
        if (job.status !== "completed")
            throw new Error("Complete training before predicting.");
        if (job.predicting) throw new Error("A prediction is already running.");
        if (
            [...this.jobs.values()].filter(
                (item) => item.status === "running" || item.predicting,
            ).length >= this.maxActive
        ) {
            throw new Error("All compute slots are busy. Try again shortly.");
        }
        job.predicting = true;
        try {
            const { slope, intercept, serialized, sx, sy } = job.model;
            const opts = { ...this.options, signal: job.controller.signal };
            const prediction = {
                x,
                extrapolation:
                    x < Math.min(...job.data.x) || x > Math.max(...job.data.x),
            };
            const selected = model || job.data.model || "both";
            if (selected !== "nn") {
                if (!job.result.lr)
                    throw requestError("This run has no trained linear model.");
                const lr = await runEngine(
                    ["lr_predict", String(slope), String(intercept), String(x)],
                    "",
                    opts,
                );
                prediction.lr = numbers(lr.predictions, 1)[0];
            }
            if (selected !== "lr") {
                if (!job.result.nn)
                    throw requestError("This run has no trained neural model.");
                const nn = await runEngine(
                    ["nn_predict"],
                    `${serialized}\n${scale(x, sx)}\n`,
                    opts,
                );
                prediction.nn = unscale(numbers(nn.predictions, 1)[0], sy);
                if (!Number.isFinite(prediction.nn))
                    throw new Error(
                        "Prediction is outside the model's numeric range.",
                    );
            }
            return prediction;
        } finally {
            job.predicting = false;
        }
    }
}

module.exports = { Jobs };
