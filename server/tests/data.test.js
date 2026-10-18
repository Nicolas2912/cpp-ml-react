const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
    validateDataset,
    splitSizes,
    splitData,
    scaler,
    scale,
    unscale,
    metrics,
} = require("../data");
const dataset = {
    x: Array.from({ length: 20 }, (_, i) => i),
    y: Array.from({ length: 20 }, (_, i) => 2 * i + 1),
};

test("rejects malformed data and excessive training work", () => {
    for (const body of [
        null,
        {},
        { ...dataset, y: [] },
        { ...dataset, x: dataset.x.map(() => 1) },
        { ...dataset, y: dataset.y.map(() => NaN) },
        { ...dataset, y: dataset.y.map(() => "1") },
        { ...dataset, epochs: 1.5 },
        { ...dataset, epochs: 10001 },
        { ...dataset, activation: "softmax" },
        { ...dataset, activation: 1 },
        { ...dataset, learningRate: 0 },
        { ...dataset, learningRate: Infinity },
        { ...dataset, layers: "1--8-1" },
        { ...dataset, layers: "2-8-1" },
        { ...dataset, layers: "1-33-1" },
        { ...dataset, layers: "1-32-32-32-32-1", epochs: 10000 },
        { ...dataset, weightDecay: -0.1 },
        { ...dataset, weightDecay: 1.5 },
        { ...dataset, weightDecay: "0.1" },
        { ...dataset, weightDecay: NaN },
        { ...dataset, patience: -1 },
        { ...dataset, patience: 1.5 },
        { ...dataset, patience: 10001 },
        { ...dataset, optimizer: "rmsprop" },
        { ...dataset, batchSize: 0 },
        { ...dataset, batchSize: 2.5 },
        { ...dataset, batchSize: 1001 },
    ]) {
        assert.throws(() => validateDataset(body));
    }
    assert.equal(validateDataset(dataset).epochs, 3000);
});

test("split is deterministic, disjoint, and retains every pair including duplicate X", () => {
    const data = { ...dataset, x: dataset.x.map((value) => value % 5) };
    const a = splitData(data.x, data.y),
        b = splitData(data.x, data.y);
    assert.deepEqual(a, b);
    assert.equal(a.test.length, 4);
    const indices = [...a.train, ...a.test].map((p) => p.index);
    assert.equal(new Set(indices).size, 20);
    for (const p of [...a.train, ...a.test]) assert.equal(p.y, data.y[p.index]);
    // The network's validation and fit points partition the training points.
    assert.equal(a.validation.length, 3);
    assert.deepEqual(
        [...a.validation, ...a.fit].map((p) => p.index).sort((x, y) => x - y),
        a.train.map((p) => p.index).sort((x, y) => x - y),
    );
    const test = new Set(a.test.map((p) => p.index));
    assert.ok(a.validation.every((p) => !test.has(p.index)));
});

test("validation is held out only when the training set can spare it", () => {
    assert.deepEqual(splitSizes(5), { test: 1, train: 4, validation: 0 });
    assert.deepEqual(splitSizes(11), { test: 2, train: 9, validation: 0 });
    assert.deepEqual(splitSizes(12), { test: 2, train: 10, validation: 2 });
    assert.deepEqual(splitSizes(60), { test: 12, train: 48, validation: 10 });
    const small = splitData([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    assert.deepEqual(small.fit, small.train);
    assert.equal(small.validation.length, 0);
    const eleven = {
        x: dataset.x.slice(0, 11),
        y: dataset.y.slice(0, 11),
    };
    assert.throws(
        () => validateDataset({ ...eleven, patience: 10 }),
        /at least 12 points/,
    );
    assert.equal(validateDataset({ ...eleven, patience: 0 }).patience, 0);
    assert.equal(
        validateDataset({
            ...dataset,
            x: dataset.x.slice(0, 12),
            y: dataset.y.slice(0, 12),
            patience: 10,
        }).patience,
        10,
    );
});

test("scalers use training values only and restore original-unit error", () => {
    const s = scaler([10, 20, 30]);
    assert.equal(scale(100, s), 4.5);
    assert.equal(unscale(scale(100, s), s), 100);
    assert.equal(scaler([7, 7]).range, 1);
    assert.deepEqual(metrics([{ y: 10 }, { y: 30 }], [12, 28]), {
        mse: 4,
        r2: 0.96,
    });
    assert.equal(metrics([{ y: 5 }, { y: 5 }], [5, 5]).r2, null);
    assert.throws(() => metrics([{ y: 1 }], [Infinity]));
});

test("defaults the activation to sigmoid and accepts tanh and relu", () => {
    assert.equal(validateDataset(dataset).activation, "sigmoid");
    for (const activation of ["tanh", "relu"]) {
        assert.equal(
            validateDataset({ ...dataset, activation }).activation,
            activation,
        );
    }
});

test("regularization options default to plain per-sample SGD", () => {
    const defaults = validateDataset(dataset);
    assert.equal(defaults.weightDecay, 0);
    assert.equal(defaults.patience, 0);
    assert.equal(defaults.optimizer, "sgd");
    assert.equal(defaults.batchSize, 1);
    const chosen = validateDataset({
        ...dataset,
        weightDecay: 0.3,
        patience: 500,
        optimizer: "adam",
        batchSize: 1000,
    });
    assert.deepEqual(
        [
            chosen.weightDecay,
            chosen.patience,
            chosen.optimizer,
            chosen.batchSize,
        ],
        [0.3, 500, "adam", 1000],
    );
});
