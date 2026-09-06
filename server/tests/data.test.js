const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
    validateDataset,
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
        { ...dataset, learningRate: 0 },
        { ...dataset, learningRate: Infinity },
        { ...dataset, layers: "1--8-1" },
        { ...dataset, layers: "2-8-1" },
        { ...dataset, layers: "1-33-1" },
        { ...dataset, layers: "1-32-32-32-32-1", epochs: 10000 },
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
