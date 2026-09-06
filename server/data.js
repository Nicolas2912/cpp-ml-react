const LIMITS = Object.freeze({
    points: 1000,
    magnitude: 1e6,
    epochs: 10000,
    work: 5e7,
});

function validateDataset(body) {
    const {
        x,
        y,
        layers = "1-8-1",
        learningRate = 0.3,
        epochs = 3000,
    } = body || {};
    if (
        !Array.isArray(x) ||
        !Array.isArray(y) ||
        x.length < 10 ||
        x.length > LIMITS.points ||
        x.length !== y.length
    ) {
        throw new Error("Enter 10 to 1,000 matching X and Y values.");
    }
    if (
        [...x, ...y].some(
            (value) =>
                typeof value !== "number" ||
                !Number.isFinite(value) ||
                Math.abs(value) > LIMITS.magnitude,
        )
    ) {
        throw new Error("Use finite numbers between −1,000,000 and 1,000,000.");
    }
    if (new Set(x).size < 2)
        throw new Error("Use at least two distinct X values.");
    if (
        typeof layers !== "string" ||
        layers.length > 24 ||
        !/^1(?:-[1-9]\d?){0,4}-1$/.test(layers)
    ) {
        throw new Error(
            "Use one input and output, with up to four hidden layers (for example 1-8-1).",
        );
    }
    const sizes = layers.split("-").map(Number);
    if (sizes.some((size) => size > 32))
        throw new Error("Use at most 32 neurons per layer.");
    if (
        typeof learningRate !== "number" ||
        !Number.isFinite(learningRate) ||
        learningRate <= 0 ||
        learningRate > 1
    ) {
        throw new Error("Learning rate must be greater than 0 and at most 1.");
    }
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > LIMITS.epochs)
        throw new Error("Use 1 to 10,000 epochs.");
    const parameters = sizes
        .slice(1)
        .reduce((total, size, i) => total + size * (sizes[i] + 1), 0);
    if (parameters * x.length * epochs > LIMITS.work)
        throw new Error(
            "This run is too large. Reduce points, neurons, or epochs.",
        );
    return { x: [...x], y: [...y], layers, learningRate, epochs };
}

function splitData(x, y) {
    // Fixed seed keeps both models and repeated runs on exactly the same split.
    let seed = 42;
    const indices = x.map((_, index) => index);
    for (let i = indices.length - 1; i > 0; i--) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        const j = Math.floor((seed / 4294967296) * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const testCount = Math.max(2, Math.round(x.length * 0.2));
    const points = indices.map((index) => ({
        x: x[index],
        y: y[index],
        index,
    }));
    return { train: points.slice(testCount), test: points.slice(0, testCount) };
}

function scaler(values) {
    const min = Math.min(...values);
    return { min, range: Math.max(...values) - min || 1 };
}
const scale = (value, stats) => (value - stats.min) / stats.range;
const unscale = (value, stats) => value * stats.range + stats.min;

function metrics(points, predictions) {
    if (
        predictions.length !== points.length ||
        predictions.some((value) => !Number.isFinite(value))
    ) {
        throw new Error("The engine returned invalid predictions.");
    }
    const mean =
        points.reduce((sum, point) => sum + point.y, 0) / points.length;
    const sse = points.reduce(
        (sum, point, i) => sum + (point.y - predictions[i]) ** 2,
        0,
    );
    const total = points.reduce((sum, point) => sum + (point.y - mean) ** 2, 0);
    const mse = sse / points.length;
    if (!Number.isFinite(mse))
        throw new Error("Training diverged. Try a smaller learning rate.");
    return { mse, r2: total === 0 ? null : 1 - sse / total };
}

module.exports = {
    LIMITS,
    validateDataset,
    splitData,
    scaler,
    scale,
    unscale,
    metrics,
};
