const LIMITS = Object.freeze({
    points: 1000,
    magnitude: 1e6,
    epochs: 10000,
    // Parameters × points × epochs; the largest runs take about 3 s of engine time.
    work: 2.5e8,
    weightDecay: 1,
});

const ACTIVATIONS = Object.freeze(["sigmoid", "tanh", "relu"]);
const OPTIMIZERS = Object.freeze(["sgd", "adam"]);

function validateDataset(body) {
    const {
        x,
        y,
        model = "both",
        layers = "1-8-1",
        learningRate = 0.3,
        epochs = 3000,
        activation = "sigmoid",
        weightDecay = 0,
        patience = 0,
        optimizer = "sgd",
        batchSize = 1,
    } = body || {};
    if (
        !Array.isArray(x) ||
        !Array.isArray(y) ||
        x.length < 2 ||
        x.length > LIMITS.points ||
        x.length !== y.length
    ) {
        throw new Error("Enter 2 to 1,000 matching X and Y values.");
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
    if (!["lr", "nn", "both"].includes(model))
        throw new Error("Choose lr, nn, or both models.");
    if (new Set(x).size < 2)
        throw new Error("Use at least two distinct X values.");
    if (model === "lr") return { x: [...x], y: [...y], model };
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
    if (!ACTIVATIONS.includes(activation))
        throw new Error("Choose a sigmoid, tanh, or relu activation.");
    if (
        typeof weightDecay !== "number" ||
        !Number.isFinite(weightDecay) ||
        weightDecay < 0 ||
        weightDecay > LIMITS.weightDecay
    )
        throw new Error("Weight decay must be between 0 and 1.");
    if (!Number.isInteger(patience) || patience < 0 || patience > LIMITS.epochs)
        throw new Error("Patience must be a whole number from 0 to 10,000.");
    if (!OPTIMIZERS.includes(optimizer))
        throw new Error("Choose the sgd or adam optimizer.");
    if (
        !Number.isInteger(batchSize) ||
        batchSize < 1 ||
        batchSize > LIMITS.points
    )
        throw new Error("Batch size must be a whole number from 1 to 1,000.");
    if (patience > 0 && splitSizes(x.length).validation === 0)
        throw new Error(
            "Early stopping needs at least 12 points so a validation set can be held out.",
        );
    const parameters = sizes
        .slice(1)
        .reduce((total, size, i) => total + size * (sizes[i] + 1), 0);
    if (parameters * x.length * epochs > LIMITS.work)
        throw new Error(
            "This run is too large. Reduce points, neurons, or epochs.",
        );
    return {
        x: [...x],
        y: [...y],
        layers,
        learningRate,
        epochs,
        activation,
        weightDecay,
        patience,
        optimizer,
        batchSize,
        model,
    };
}

// Test points are held out from every model. The network also holds out validation
// points from its training data to watch for overfitting and to stop early.
function splitSizes(count) {
    const test = count < 5 ? 0 : Math.max(1, Math.round(count * 0.2));
    const train = count - test;
    const validation = train < 10 ? 0 : Math.max(2, Math.round(train * 0.2));
    return { test, train, validation };
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
    const sizes = splitSizes(x.length);
    const points = indices.map((index) => ({
        x: x[index],
        y: y[index],
        index,
    }));
    const train = points.slice(sizes.test);
    // Validation is a subset of the training points: linear regression has nothing to
    // tune and fits on all of them, while the network learns only from the rest.
    return {
        train,
        test: points.slice(0, sizes.test),
        validation: train.slice(0, sizes.validation),
        fit: train.slice(sizes.validation),
    };
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
    if (points.length === 0) return { mse: null, r2: null };
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
    splitSizes,
    splitData,
    scaler,
    scale,
    unscale,
    metrics,
};
