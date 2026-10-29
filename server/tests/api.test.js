const { test } = require("node:test");
const assert = require("node:assert/strict");
const { once } = require("node:events");
const WebSocket = require("ws");
const { createServer } = require("../server");
const { runEngine, numbers } = require("../engine");
const { scale, unscale, scaler, splitSizes } = require("../data");

async function setup(t, options = {}) {
    const app = createServer(options);
    app.server.listen(0, "127.0.0.1");
    await once(app.server, "listening");
    t.after(() => app.close());
    const port = app.server.address().port;
    const connect = async () => {
        const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
        const events = [];
        ws.on("message", (value) => events.push(JSON.parse(value)));
        await once(ws, "message");
        const token = events[0].token;
        const request = async (path, body) => {
            const response = await fetch(
                `http://127.0.0.1:${port}/api${path}`,
                {
                    method: body === undefined ? "GET" : "POST",
                    headers: {
                        "Content-Type": "application/json",
                        Authorization: `Bearer ${token}`,
                    },
                    ...(body === undefined
                        ? {}
                        : { body: JSON.stringify(body) }),
                },
            );
            return { status: response.status, body: await response.json() };
        };
        return { ws, events, token, request };
    };
    return { ...app, connect, port };
}
const data = {
    x: Array.from({ length: 30 }, (_, i) => i / 5 - 3),
    y: Array.from({ length: 30 }, (_, i) => (i / 5 - 3) ** 2 + 10),
    layers: "1-8-1",
    epochs: 1000,
    learningRate: 0.1,
};
async function waitFor(client, id, status = "completed") {
    const deadline = Date.now() + 10000;
    while (Date.now() < deadline) {
        const state = await client.request(`/jobs/${id}`);
        if (state.body.status === status) return state.body;
        if (
            ["failed", "cancelled"].includes(state.body.status) &&
            state.body.status !== status
        )
            throw new Error(state.body.error || state.body.status);
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
    throw new Error(`Timed out waiting for ${status}`);
}

test("real C++ comparison streams original-unit loss and predicts with restored weights", async (t) => {
    const app = await setup(t);
    const owner = await app.connect(),
        other = await app.connect();
    const created = await owner.request("/jobs", data);
    assert.equal(created.status, 202);
    const { id } = created.body;
    assert.equal((await other.request(`/jobs/${id}`)).status, 404);
    assert.equal((await other.request(`/jobs/${id}/cancel`, {})).status, 404);
    assert.equal(
        (await other.request(`/jobs/${id}/predict`, { x: 0 })).status,
        404,
    );
    const state = await waitFor(owner, id);
    assert.equal(state.result.split.test.length, 6);
    assert.ok(
        owner.events.some(
            (event) => event.type === "progress" && event.id === id,
        ),
    );
    assert.ok(!other.events.some((event) => event.id === id));
    const saved = app.jobs.owned(id, owner.token);
    const originalMse =
        state.result.split.test.reduce(
            (sum, p) =>
                sum +
                (p.y - (saved.model.slope * p.x + saved.model.intercept)) ** 2,
            0,
        ) / 6;
    assert.ok(Math.abs(originalMse - state.result.lr.test.mse) < 1e-10);
    assert.ok(
        Math.abs(state.loss.at(-1).mse - state.result.nn.train.mse) < 1e-9,
    );
    for (const x of [0.137, 5, -5]) {
        const response = await owner.request(`/jobs/${id}/predict`, { x });
        assert.equal(response.status, 200);
        const direct = await runEngine(
            ["nn_predict"],
            `${saved.model.serialized}\n${scale(x, saved.model.sx)}\n`,
        );
        const expected = unscale(
            numbers(direct.predictions, 1)[0],
            saved.model.sy,
        );
        assert.equal(response.body.nn, expected);
        assert.ok(
            Math.abs(
                response.body.lr -
                    (saved.model.slope * x + saved.model.intercept),
            ) < 1e-10,
        );
        assert.equal(response.body.extrapolation, Math.abs(x) > 3);
    }
    const gridPoint = state.result.nn.curve[37];
    const gridPrediction = await owner.request(`/jobs/${id}/predict`, {
        x: gridPoint.x,
    });
    assert.ok(Math.abs(gridPrediction.body.nn - gridPoint.y) < 1e-9);
    assert.equal(
        (await owner.request(`/jobs/${id}/predict`, { x: "hello" })).status,
        400,
    );
    const second = await other.request("/jobs", {
        ...data,
        y: data.y.map((v) => v + 100),
    });
    const secondState = await waitFor(other, second.body.id);
    assert.ok(
        Math.abs(
            secondState.result.lr.intercept - state.result.lr.intercept - 100,
        ) < 1e-9,
    );
    const repeated = await owner.request("/jobs", data);
    const repeatState = await waitFor(owner, repeated.body.id);
    assert.deepEqual(repeatState.result.nn.curve, state.result.nn.curve);
});

test("validation, cancellation, ownership and disconnected-session cleanup", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    for (const body of [
        {},
        { ...data, epochs: 0 },
        { ...data, y: [1] },
        { ...data, layers: "1-999-1" },
    ]) {
        assert.equal((await owner.request("/jobs", body)).status, 400);
    }
    const created = await owner.request("/jobs", { ...data, epochs: 10000 });
    const id = created.body.id;
    assert.equal((await owner.request("/jobs", data)).status, 429);
    const cancelled = await owner.request(`/jobs/${id}/cancel`, {});
    assert.equal(cancelled.body.status, "cancelled");
    assert.equal(
        (await owner.request(`/jobs/${id}/predict`, { x: 1 })).status,
        409,
    );
    const closed = once(owner.ws, "close");
    owner.ws.close();
    await closed;
    // The client can observe "close" before the server's close handler runs.
    for (let i = 0; i < 100 && app.jobs.jobs.size > 0; i++) {
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.equal(app.jobs.jobs.size, 0);
    assert.equal((await owner.request(`/jobs/${id}`)).status, 401);
});

test("process launch errors and timeouts become terminal job failures", async (t) => {
    for (const options of [
        { command: "/no-such-cpp-engine" },
        { timeoutMs: 1 },
    ]) {
        await t.test(JSON.stringify(options), async (t) => {
            const app = await setup(t, options);
            const owner = await app.connect();
            const { body } = await owner.request("/jobs", {
                ...data,
                epochs: 10000,
            });
            const state = await waitFor(owner, body.id, "failed");
            assert.match(
                state.error,
                options.command
                    ? /Could not start/
                    : /timed out after 1 second\./,
            );
        });
    }
});

test("expired jobs and global capacity are enforced", async (t) => {
    const app = await setup(t, { maxActive: 1 });
    const owner = await app.connect(),
        other = await app.connect();
    const { body } = await owner.request("/jobs", { ...data, epochs: 10000 });
    assert.equal((await other.request("/jobs", data)).status, 429);
    await owner.request(`/jobs/${body.id}/cancel`, {});
    app.jobs.jobs.get(body.id).created = 0;
    assert.equal((await owner.request(`/jobs/${body.id}`)).status, 404);
});

test("constant targets return finite MSE and undefined R²", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    const { body } = await owner.request("/jobs", {
        ...data,
        y: data.y.map(() => 7),
    });
    const state = await waitFor(owner, body.id);
    assert.equal(state.result.lr.test.r2, null);
    assert.equal(state.result.nn.test.r2, null);
    assert.equal(state.result.lr.test.mse, 0);
    assert.ok(Number.isFinite(state.result.nn.test.mse));
});

test("restored independent LR and NN workflows retain both models", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    const lrCreated = await owner.request("/jobs", {
        ...data,
        model: "lr",
        layers: "invalid",
        epochs: -1,
    });
    const lr = await waitFor(owner, lrCreated.body.id);
    assert.ok(lr.result.lr);
    assert.equal(lr.result.nn, null);
    assert.equal(lr.loss.length, 0);
    let nn;
    for (let i = 0; i < 4; i++) {
        const created = await owner.request("/jobs", {
            ...data,
            model: "nn",
            layers: "1-6-4-1",
            epochs: 100,
        });
        nn = await waitFor(owner, created.body.id);
        assert.equal(nn.result.lr, null);
        assert.ok(nn.result.nn);
        assert.equal(nn.result.settings.layers, "1-6-4-1");
    }
    assert.deepEqual(lr.result.split, nn.result.split);
    assert.equal(
        (
            await owner.request(`/jobs/${lr.id}/predict`, {
                x: 0.137,
                model: "lr",
            })
        ).status,
        200,
    );
    const prediction = await owner.request(`/jobs/${nn.id}/predict`, {
        x: 0.137,
        model: "nn",
    });
    assert.equal(prediction.status, 200);
    assert.ok(Number.isFinite(prediction.body.nn));
    assert.equal(prediction.body.lr, undefined);
    assert.equal(
        (await owner.request("/jobs", { ...data, model: "invalid" })).status,
        400,
    );
});

test("history eviction keeps the latest completed LR and NN models", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    const train = async (body) =>
        (await owner.request("/jobs", { ...data, ...body })).body.id;
    const lr = await train({ model: "lr" });
    await waitFor(owner, lr);
    await waitFor(owner, await train({ model: "nn", epochs: 100 }));
    // Cancel a comparison once its linear part is done and the network is training.
    const both = await train({ epochs: 10000 });
    const deadline = Date.now() + 10000;
    while (
        !owner.events.some(
            (event) => event.type === "progress" && event.id === both,
        )
    ) {
        assert.ok(Date.now() < deadline, "comparison never reached the NN");
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    await owner.request(`/jobs/${both}/cancel`, {});
    const cancelled = await waitFor(owner, both, "cancelled");
    assert.ok(cancelled.result.lr);
    await waitFor(owner, await train({ model: "nn", epochs: 100 }));
    assert.equal((await owner.request(`/jobs/${both}`)).status, 404);
    const prediction = await owner.request(`/jobs/${lr}/predict`, {
        x: 0.137,
        model: "lr",
    });
    assert.equal(prediction.status, 200);
    const missing = await owner.request(`/jobs/${lr}/predict`, {
        x: 0.137,
        model: "nn",
    });
    assert.equal(missing.status, 400);
    assert.equal(missing.body.error, "This run has no trained neural model.");
});

test("models with subnormal weights reload for prediction", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    // Weight decay shrinks dead ReLU weights below the smallest normal double.
    const x = Array.from({ length: 60 }, (_, i) => i / 59);
    const { body } = await owner.request("/jobs", {
        x,
        y: x.map((value) => 0.5 + 0.4 * Math.sin(6 * value)),
        model: "nn",
        layers: "1-8-1",
        activation: "relu",
        learningRate: 0.3,
        weightDecay: 0.1,
        epochs: 3000,
    });
    await waitFor(owner, body.id);
    const weights = app.jobs.jobs
        .get(body.id)
        .model.serialized.split(" ")
        .map(Number);
    assert.ok(
        weights.some((value) => value !== 0 && Math.abs(value) < 2.3e-308),
    );
    for (const value of [0.25, 1e-310]) {
        const prediction = await owner.request(`/jobs/${body.id}/predict`, {
            x: value,
            model: "nn",
        });
        assert.equal(prediction.status, 200, prediction.body.error);
        assert.ok(Number.isFinite(prediction.body.nn));
    }
});

test("original two-point and five-point datasets remain supported with honest test scores", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    for (const n of [2, 5]) {
        const created = await owner.request("/jobs", {
            x: [1, 2, 3, 4, 5].slice(0, n),
            y: [2, 4, 6, 8, 10].slice(0, n),
            epochs: 100,
        });
        assert.equal(created.status, 202);
        const state = await waitFor(owner, created.body.id);
        assert.equal(state.result.split.test.length, n === 2 ? 0 : 1);
        assert.equal(state.result.lr.test.r2, null);
        assert.equal(state.result.nn.test.r2, null);
        if (n === 2) assert.equal(state.result.lr.test.mse, null);
        else assert.equal(state.result.lr.test.mse, 0);
    }
});

test("each hidden activation trains and predicts through the reloaded model", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    for (const activation of ["sigmoid", "tanh", "relu"]) {
        const created = await owner.request("/jobs", {
            ...data,
            model: "nn",
            activation,
        });
        assert.equal(created.status, 202);
        const state = await waitFor(owner, created.body.id);
        assert.equal(state.result.settings.activation, activation);
        // y = x^2 + 10 over [-3, 3]: a working network beats predicting the mean (variance ~ 6.5).
        assert.ok(
            state.result.nn.train.mse < 1,
            `${activation} train MSE ${state.result.nn.train.mse}`,
        );
        const prediction = await owner.request(`/jobs/${state.id}/predict`, {
            x: 0,
            model: "nn",
        });
        assert.equal(prediction.status, 200);
        assert.ok(
            Math.abs(prediction.body.nn - 10) < 1.5,
            `${activation} predicts ${prediction.body.nn} at x=0`,
        );
    }
});

// A noisy sine like the client's "Noisy wave" sample: few points, so a wide network overfits.
function noisyWave(count = 40, seed = 12) {
    const x = [],
        y = [];
    for (let i = 0; i < count; i++) {
        const value = -3 + (6 * i) / (count - 1);
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        x.push(value);
        y.push(2 * Math.sin(value) + (seed / 4294967296 - 0.5) * 2.5);
    }
    return { x, y };
}
const lab = {
    ...noisyWave(),
    model: "nn",
    layers: "1-16-16-1",
    activation: "tanh",
    optimizer: "adam",
    batchSize: 1000,
    learningRate: 0.02,
    epochs: 1500,
};
const near = (a, b, message) =>
    assert.ok(
        Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)),
        `${message}: ${a} vs ${b}`,
    );

test("network holds out validation points and streams validation loss", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    const created = await owner.request("/jobs", lab);
    assert.equal(created.status, 202);
    const state = await waitFor(owner, created.body.id);
    const { split, nn, settings } = state.result;
    assert.equal(split.validation.length, splitSizes(40).validation);
    assert.equal(
        split.fit.length + split.validation.length,
        split.train.length,
    );
    const test = new Set(split.test.map((p) => p.index));
    assert.ok(split.validation.every((p) => !test.has(p.index)));
    assert.deepEqual(
        [settings.optimizer, settings.batchSize, settings.patience],
        ["adam", 1000, 0],
    );
    // Every progress point carries both curves; without early stopping the last one is the final model.
    assert.ok(state.loss.length >= 100);
    assert.ok(state.loss.every((p) => Number.isFinite(p.valMse)));
    const progress = owner.events.filter((e) => e.type === "progress");
    assert.ok(progress.every((e) => Number.isFinite(e.valMse)));
    near(state.loss.at(-1).mse, nn.train.mse, "final training MSE");
    near(state.loss.at(-1).valMse, nn.validation.mse, "final validation MSE");
    assert.equal(nn.epochsRun, 1500);
    assert.ok(nn.bestEpoch >= 1 && nn.bestEpoch <= 1500);
    assert.equal(nn.stoppedEarly, false);
    assert.equal(nn.restoredBest, false);
    // The network's scalers come from the points it learns from, not from validation or test points.
    const saved = app.jobs.owned(state.id, owner.token);
    assert.deepEqual(saved.model.sx, scaler(split.fit.map((p) => p.x)));
    assert.deepEqual(saved.model.sy, scaler(split.fit.map((p) => p.y)));
});

test("early stopping restores the best validation epoch through the real engine", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    const patience = 100;
    const created = await owner.request("/jobs", {
        ...lab,
        epochs: 4000,
        patience,
    });
    const state = await waitFor(owner, created.body.id);
    const { nn } = state.result;
    assert.equal(nn.stoppedEarly, true);
    assert.equal(nn.restoredBest, true);
    assert.equal(nn.epochsRun, nn.bestEpoch + patience);
    assert.equal(state.loss.at(-1).epoch, nn.epochsRun);
    // The restored weights score the minimum validation loss seen at any epoch.
    const lowest = Math.min(...state.loss.map((p) => p.valMse));
    assert.ok(nn.validation.mse <= lowest * (1 + 1e-12));
    assert.ok(nn.validation.mse < state.loss.at(-1).valMse);
    // Predictions come from the restored model, matching its stored curve.
    const point = nn.curve[50];
    const prediction = await owner.request(`/jobs/${state.id}/predict`, {
        x: point.x,
        model: "nn",
    });
    near(prediction.body.nn, point.y, "restored model prediction");
});

test("weight decay shrinks the network and invalid regularization is rejected", async (t) => {
    const app = await setup(t);
    const owner = await app.connect();
    const norm = async (weightDecay) => {
        const created = await owner.request("/jobs", { ...lab, weightDecay });
        const state = await waitFor(owner, created.body.id);
        assert.equal(state.result.settings.weightDecay, weightDecay);
        const serialized = app.jobs.owned(state.id, owner.token).model
            .serialized;
        // "4 1 16 16 1 tanh p1 p2 ...": skip the layer count, sizes, and activation.
        const parameters = serialized.split(" ").slice(6).map(Number);
        return parameters.reduce((sum, value) => sum + value * value, 0);
    };
    const plain = await norm(0);
    const decayed = await norm(0.5);
    assert.ok(decayed < 0.5 * plain, `${decayed} vs ${plain}`);
    for (const body of [
        { ...lab, weightDecay: 2 },
        { ...lab, optimizer: "rmsprop" },
        { ...lab, batchSize: 0 },
        { ...lab, patience: 1.5 },
        { ...lab, x: lab.x.slice(0, 11), y: lab.y.slice(0, 11), patience: 5 },
    ]) {
        assert.equal((await owner.request("/jobs", body)).status, 400);
    }
});

test("serves the built client with SPA fallback when staticDir is set", async (t) => {
    const fs = require("node:fs");
    const os = require("node:os");
    const path = require("node:path");
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ml-static-"));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    fs.writeFileSync(path.join(dir, "index.html"), "<!doctype html>app");
    fs.writeFileSync(path.join(dir, "asset.js"), "console.log(1)");
    const app = await setup(t, { staticDir: dir });
    const base = `http://127.0.0.1:${app.port}`;
    assert.equal(await (await fetch(`${base}/`)).text(), "<!doctype html>app");
    assert.equal(
        await (await fetch(`${base}/asset.js`)).text(),
        "console.log(1)",
    );
    assert.equal(
        await (await fetch(`${base}/some/route`)).text(),
        "<!doctype html>app",
    );
    assert.equal((await fetch(`${base}/api/health`)).status, 200);
    assert.equal((await fetch(`${base}/api/missing`)).status, 401);
});
