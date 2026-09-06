const { test } = require("node:test");
const assert = require("node:assert/strict");
const { once } = require("node:events");
const WebSocket = require("ws");
const { createServer } = require("../server");
const { runEngine, numbers } = require("../engine");
const { scale, unscale } = require("../data");

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
                options.command ? /Could not start/ : /timed out/,
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
