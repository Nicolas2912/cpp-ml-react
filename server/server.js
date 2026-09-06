const express = require("express");
const http = require("node:http");
const { randomUUID } = require("node:crypto");
const { WebSocketServer, WebSocket } = require("ws");
const { validateDataset, LIMITS } = require("./data");
const { Jobs } = require("./jobs");

function createServer(options = {}) {
    const app = express();
    const server = http.createServer(app);
    const sessions = new Map();
    const jobs = new Jobs(options);
    const wss = new WebSocketServer({ noServer: true, maxPayload: 1024 });
    // The Vite proxy keeps browser HTTP and WebSocket traffic on one origin.
    server.on("upgrade", (req, socket, head) => {
        const origin = req.headers.origin;
        let sameOrigin = !origin;
        try {
            if (origin) sameOrigin = new URL(origin).host === req.headers.host;
        } catch {
            sameOrigin = false;
        }
        if (req.url !== "/ws" || !sameOrigin || sessions.size >= 32)
            return socket.destroy();
        wss.handleUpgrade(req, socket, head, (ws) =>
            wss.emit("connection", ws),
        );
    });
    wss.on("connection", (ws) => {
        const token = randomUUID();
        sessions.set(token, ws);
        ws.isAlive = true;
        ws.on("pong", () => {
            ws.isAlive = true;
        });
        ws.on("error", () => ws.terminate());
        ws.on("close", () => {
            sessions.delete(token);
            jobs.removeOwner(token);
        });
        ws.send(JSON.stringify({ type: "session", token }));
    });
    const maintenance = setInterval(() => {
        jobs.sweep();
        for (const ws of sessions.values()) {
            if (!ws.isAlive) {
                ws.terminate();
                continue;
            }
            ws.isAlive = false;
            ws.ping();
        }
    }, 30000);
    maintenance.unref();
    server.on("close", () => clearInterval(maintenance));
    app.use(express.json({ limit: "100kb" }));
    app.get("/api/health", (req, res) => res.json({ status: "ok" }));
    app.use("/api", (req, res, next) => {
        const token = req.headers.authorization?.replace(/^Bearer /, "");
        if (!sessions.has(token))
            return res.status(401).json({
                error: "Connection expired. Reconnect and train again.",
            });
        req.owner = token;
        next();
    });
    app.post("/api/jobs", (req, res) => {
        let data;
        try {
            data = validateDataset(req.body);
        } catch (error) {
            return res.status(400).json({ error: error.message });
        }
        try {
            const ws = sessions.get(req.owner);
            const job = jobs.create(req.owner, data, (event) => {
                if (ws.readyState === WebSocket.OPEN)
                    ws.send(JSON.stringify(event));
            });
            res.status(202).json({ id: job.id });
        } catch (error) {
            res.status(429).json({ error: error.message });
        }
    });
    app.use("/api/jobs/:id", (req, res, next) => {
        req.job = jobs.owned(req.params.id, req.owner);
        if (!req.job)
            return res
                .status(404)
                .json({ error: "Run not found or expired. Train again." });
        next();
    });
    app.get("/api/jobs/:id", (req, res) => res.json(jobs.snapshot(req.job)));
    app.post("/api/jobs/:id/cancel", (req, res) => {
        jobs.cancel(req.job);
        res.json(jobs.snapshot(req.job));
    });
    app.post("/api/jobs/:id/predict", async (req, res) => {
        const x = req.body?.x;
        const model = req.body?.model;
        if (model !== undefined && !["lr", "nn", "both"].includes(model))
            return res
                .status(400)
                .json({ error: "Choose lr, nn, or both models." });
        if (
            typeof x !== "number" ||
            !Number.isFinite(x) ||
            Math.abs(x) > LIMITS.magnitude
        ) {
            return res.status(400).json({
                error: "Enter a finite X between −1,000,000 and 1,000,000.",
            });
        }
        if (req.job.status !== "completed" || req.job.predicting)
            return res
                .status(409)
                .json({ error: "Wait for the current operation to finish." });
        try {
            res.json(await jobs.predict(req.job, x, model));
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });
    app.use((req, res) =>
        res.status(404).json({ error: "Endpoint not found." }),
    );
    app.use((error, req, res, next) => {
        res.status(error.type === "entity.too.large" ? 413 : 400).json({
            error: "Send a valid JSON dataset under 100 KB.",
        });
    });
    return {
        server,
        jobs,
        close: () =>
            new Promise((resolve) => {
                for (const [token, ws] of sessions) {
                    jobs.removeOwner(token);
                    ws.terminate();
                }
                wss.close();
                server.close(resolve);
                server.closeAllConnections();
            }),
    };
}

if (require.main === module) {
    const instance = createServer();
    const port = Number(process.env.PORT || 3001);
    instance.server.listen(port, "127.0.0.1", () =>
        console.log(`ML engine API: http://127.0.0.1:${port}`),
    );
    for (const signal of ["SIGINT", "SIGTERM"])
        process.once(signal, () =>
            instance.close().then(() => process.exit(0)),
        );
}
module.exports = { createServer };
