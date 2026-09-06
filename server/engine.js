const { spawn } = require("node:child_process");
const path = require("node:path");

const executable = path.join(
    __dirname,
    "..",
    "cpp",
    `linear_regression_app${process.platform === "win32" ? ".exe" : ""}`,
);

function runEngine(
    args,
    input,
    { signal, onLoss = () => {}, timeoutMs = 30000, command = executable } = {},
) {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) return reject(new Error("Training cancelled."));
        const child = spawn(command, args, {
            env: { ...process.env, OMP_NUM_THREADS: "1" },
            stdio: ["pipe", "pipe", "pipe"],
        });
        const result = {};
        let buffer = "";
        let stderr = "";
        let bytes = 0;
        let failure;
        let settled = false;
        const stop = (message) => {
            failure = new Error(message);
            child.kill("SIGKILL");
        };
        const cancel = () => stop("Training cancelled.");
        const timer = setTimeout(
            () =>
                stop(
                    "Training timed out after 30 seconds. Reduce epochs or network size.",
                ),
            timeoutMs,
        );
        signal?.addEventListener("abort", cancel, { once: true });
        const finish = (error) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            signal?.removeEventListener("abort", cancel);
            if (error) reject(error);
            else resolve(result);
        };
        const parse = (line) => {
            if (line.startsWith("epoch=")) {
                const match = /^epoch=(\d+),mse=([^\s]+)$/.exec(line);
                if (!match || !Number.isFinite(Number(match[2])))
                    throw new Error("Invalid training progress from engine.");
                onLoss({ epoch: Number(match[1]), mse: Number(match[2]) });
            } else if (line.includes("=")) {
                const at = line.indexOf("=");
                result[line.slice(0, at)] = line.slice(at + 1);
            }
        };
        child.stdout.on("data", (chunk) => {
            bytes += chunk.length;
            if (bytes > 2e6) return stop("Engine output limit exceeded.");
            buffer += chunk.toString();
            const lines = buffer.split("\n");
            buffer = lines.pop();
            try {
                lines.forEach(parse);
            } catch (error) {
                stop(error.message);
            }
        });
        child.stderr.on("data", (chunk) => {
            stderr = (stderr + chunk).slice(-4096);
        });
        child.stdin.on("error", () => {
            /* Process exit/error carries the useful failure. */
        });
        child.on("error", () =>
            finish(
                new Error("Could not start the C++ engine. Run make -C cpp."),
            ),
        );
        child.on("close", (code) => {
            if (failure) return finish(failure);
            if (code !== 0)
                return finish(
                    new Error(stderr.split("\n")[0] || "C++ training failed."),
                );
            try {
                if (buffer) parse(buffer);
                finish();
            } catch (error) {
                finish(error);
            }
        });
        child.stdin.end(input);
    });
}

function numbers(value, expected) {
    const parsed =
        typeof value === "string" && value.length
            ? value.split(",").map(Number)
            : [];
    if (parsed.length !== expected || parsed.some((n) => !Number.isFinite(n)))
        throw new Error("The engine returned incomplete or invalid results.");
    return parsed;
}

module.exports = { runEngine, numbers };
