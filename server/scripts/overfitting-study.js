// Does validation-based early stopping or weight decay actually help? Trains the overfitting
// preset on many independently drawn noisy datasets through the real job pipeline and compares
// held-out test MSE, pairing every run with the unregularized run on the same data.
//
//   node scripts/overfitting-study.js [datasets=100]
//
// The seeds start at 100001; the settings were chosen on other seeds, so these are fresh data.

const os = require("node:os");
const { Jobs } = require("../jobs");

// The client's "Noisy wave" sample generator: 2 sin(x) plus uniform noise in [-1.25, 1.25].
function noisyWave(seed, count = 40) {
    const x = [],
        y = [];
    for (let i = 0; i < count; i++) {
        const value = -3 + (6 * i) / (count - 1);
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        x.push(Number(value.toFixed(4)));
        y.push(
            Number(
                (2 * Math.sin(value) + (seed / 4294967296 - 0.5) * 2.5).toFixed(
                    4,
                ),
            ),
        );
    }
    return { x, y };
}

const preset = {
    layers: "1-32-32-1",
    activation: "tanh",
    optimizer: "adam",
    batchSize: 1000,
    learningRate: 0.01,
    epochs: 5000,
};
const variants = [
    ["No regularization", {}],
    ["Early stopping (patience 1000)", { patience: 1000 }],
    ["Weight decay 1", { weightDecay: 1 }],
];

const jobs = new Jobs({ maxActive: Infinity, timeoutMs: 120000 });
async function train(data) {
    const job = jobs.create(Symbol("study"), data, () => {});
    while (job.status === "running")
        await new Promise((resolve) => setTimeout(resolve, 5));
    jobs.jobs.delete(job.id);
    if (job.status !== "completed") throw new Error(job.error);
    return job.result;
}

async function pool(tasks, limit) {
    const results = new Array(tasks.length);
    let next = 0;
    const worker = async () => {
        while (next < tasks.length) {
            const index = next++;
            results[index] = await tasks[index]();
        }
    };
    await Promise.all(Array.from({ length: limit }, worker));
    return results;
}

const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
const interval = (values) => {
    const m = mean(values);
    const sd = Math.sqrt(
        values.reduce((sum, v) => sum + (v - m) ** 2, 0) / (values.length - 1),
    );
    return [
        m - (1.96 * sd) / Math.sqrt(values.length),
        m + (1.96 * sd) / Math.sqrt(values.length),
    ];
};
const f = (value) => value.toFixed(3);

async function main() {
    const count = Number(process.argv[2] || 100);
    const datasets = Array.from({ length: count }, (_, i) =>
        noisyWave(100001 + i),
    );
    const started = Date.now();
    const runs = await pool(
        datasets.flatMap((data) =>
            variants.map(
                ([, options], v) =>
                    () =>
                        train({
                            ...data,
                            ...preset,
                            ...options,
                            model: v === 0 ? "both" : "nn",
                        }),
            ),
        ),
        Math.max(1, os.availableParallelism() - 1),
    );
    const byVariant = variants.map((_, v) =>
        runs.filter((_, i) => i % variants.length === v),
    );
    const baseline = byVariant[0].map((r) => r.nn.test.mse);
    // The noise floor: the error of the true function 2 sin(x) on the same test points.
    const floor = byVariant[0].map((r) =>
        mean(r.split.test.map((p) => (p.y - 2 * Math.sin(p.x)) ** 2)),
    );
    console.log(
        `${count} noisy-wave datasets (40 points: ${byVariant[0][0].split.fit.length} fit, ` +
            `${byVariant[0][0].split.validation.length} validation, ${byVariant[0][0].split.test.length} test), ` +
            `${Math.round((Date.now() - started) / 1000)} s\n`,
    );
    console.log(
        "| Model | Mean test MSE | Change vs. no regularization (95% CI) | Better on |",
    );
    console.log("| --- | --- | --- | --- |");
    const lr = byVariant[0].map((r) => r.lr.test.mse);
    const row = (name, values) => {
        if (values === baseline) {
            console.log(`| ${name} | ${f(mean(values))} | — | — |`);
            return;
        }
        const diff = values.map((v, i) => v - baseline[i]);
        const [low, high] = interval(diff);
        const better = diff.filter((d) => d < 0).length;
        console.log(
            `| ${name} | ${f(mean(values))} | ${f(mean(diff))} (${f(low)} to ${f(high)}) | ${better}/${count} |`,
        );
    };
    row(
        "Linear regression",
        byVariant[0].map((r) => r.lr.test.mse),
    );
    row("Network, no regularization", baseline);
    variants.slice(1).forEach(([name], v) =>
        row(
            `Network, ${name.toLowerCase()}`,
            byVariant[v + 1].map((r) => r.nn.test.mse),
        ),
    );
    console.log(`| Noise floor (true function) | ${f(mean(floor))} | | |`);
    const stopped = byVariant[1].map((r) => r.nn.epochsRun);
    console.log(
        `\nEarly stopping ran ${Math.round(mean(stopped))} of 5000 epochs on average.`,
    );
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
