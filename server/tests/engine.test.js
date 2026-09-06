const { test } = require("node:test");
const assert = require("node:assert/strict");
const { runEngine } = require("../engine");

test("buffers fragmented stdout and flushes its final line", async () => {
    const loss = [];
    const script =
        'process.stdout.write("epo"); setTimeout(() => process.stdout.write("ch=2,mse=0.5\\nvalue=3"), 10)';
    const output = await runEngine(["-e", script], "", {
        command: process.execPath,
        onLoss: (p) => loss.push(p),
    });
    assert.deepEqual(loss, [{ epoch: 2, mse: 0.5 }]);
    assert.equal(output.value, "3");
});

test("handles failed exit, malformed progress, cancellation and excess output", async () => {
    await assert.rejects(
        runEngine(
            ["-e", 'process.stderr.write("broken\\n");process.exit(2)'],
            "",
            {
                command: process.execPath,
            },
        ),
        /broken/,
    );
    await assert.rejects(
        runEngine(["-e", 'console.log("epoch=2,mse=nan")'], "", {
            command: process.execPath,
        }),
        /Invalid training progress/,
    );
    const controller = new AbortController();
    const pending = runEngine(["-e", "setTimeout(() => {}, 10000)"], "", {
        command: process.execPath,
        signal: controller.signal,
    });
    controller.abort();
    await assert.rejects(pending, /cancelled/);
    await assert.rejects(
        runEngine(["-e", 'process.stdout.write("x".repeat(2100000))'], "", {
            command: process.execPath,
        }),
        /output limit/,
    );
});
