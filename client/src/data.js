export function sampleData(kind = "curve", count = 60, noise = 0.15) {
  let seed = 17;
  const x = [],
    y = [];
  for (let i = 0; i < count; i++) {
    const value = -3 + (6 * i) / (count - 1);
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const offset = (seed / 4294967296 - 0.5) * noise * 5;
    const target =
      kind === "line"
        ? 1.4 * value + 2
        : kind === "wave"
          ? 2 * Math.sin(value)
          : 0.7 * value * value - 1;
    x.push(Number(value.toFixed(4)));
    y.push(Number((target + offset).toFixed(4)));
  }
  return { x: x.join(", "), y: y.join(", ") };
}

export function parseData(text) {
  const parse = (value) => {
    if (!value.trim()) return [];
    return value
      .split(",")
      .map((token) => (token.trim() === "" ? NaN : Number(token.trim())));
  };
  const x = parse(text.x),
    y = parse(text.y);
  if (x.length < 2 || x.length > 1000 || x.length !== y.length)
    throw new Error("Enter 2 to 1,000 matching X and Y values.");
  if (
    [...x, ...y].some(
      (value) => !Number.isFinite(value) || Math.abs(value) > 1e6,
    )
  )
    throw new Error(
      "Use finite numbers between −1,000,000 and 1,000,000, separated by commas.",
    );
  if (new Set(x).size < 2)
    throw new Error("Use at least two distinct X values.");
  return { x, y };
}

export function format(value) {
  if (value === null || value === undefined || !Number.isFinite(value))
    return "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 1e5)
    return value.toExponential(2);
  return Number(value.toPrecision(4)).toLocaleString("en-US", {
    maximumFractionDigits: 5,
  });
}
