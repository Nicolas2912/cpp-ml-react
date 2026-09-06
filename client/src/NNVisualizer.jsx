import { useMemo, useState } from "react";
import ReactFlow, { Controls, Background, Position } from "reactflow";
import "reactflow/dist/style.css";

export function parseArchitecture(value) {
  if (!/^1(?:-[1-9]\d?){0,4}-1$/.test(value)) return null;
  const layers = value.split("-").map(Number);
  return layers.some((size) => size > 32) ? null : layers;
}

export function generateFlowElements(layers, selected, isDark) {
  const nodes = [],
    edges = [];
  layers.forEach((size, layer) => {
    for (let index = 0; index < size; index++) {
      const id = `l${layer}-n${index}`;
      nodes.push({
        id,
        data: {
          label:
            layer === 0
              ? "Input"
              : layer === layers.length - 1
                ? "Output"
                : `L${layer} · Neuron ${index + 1}`,
          layer,
          index,
        },
        position: { x: layer * 180, y: (index - (size - 1) / 2) * 65 },
        type:
          layer === 0
            ? "input"
            : layer === layers.length - 1
              ? "output"
              : "default",
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        selected: selected === id,
        style: {
          width: 130,
          borderRadius: 12,
          border: `2px solid ${selected === id ? "#a78bfa" : "#818cf8"}`,
          color: isDark ? "#e2e8f0" : "#334155",
          background: isDark ? "#1e293b" : "#eef2ff",
        },
      });
      if (layer)
        for (let previous = 0; previous < layers[layer - 1]; previous++)
          edges.push({
            id: `${layer}-${previous}-${index}`,
            source: `l${layer - 1}-n${previous}`,
            target: id,
            style: { stroke: isDark ? "#64748b" : "#a5b4fc" },
          });
    }
  });
  return { nodes, edges };
}

export default function NNVisualizer({
  layerStructure,
  onChange,
  disabled = false,
  isDark = false,
  height = 420,
}) {
  const [selection, setSelection] = useState(null);
  const layers = useMemo(
    () => parseArchitecture(layerStructure),
    [layerStructure],
  );
  const selected =
    selection?.architecture === layerStructure ? selection : null;
  const { nodes, edges } = useMemo(
    () =>
      layers
        ? generateFlowElements(layers, selected?.id, isDark)
        : { nodes: [], edges: [] },
    [layers, selected, isDark],
  );
  const update = (values) => {
    setSelection(null);
    onChange?.(values.join("-"));
  };
  if (!layers)
    return (
      <p role="status" className="p-4 text-sm">
        Enter a valid architecture: one input, one output, and up to four hidden
        layers of 1–32 neurons (for example 1-4-1).
      </p>
    );
  return (
    <div className="space-y-4" aria-label="Neural network architecture builder">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="rounded-full border border-indigo-400/40 px-3 py-1">
          {layers.reduce((a, b) => a + b, 0)} neurons · {edges.length}{" "}
          connections
        </span>
        <span className="font-mono">{layerStructure}</span>
        <button
          type="button"
          className="blueprint-button"
          disabled={disabled || layers.length >= 6}
          onClick={() => update([...layers.slice(0, -1), 4, 1])}
        >
          Add hidden layer
        </button>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {layers.slice(1, -1).map((size, i) => (
          <div key={i} className="flex flex-wrap items-center gap-2 text-sm">
            <label htmlFor={`hidden-${i}`}>Hidden layer {i + 1} neurons</label>
            <select
              id={`hidden-${i}`}
              className="blueprint-select"
              value={size}
              disabled={disabled}
              onChange={(event) => {
                const next = [...layers];
                next[i + 1] = Number(event.target.value);
                update(next);
              }}
            >
              {Array.from({ length: 32 }, (_, j) => (
                <option key={j} value={j + 1}>
                  {j + 1}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="blueprint-button"
              aria-label={`Remove hidden layer ${i + 1}`}
              disabled={disabled}
              onClick={() =>
                update(layers.filter((_, index) => index !== i + 1))
              }
            >
              Remove
            </button>
          </div>
        ))}
      </div>
      <p className="text-xs opacity-80" role="status">
        {selected
          ? `${selected.label} selected. ${selected.layer === 0 || selected.layer === layers.length - 1 ? "Input and output stay at one neuron for X-to-Y regression." : `Edit hidden layer ${selected.layer} above to change its neurons.`}`
          : "Select a neuron to inspect its layer. Pan, zoom, or use fit view to explore the network."}
      </p>
      <div
        style={{ height }}
        className="overflow-hidden rounded-2xl border border-indigo-400/30"
      >
        <ReactFlow
          key={layerStructure}
          nodes={nodes}
          edges={edges}
          nodesDraggable={false}
          nodesConnectable={false}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          onNodeClick={(_, node) =>
            setSelection({
              architecture: layerStructure,
              id: node.id,
              layer: node.data.layer,
              label: node.data.label,
            })
          }
        >
          <Controls showInteractive={false} />
          <Background
            variant="dots"
            gap={16}
            size={1}
            color={isDark ? "#475569" : "#cbd5e1"}
          />
        </ReactFlow>
      </div>
    </div>
  );
}
