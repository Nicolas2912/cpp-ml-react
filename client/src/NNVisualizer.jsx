import React from "react";
import ReactFlow, { Controls, Background, Position } from "reactflow";

import "reactflow/dist/style.css";

// Helper function to parse layer string and generate nodes/edges
const generateFlowElements = (layerString) => {
  if (!layerString || !/^(\d+-)+\d+$/.test(layerString)) {
    return { nodes: [], edges: [] }; // Return empty if format is invalid
  }

  const layerSizes = layerString.split('-').map(Number);
  const nodes = [];
  const edges = [];
  let xOffset = 0;
  const ySpacing = 80; // Vertical space between nodes in a layer
  const xSpacing = 150; // Horizontal space between layers

  const layerNodeIds = []; // Store node IDs for each layer to create edges

  layerSizes.forEach((size, layerIndex) => {
    const currentLayerIds = [];
    // Calculate vertical offset to center the layer
    const totalLayerHeight = (size - 1) * ySpacing;
    const yOffsetStart = -totalLayerHeight / 2;

    for (let i = 0; i < size; i++) {
      const nodeId = `l${layerIndex}-n${i}`;
      const yPos = yOffsetStart + i * ySpacing;
      const isInput = layerIndex === 0;
      const isOutput = layerIndex === layerSizes.length - 1;
      let label = `Node ${i}`;
      let nodeType = 'default'; // Default intermediate node

      if (isInput) {
        label = `Input ${i}`;
        nodeType = 'input';
      } else if (isOutput) {
        label = `Output ${i}`;
        nodeType = 'output';
      }

      nodes.push({
        id: nodeId,
        data: { label },
        position: { x: xOffset, y: yPos },
        type: nodeType,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: { width: 100, textAlign: 'center' }, // Basic styling
      });
      currentLayerIds.push(nodeId);
    }

    layerNodeIds.push(currentLayerIds);
    xOffset += xSpacing;

    // Create edges from previous layer to current layer
    if (layerIndex > 0) {
      const prevLayerIds = layerNodeIds[layerIndex - 1];
      prevLayerIds.forEach(sourceNodeId => {
        currentLayerIds.forEach(targetNodeId => {
          edges.push({
            id: `e-${sourceNodeId}-${targetNodeId}`,
            source: sourceNodeId,
            target: targetNodeId,
            // type: 'smoothstep', // Optional: change edge type
            animated: false, // Make edges non-animated by default
          });
        });
      });
    }
  });

  return { nodes, edges };
};

function NNVisualizer({ layerStructure, height = 360 }) {
  const { nodes, edges } = generateFlowElements(layerStructure);

  // Basic error handling or placeholder if structure is invalid/empty
  if (nodes.length === 0) {
    return (
      <div
        style={{
          height: typeof height === "number" ? `${height}px` : height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "grey",
          width: "100%",
        }}
      >
        <p>Enter a valid layer structure (e.g., 1-4-1) to visualize.</p>
      </div>
    );
  }

  return (
    // Set explicit height for the container
    <div
      style={{
        height: typeof height === "number" ? `${height}px` : height,
        width: "100%",
        border: "1px solid #ccc",
        borderRadius: "8px",
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodesDraggable={false} // Disable dragging for simplicity
        nodesConnectable={false} // Disable connecting
        fitView // Zoom/pan to fit all nodes
        fitViewOptions={{ padding: 0.2 }} // Add some padding on fitView
      >
        <Controls showInteractive={false} />
        {/* <MiniMap nodeStrokeWidth={3} zoomable pannable /> */}
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export default NNVisualizer; 
