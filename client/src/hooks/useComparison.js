import { useEffect, useRef, useState } from "react";

const empty = { id: null, status: "idle", loss: [], result: null, error: null };

export function mergeUpdate(current, event) {
  if (event.id !== current.id) return current;
  if (
    ["completed", "failed", "cancelled"].includes(current.status) &&
    event.status === "running"
  )
    return current;
  if (event.type === "progress") {
    if (current.status !== "running") return current;
    const loss = [
      ...current.loss.filter((point) => point.epoch !== event.epoch),
      { epoch: event.epoch, mse: event.mse },
    ];
    return { ...current, loss: loss.sort((a, b) => a.epoch - b.epoch) };
  }
  const loss = event.loss
    ? [
        ...new Map(
          [...event.loss, ...current.loss].map((point) => [point.epoch, point]),
        ).values(),
      ].sort((a, b) => a.epoch - b.epoch)
    : current.loss;
  return { ...current, ...event, loss, error: event.error || null };
}

export default function useComparison() {
  const [run, setRun] = useState(empty);
  const [connected, setConnected] = useState(false);
  const token = useRef(null);
  const active = useRef(null);
  const version = useRef(0);

  useEffect(() => {
    let stopped = false,
      socket,
      retry;
    const connect = () => {
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`,
      );
      socket.onmessage = ({ data }) => {
        if (stopped) return;
        const event = JSON.parse(data);
        if (event.type === "session") {
          token.current = event.token;
          setConnected(true);
          return;
        }
        if (event.id === active.current)
          setRun((current) => mergeUpdate(current, event));
      };
      socket.onclose = () => {
        if (stopped) return;
        token.current = null;
        active.current = null;
        version.current++;
        setConnected(false);
        setRun({
          ...empty,
          error: "Connection lost. Reconnecting… Train again once connected.",
        });
        retry = setTimeout(connect, 1500);
      };
      socket.onerror = () => socket.close();
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      socket.close();
      token.current = null;
    };
  }, []);

  const request = async (path, body) => {
    if (!token.current)
      throw new Error(
        "Waiting for the training server. Check that it is running.",
      );
    const response = await fetch(`/api${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token.current}`,
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: AbortSignal.timeout(35000),
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.error || "Request failed. Try again.");
    return data;
  };

  const start = async (data) => {
    const revision = ++version.current;
    active.current = null;
    setRun({ ...empty, status: "running" });
    try {
      const { id } = await request("/jobs", data);
      if (revision !== version.current) return;
      active.current = id;
      setRun({ ...empty, id, status: "running" });
      // Recover events that arrived before the HTTP acknowledgement.
      const snapshot = await request(`/jobs/${id}`);
      if (revision === version.current)
        setRun((current) => mergeUpdate(current, snapshot));
    } catch (error) {
      if (revision === version.current)
        setRun((current) => ({
          ...current,
          status: "failed",
          error: error.message,
        }));
    }
  };
  const cancel = async () => {
    const id = active.current;
    if (!id) return;
    try {
      const snapshot = await request(`/jobs/${id}/cancel`, {});
      if (id === active.current)
        setRun((current) => mergeUpdate(current, snapshot));
    } catch (error) {
      setRun((current) => ({ ...current, error: error.message }));
    }
  };
  const reset = () => {
    active.current = null;
    version.current++;
    setRun(empty);
  };
  const predict = async (x, id = active.current, model) => {
    const revision = version.current;
    const prediction = await request(`/jobs/${id}/predict`, {
      x,
      ...(model ? { model } : {}),
    });
    if (revision !== version.current)
      throw new Error("The dataset changed. Train again before predicting.");
    return prediction;
  };
  return { run, connected, start, cancel, reset, predict };
}
