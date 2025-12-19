async function fetchTelemetry() {
  const res = await fetch("/api/telemetry", { cache: "no-store" });
  if (!res.ok) return null;
  return await res.json();
}

function fmtPct(x) {
  const p = Math.max(0, Math.min(1, Number(x))) * 100;
  return `${p.toFixed(1)}%`;
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function renderHistory(items) {
  const ul = document.getElementById("gestureHistory");
  if (!ul) return;

  ul.innerHTML = "";
  for (const e of items.slice(0, 12)) {
    const li = document.createElement("li");
    li.className = "histItem";
    li.innerHTML = `
      <div class="histTop">
        <span class="histLabel">${e.label}</span>
        <span class="histConf">${fmtPct(e.conf)}</span>
      </div>
      <div class="histSub">
        <span class="histState">${e.state}</span>
        <span class="histTime">${fmtTime(e.t)}</span>
      </div>
    `;
    ul.appendChild(li);
  }
}

async function tick() {
  const data = await fetchTelemetry();
  if (!data) return;

  const cur = data.current || {};
  setText("uiMode", (cur.state || "idle").toUpperCase());
  setText("uiLabel", cur.label || "-");
  setText("uiConf", fmtPct(cur.conf || 0));
  setText("uiCountdown", `${(cur.seconds_left ?? 0).toFixed(2)}s`);

  renderHistory(data.history || []);
}

window.addEventListener("load", () => {
  tick();
  setInterval(tick, 150); // ~6-7 Hz, reicht völlig
});
