// frontend_subway/ws_controls.js
(() => {
  const wsStateEl = document.getElementById("wsState");
  const fsmStateEl = document.getElementById("fsmState");
  const liveLabelEl = document.getElementById("liveLabel");
  const liveConfEl = document.getElementById("liveConf");
  const lastCmdEl = document.getElementById("lastCmd");
  const armFill = document.getElementById("armFill");

  function fmtPct01(x) {
    const p = Math.max(0, Math.min(1, Number(x))) * 100;
    return p.toFixed(1) + "%";
  }

  function connect() {
    const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => { wsStateEl.textContent = "✅ verbunden"; };
    ws.onclose = () => { wsStateEl.textContent = "⚠️ getrennt – retry…"; setTimeout(connect, 800); };
    ws.onerror = () => { wsStateEl.textContent = "❌ Fehler"; };

    // keep alive ping (server wartet auf receive_text)
    const ping = setInterval(() => {
      try { ws.send("ping"); } catch {}
    }, 1000);

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);

        if (msg.type === "state") {
          fsmStateEl.textContent = msg.state || "IDLE";
          const prog = Math.max(0, Math.min(1, Number(msg.progress ?? 0)));
          armFill.style.width = (prog * 100) + "%";

          if (msg.info) {
            liveLabelEl.textContent = msg.info.label ?? "-";
            liveConfEl.textContent = fmtPct01(msg.info.conf ?? 0);
          }
          if (window.gameDebug) window.gameDebug(msg);
        }

        if (msg.type === "command") {
          lastCmdEl.textContent = msg.cmd || "-";
          if (window.game && typeof window.game.applyCommand === "function") {
            window.game.applyCommand(msg.cmd);
          }
          if (window.gameDebug) window.gameDebug(msg);
        }
      } catch (e) {}
    };

    ws.addEventListener("close", () => clearInterval(ping));
  }

  connect();
})();
