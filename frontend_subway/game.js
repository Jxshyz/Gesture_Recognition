// frontend_subway/game.js
(() => {
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");

  const scoreEl = document.getElementById("score");
  const speedEl = document.getElementById("speed");
  const livesEl = document.getElementById("lives");
  const debugBox = document.getElementById("debugBox");
  const btnRestart = document.getElementById("btnRestart");

  const W = canvas.width, H = canvas.height;

  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  class Game {
    constructor() {
      this.reset();
    }

    reset() {
      this.t = 0;
      this.score = 0;
      this.speed = 1.0;      // increases
      this.lives = 1;
      this.gameOver = false;

      this.lane = 1;         // 0,1,2
      this.player = {
        x: W/2,
        y: H*0.78,
        w: 58,
        h: 86,
        vy: 0,
        onGround: true,
        duckT: 0,
      };

      this.obstacles = [];
      this.spawnT = 0;
      this.spawnEvery = 0.9; // seconds

      this.lastTs = performance.now();
      this.loop();
      this.renderHUD();
    }

    applyCommand(cmd) {
      if (!cmd) return;

      if (cmd === "RESTART") {
        this.reset();
        return;
      }

      if (this.gameOver) return;

      if (cmd === "LEFT") this.lane = clamp(this.lane - 1, 0, 2);
      else if (cmd === "RIGHT") this.lane = clamp(this.lane + 1, 0, 2);
      else if (cmd === "JUMP") this.jump();
      else if (cmd === "DUCK") this.duck();
    }

    jump() {
      if (!this.player.onGround) return;
      this.player.vy = -820;
      this.player.onGround = false;
    }

    duck() {
      // duck for a short time
      this.player.duckT = 0.55;
    }

    spawnObstacle() {
      const lane = Math.floor(Math.random() * 3);
      const type = (Math.random() < 0.5) ? "LOW" : "HIGH"; // LOW => jump, HIGH => duck
      const o = {
        lane,
        z: 0, // distance from player (we move towards player)
        y: H*0.2,
        type,
      };
      this.obstacles.push(o);
    }

    update(dt) {
      if (this.gameOver) return;

      this.t += dt;
      this.score += dt * 120 * this.speed;
      this.speed = 1.0 + (this.score / 5000);
      this.spawnEvery = Math.max(0.45, 0.9 - (this.speed - 1.0) * 0.15);

      // player lane position smoothing
      const laneX = [W*0.28, W*0.50, W*0.72][this.lane];
      this.player.x += (laneX - this.player.x) * (1 - Math.pow(0.001, dt));

      // duck timer
      if (this.player.duckT > 0) this.player.duckT -= dt;

      // gravity
      const g = 2400;
      this.player.vy += g * dt;
      this.player.y += this.player.vy * dt;

      const groundY = H*0.78;
      if (this.player.y >= groundY) {
        this.player.y = groundY;
        this.player.vy = 0;
        this.player.onGround = true;
      }

      // spawn
      this.spawnT += dt;
      if (this.spawnT >= this.spawnEvery) {
        this.spawnT = 0;
        this.spawnObstacle();
      }

      // move obstacles towards player
      const speedPx = 520 * this.speed;
      for (const o of this.obstacles) {
        o.y += speedPx * dt;
      }
      // remove passed
      this.obstacles = this.obstacles.filter(o => o.y < H + 120);

      // collision check
      this.checkCollision();

      this.renderHUD();
    }

    checkCollision() {
      const p = this.player;
      const pw = p.w;
      let ph = p.h;
      if (p.duckT > 0 && p.onGround) ph = 52; // duck reduces height

      const px = p.x - pw/2;
      const py = p.y - ph;

      for (const o of this.obstacles) {
        const ox = [W*0.28, W*0.50, W*0.72][o.lane] - 34;
        let oy = o.y;
        let ow = 68;
        let oh = 68;

        // obstacle height depends on type
        if (o.type === "HIGH") {
          oy = o.y - 40;
          oh = 58;
        }

        // lane mismatch => no collision
        const laneMatch = (o.lane === this.lane);
        if (!laneMatch) continue;

        // if LOW obstacle: must jump (player not on ground) to avoid
        if (o.type === "LOW") {
          if (!p.onGround) continue; // jumped over
        }
        // if HIGH obstacle: must duck to avoid
        if (o.type === "HIGH") {
          if (p.duckT > 0 && p.onGround) continue; // ducked under
        }

        // AABB overlap
        const hit = !(px + pw < ox || px > ox + ow || py + ph < oy || py > oy + oh);
        if (hit) {
          this.lives -= 1;
          if (this.lives <= 0) {
            this.gameOver = true;
          } else {
            // simple knockback: clear obstacles
            this.obstacles = [];
          }
          break;
        }
      }
    }

    draw() {
      ctx.clearRect(0, 0, W, H);

      // road lanes
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "#1f2937";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(W*0.39, 0); ctx.lineTo(W*0.39, H);
      ctx.moveTo(W*0.61, 0); ctx.lineTo(W*0.61, H);
      ctx.stroke();

      // obstacles
      for (const o of this.obstacles) {
        const x = [W*0.28, W*0.50, W*0.72][o.lane];
        ctx.fillStyle = (o.type === "LOW") ? "#ef4444" : "#a78bfa";
        const w = 68, h = 68;
        let y = o.y;
        if (o.type === "HIGH") y = o.y - 40;
        ctx.fillRect(x - w/2, y, w, h);
      }

      // player
      const p = this.player;
      const w = p.w;
      let h = p.h;
      if (p.duckT > 0 && p.onGround) h = 52;

      ctx.fillStyle = this.gameOver ? "#94a3b8" : "#34d399";
      ctx.fillRect(p.x - w/2, p.y - h, w, h);

      // UI overlay
      if (this.gameOver) {
        ctx.fillStyle = "rgba(2,6,23,0.75)";
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "bold 32px system-ui";
        ctx.textAlign = "center";
        ctx.fillText("GAME OVER", W/2, H*0.45);

        ctx.font = "16px system-ui";
        ctx.fillText("Sende Command RESTART oder Button drücken", W/2, H*0.52);
      }
    }

    renderHUD() {
      scoreEl.textContent = Math.floor(this.score).toString();
      speedEl.textContent = this.speed.toFixed(2) + "x";
      livesEl.textContent = String(this.lives);
    }

    loop = () => {
      const ts = performance.now();
      const dt = Math.min(0.04, (ts - this.lastTs) / 1000);
      this.lastTs = ts;

      this.update(dt);
      this.draw();

      requestAnimationFrame(this.loop);
    };
  }

  window.game = new Game();

  btnRestart.addEventListener("click", () => {
    window.game.applyCommand("RESTART");
  });

  // debug helper (optional)
  window.gameDebug = (obj) => {
    debugBox.textContent = JSON.stringify(obj, null, 2);
  };
})();
