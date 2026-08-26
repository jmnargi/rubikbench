import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const RUN = window.RUN;

// ---- cube construction ------------------------------------------------------
const COLORS = { U: 0xf5f5f5, R: 0xc0392b, F: 0x27ae60, D: 0xf7d51e, L: 0xe67e22, B: 0x2c5f9e };
// Face frames (normal, right, down) mirror rubikbench/cube.py exactly.
const FACES = [
  { n: [0, 1, 0], r: [1, 0, 0], d: [0, 1, 0] },   // U
  { n: [1, 0, 0], r: [0, -1, 0], d: [0, 0, -1] },  // R
  { n: [0, 0, 1], r: [1, 0, 0], d: [0, 0, -1] },   // F
  { n: [0, -1, 0], r: [1, 0, 0], d: [0, -1, 0] },  // D
  { n: [-1, 0, 0], r: [0, 1, 0], d: [0, 0, -1] },  // L
  { n: [0, 0, -1], r: [-1, 0, 0], d: [0, 0, -1] }, // B
];
const CELL = 2 / 3;
const FACE = 1;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(3.6, 3.0, 5.2);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
document.getElementById('threed').appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 1.05));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(3, 6, 4);
scene.add(key);

// Rubik's cube body: dark rounded-ish core box + white-edge frame look.
const body = new THREE.Mesh(
  new THREE.BoxGeometry(2.28, 2.28, 2.28),
  new THREE.MeshLambertMaterial({ color: 0x0b0e12, roughness: 0.9 }),
);
scene.add(body);

const stickers = []; // flat [face][row][col] -> Mesh
FACES.forEach((f) => {
  const n = new THREE.Vector3(...f.n);
  const right = new THREE.Vector3(...f.r);
  const down = new THREE.Vector3(...f.d);
  const basis = new THREE.Matrix4().makeBasis(right, down, n);
  const quat = new THREE.Quaternion().setFromRotationMatrix(basis);
  const plate = [];
  for (let r = 0; r < 3; r++) {
    const row = [];
    for (let c = 0; c < 3; c++) {
      const pos = n.clone().multiplyScalar(FACE)
        .addScaledVector(right, (c - 1) * CELL)
        .addScaledVector(down, (r - 1) * CELL);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.52, 0.52, 0.09),
        new THREE.MeshStandardMaterial({ color: 0x222, roughness: 0.55, metalness: 0.05 }),
      );
      mesh.position.copy(pos);
      mesh.quaternion.copy(quat);
      mesh.translateOnAxis(n, 0.05); // lift stickers off the body
      scene.add(mesh);
      row.push(mesh);
    }
    plate.push(row);
  }
  stickers.push(plate);
});

// ---- state ------------------------------------------------------------------
let solves = RUN && Array.isArray(RUN.solves) ? RUN.solves : [];
let cur = 0;        // selected solve index
let pi = 0;         // playback index into timeline
let playing = false;
let playhead = 0;   // seconds since solve start
let lastTs = 0;
const speedEl = document.getElementById('speed');
const slider = document.getElementById('slider');
const el = (id) => document.getElementById(id);

function timeline() { return solves[cur]?.timeline || []; }
function setStickers(facelets) {
  const s = String(facelets || '');
  for (let f = 0; f < 6; f++) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const ch = s[f * 9 + r * 3 + c] || '?';
        stickers[f][r][c].material.color.setHex(COLORS[ch] ?? 0x333333);
      }
    }
  }
}
function fmtMoves(m) { return (m || []).join(' '); }

function renderInfo() {
  const s = solves[cur] || {};
  const t = timeline()[pi] || {};
  el('step-info').innerHTML =
    `<div>step ${pi}/${timeline().length - 1} &middot; turn ${t.turn ?? '-'}` +
    ` &middot; ${t.action ?? ''}</div>` +
    `<div class="moves">${t.moves && t.moves.length ? fmtMoves(t.moves) : '&nbsp;'}</div>` +
    `<div>${t.solved ? '<span class="ok">solved</span>' : 'state updated'}</div>`;
  el('clock').textContent = `t=${(t.t ?? 0).toFixed(1)}s`;
}

function renderList() {
  const box = el('list');
  box.innerHTML = '';
  timeline().forEach((t, i) => {
    const row = document.createElement('div');
    row.className = i === pi ? 'active' : '';
    row.innerHTML =
      `<span style="color:#8fa3b8">#${i}</span> ` +
      `<span style="color:#8fa3b8">${(t.t ?? 0).toFixed(1)}s</span> ` +
      `<b style="color:#a9c7e8">${t.action || ''}</b> ` +
      (t.moves && t.moves.length ? fmtMoves(t.moves).slice(0, 60) : '');
    row.onclick = () => { playing = false; el('btn-play').textContent = 'Play'; goTo(i); };
    box.appendChild(row);
  });
  el('list-count').textContent = `${timeline().length} steps`;
}

function renderChips() {
  const s = solves[cur] || {};
  const chips = [
    ['input', s.prompt_tokens ?? 0], ['output', s.completion_tokens ?? 0],
    ['cached', s.cached_tokens ?? 0], ['tokens', s.total_tokens ?? 0],
    ['turns', s.turns ?? 0], ['tools', s.tool_calls ?? 0],
    ['moves', s.total_moves ?? 0], ['par', s.par ?? 0],
    ['score', s.score ?? 0], ['time', `${(s.elapsed ?? 0).toFixed(1)}s`],
    ['retries', s.retries ?? 0],
  ];
  el('chips').innerHTML = chips
    .map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join('');
  const warns = [];
  if (s.truncated) warns.push('output truncated (finish_reason=length)');
  if (s.error) warns.push(`error: ${s.error}`);
  if ((s.finish_reasons || []).some((r) => r === 'length')) warns.push('at least one turn hit the output cap');
  el('warnings').style.display = warns.length ? '' : 'none';
  el('warnings').innerHTML = '<h3>Warnings</h3>' + warns.map((w) => `<div class="warn">${w}</div>`).join('');
}

function goTo(i) {
  pi = Math.max(0, Math.min(timeline().length - 1, i));
  const t = timeline()[pi] || {};
  playhead = t.t ?? 0;
  slider.value = String(pi);
  setStickers(t.facelets);
  renderInfo();
  renderList();
}

function selectSolve() {
  const s = solves[cur] || {};
  el('file').textContent = RUN?.runs_detail?.[0]?.file || '';
  el('model').textContent = s.model || RUN?.models?.[0] || '';
  el('hint').textContent = s.solved ? 'SOLVED' : s.error ? 'FAILED' : '';
  el('hint').style.color = s.solved ? '#7fd4a2' : '#ff7b7b';
  renderChips();
  goTo(0);
}

// solve selector
const sel = el('solve-select');
sel.innerHTML = solves.map((s, i) =>
  `<option value="${i}">#${s.index ?? i} ${s.solved ? 'solved' : 'unsolved'} ` +
  `moves=${s.total_moves ?? '-'} score=${s.score ?? '-'} tokens=${s.total_tokens ?? '-'}</option>`
).join('');
sel.onchange = () => { cur = Number(sel.value); playing = false; el('btn-play').textContent = 'Play'; selectSolve(); };

// controls
const btnPlay = el('btn-play');
btnPlay.onclick = () => {
  playing = !playing;
  btnPlay.textContent = playing ? 'Pause' : 'Play';
  lastTs = performance.now();
  if (playing && pi >= timeline().length - 1) goTo(0);
};
el('btn-prev').onclick = () => { playing = false; btnPlay.textContent = 'Play'; goTo(pi - 1); };
el('btn-next').onclick = () => { playing = false; btnPlay.textContent = 'Play'; goTo(pi + 1); };
slider.oninput = () => { playing = false; btnPlay.textContent = 'Play'; goTo(Number(slider.value)); };
slider.max = String(Math.max(0, timeline().length - 1));

// autoplay: advance playhead seconds scaled by speed, snap to timeline entries
const tick = () => {
  if (playing) {
    const now = performance.now();
    const dt = (now - lastTs) / 1000;
    lastTs = now;
    playhead += dt * Number(speedEl.value);
    const tl = timeline();
    let target = pi;
    for (let i = pi; i < tl.length && (tl[i].t ?? 0) <= playhead; i++) target = i;
    if (target !== pi) {
      if (target >= tl.length - 1 && (tl[tl.length - 1].t ?? 0) <= playhead) {
        // reached the end
        if (el('loop').checked) {
          goTo(0); // restart from the beginning
        } else {
          playing = false;
          btnPlay.textContent = 'Play';
          goTo(target); // park on the final state
        }
      } else {
        goTo(target);
      }
    }
  }
  requestAnimationFrame(tick);
};

function resize() {
  const host = el('threed');
  const w = host.clientWidth, h = host.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(el('scene'));
resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
requestAnimationFrame(tick);

selectSolve();
