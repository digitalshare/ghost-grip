import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { Reflector } from "three/addons/objects/Reflector.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
// Planner world is Z-up (x right, y into the set, z up); three.js is Y-up.
const W = (v) => new THREE.Vector3(v[0], v[2], -v[1]);
const SCOPE_SENSOR_H_MM = 24.9 / 2.39; // Super 35 width, 2.39:1 extraction
const LINK_NAMES = ["base column", "upper arm", "forearm", "wrist", "wrist", "flange", "camera mount"]; // collision.py order
const AXIS_NAMES = ["rail", "base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"];

const state = {
  runId: null, scene: null, sceneText: "", rounds: [], roundPlans: {}, roundIdx: 0, plan: null, shotList: null,
  shotIdx: 0, t: 0, playing: false, view: "rig", cut: null, highlight: null, froze: new Set(), freezeUntil: 0,
  tokens: {}, elapsed: null, references: [], clips: null, compare: null,
};

/* ================= renderers & post ================= */
const bigEl = $("big"), pipEl = $("pip");
function makeRenderer(el) {
  const r = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  r.setPixelRatio(Math.min(devicePixelRatio, 2));
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.45;
  el.prepend(r.domElement);
  return r;
}
const bigR = makeRenderer(bigEl), pipR = makeRenderer(pipEl);

const GradeShader = {
  uniforms: { tDiffuse: { value: null }, time: { value: 0 }, grain: { value: 0.06 }, vignette: { value: 0.9 }, grade: { value: 1 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float time, grain, vignette, grade; varying vec2 vUv;
    float rnd(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      float l = dot(c.rgb, vec3(0.299,0.587,0.114));
      vec3 graded = mix(vec3(l), c.rgb, 0.8);                      // a little desaturation
      graded *= mix(vec3(0.86,0.98,1.08), vec3(1.06,1.0,0.92), smoothstep(0.05,0.7,l)); // teal shadows, warm highlights
      c.rgb = mix(c.rgb, graded, grade);
      vec2 d = vUv - 0.5; c.rgb *= 1.0 - vignette * dot(d, d) * 1.6;
      c.rgb += (rnd(vUv * 731.0 + fract(time) * 97.0) - 0.5) * grain;
      gl_FragColor = c;
    }`,
};
function makeComposer(renderer) {
  const comp = new EffectComposer(renderer);
  const rp = new RenderPass(new THREE.Scene(), new THREE.PerspectiveCamera());
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.55, 0.6, 0.82);
  const grade = new ShaderPass(GradeShader);
  comp.addPass(rp); comp.addPass(bloom); comp.addPass(new OutputPass()); comp.addPass(grade);
  return { comp, rp, bloom, grade };
}
const bigC = makeComposer(bigR), pipC = makeComposer(pipR);

const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x07060a);
scene3.fog = new THREE.FogExp2(0x07060a, 0.035);
bigC.rp.scene = scene3; pipC.rp.scene = scene3;

const orbitCam = new THREE.PerspectiveCamera(42, 1, 0.05, 120);
orbitCam.position.set(2.5, 5, 7);
const controls = new OrbitControls(orbitCam, bigR.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
const shotCam = new THREE.PerspectiveCamera(30, 2.39, 0.05, 80);

scene3.add(new THREE.HemisphereLight(0x9aa0c8, 0x2a1c1c, 1.05));
const moon = new THREE.DirectionalLight(0xa9c0ff, 1.7);
moon.castShadow = true;
moon.shadow.mapSize.set(2048, 2048);
moon.shadow.bias = -0.0004;
scene3.add(moon, moon.target);
// soft warm key from the open fourth wall so faces read; the practicals and the grade keep it moody
const key = new THREE.DirectionalLight(0xffd9b0, 0.55);
scene3.add(key, key.target);

const setGroup = new THREE.Group(), castGroup = new THREE.Group(), rigGroup = new THREE.Group(), fxGroup = new THREE.Group();
scene3.add(setGroup, castGroup, rigGroup, fxGroup);

/* ================= procedural textures ================= */
function canvasTex(w, h, draw, repeat = [1, 1]) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat);
  t.anisotropy = 8;
  return t;
}
const noise = (g, w, h, a) => { for (let i = 0; i < w * h / 30; i++) { g.fillStyle = `rgba(0,0,0,${Math.random() * a})`; g.fillRect(Math.random() * w, Math.random() * h, 2, 2); } };
const woodTex = () => canvasTex(512, 512, (g, w, h) => {
  for (let i = 0; i < 8; i++) {
    const shade = 52 + Math.random() * 22;
    g.fillStyle = `rgb(${shade + 18},${shade}, ${shade - 16})`; g.fillRect(0, i * h / 8, w, h / 8);
    g.strokeStyle = "rgba(0,0,0,.55)"; g.lineWidth = 2; g.beginPath(); g.moveTo(0, i * h / 8); g.lineTo(w, i * h / 8); g.stroke();
    const off = Math.random() * w; g.beginPath(); g.moveTo(off, i * h / 8); g.lineTo(off, (i + 1) * h / 8); g.stroke();
    for (let k = 0; k < 14; k++) { g.strokeStyle = `rgba(0,0,0,${0.05 + Math.random() * 0.08})`; g.lineWidth = 1; g.beginPath(); const y = i * h / 8 + Math.random() * h / 8; g.moveTo(0, y); g.bezierCurveTo(w / 3, y + 3, w * 2 / 3, y - 3, w, y); g.stroke(); }
  }
  noise(g, w, h, 0.25);
}, [3, 3]);
const wallpaperTex = () => canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = "#3b4a3f"; g.fillRect(0, 0, w, h);
  for (let x = 0; x < w; x += 32) { g.fillStyle = "rgba(255,255,255,.035)"; g.fillRect(x, 0, 12, h); }
  g.fillStyle = "rgba(210,190,140,.13)";
  for (let y = 16; y < h; y += 64) for (let x = 16; x < w; x += 64) { g.beginPath(); g.ellipse(x, y, 7, 13, 0, 0, Math.PI * 2); g.fill(); g.beginPath(); g.ellipse(x + 32, y + 32, 7, 13, 0, 0, Math.PI * 2); g.fill(); }
  const grd = g.createLinearGradient(0, 0, 0, h); grd.addColorStop(0, "rgba(0,0,0,.15)"); grd.addColorStop(1, "rgba(60,40,10,.25)"); g.fillStyle = grd; g.fillRect(0, 0, w, h);
  noise(g, w, h, 0.18);
}, [4, 2]);
const TEX = { wood: woodTex(), paper: wallpaperTex() };

const M = {
  floor: new THREE.MeshStandardMaterial({ map: TEX.wood, roughness: 0.75 }),
  wainscot: new THREE.MeshStandardMaterial({ color: 0x2a1d15, roughness: 0.7 }),
  darkwood: new THREE.MeshStandardMaterial({ color: 0x3a2516, roughness: 0.6 }),
  wood: new THREE.MeshStandardMaterial({ color: 0x5a3b22, roughness: 0.65 }),
  porcelain: new THREE.MeshStandardMaterial({ color: 0xe6e3dc, roughness: 0.25 }),
  linen: new THREE.MeshStandardMaterial({ color: 0xc9c3b6, roughness: 0.95 }),
  blanket: new THREE.MeshStandardMaterial({ color: 0x5b2430, roughness: 0.95 }),
  void: new THREE.MeshBasicMaterial({ color: 0x000000 }),
  rug: new THREE.MeshStandardMaterial({ color: 0x4a1f22, roughness: 1 }),
  curtain: new THREE.MeshStandardMaterial({ color: 0xd9d4c4, roughness: 0.9, transparent: true, opacity: 0.82, side: THREE.DoubleSide }),
  metal: new THREE.MeshStandardMaterial({ color: 0x8c8f96, metalness: 0.8, roughness: 0.35 }),
  mannequin: new THREE.MeshStandardMaterial({ color: 0xc8c0b4, roughness: 0.55, metalness: 0.05 }),
  mannequinJoint: new THREE.MeshStandardMaterial({ color: 0x34343c, roughness: 0.5, metalness: 0.2 }),
  ghost: new THREE.MeshStandardMaterial({ color: 0xdff7ff, emissive: 0x9fe8ff, emissiveIntensity: 0.55, transparent: true, opacity: 0.5, roughness: 0.4, depthWrite: false }),
};
const shadowy = (m) => { m.castShadow = true; m.receiveShadow = true; return m; };
const box = (w, h, d, mat, x, y, z) => { const m = shadowy(new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat)); m.position.set(x, y, z); return m; };

function label(text, color = "#ece8f4") {
  const c = document.createElement("canvas"); c.width = 256; c.height = 48;
  const g = c.getContext("2d");
  g.font = "600 26px Inter, sans-serif"; g.textAlign = "center";
  g.fillStyle = "rgba(0,0,0,.55)"; const w = Math.min(g.measureText(text).width + 22, 250); g.beginPath(); g.roundRect(128 - w / 2, 6, w, 36, 18); g.fill();
  g.fillStyle = color; g.fillText(text, 128, 33, 230);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
  s.scale.set(0.95, 0.18, 1); s.renderOrder = 10; s.userData.label = true;
  return s;
}
const clear = (g) => { for (const c of [...g.children]) g.remove(c); };

/* ================= the set ================= */
const practicals = [];
function buildSet(sc) {
  clear(setGroup); practicals.length = 0;
  const { width: w, depth: d, height: h } = sc.set;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(w + 6, d + 6), M.floor);
  floor.rotation.x = -Math.PI / 2; floor.position.set(w / 2, 0, -d / 2 + 1.5); floor.receiveShadow = true;
  M.floor.map.repeat.set((w + 6) / 1.6, (d + 6) / 1.6);
  setGroup.add(floor);
  const wall = (len, pos, rotY) => {
    const g = new THREE.Group(); g.position.copy(pos); g.rotation.y = rotY;
    const map = TEX.paper.clone(); map.repeat.set(len / 1.2, h / 1.2); map.needsUpdate = true;
    const paper = new THREE.Mesh(new THREE.PlaneGeometry(len, h), new THREE.MeshStandardMaterial({ map, roughness: 0.92 }));
    paper.position.y = h / 2; paper.receiveShadow = true; g.add(paper);
    const wain = new THREE.Mesh(new THREE.BoxGeometry(len, 0.95, 0.04), M.wainscot); wain.position.set(0, 0.475, 0.02); wain.receiveShadow = true; g.add(wain);
    g.add(box(len, 0.05, 0.06, M.darkwood, 0, 0.97, 0.03));
    g.add(box(len, 0.12, 0.08, M.darkwood, 0, h - 0.06, 0.04));
    setGroup.add(g);
  };
  wall(w, new THREE.Vector3(w / 2, 0, -d), 0);
  wall(d, new THREE.Vector3(0, 0, -d / 2), Math.PI / 2);
  wall(d, new THREE.Vector3(w, 0, -d / 2), -Math.PI / 2);
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshStandardMaterial({ color: 0x15121a, roughness: 1 }));
  ceiling.rotation.x = Math.PI / 2; ceiling.position.set(w / 2, h, -d / 2); setGroup.add(ceiling); // faces down: invisible from the orbit view above

  for (const p of sc.props) buildProp(p, sc);

  // moonlight through the open fourth wall, from high camera-left
  moon.position.copy(W([-2, -3, h + 3])); moon.target.position.copy(W([w / 2, d / 2, 0]));
  key.position.copy(W([w * 0.75, -4, 2.2])); key.target.position.copy(W([w / 2, d * 0.6, 1.2]));
  const ext = Math.max(w, d) * 0.8;
  Object.assign(moon.shadow.camera, { left: -ext, right: ext, top: ext, bottom: -ext, near: 0.5, far: 30 }); moon.shadow.camera.updateProjectionMatrix();
  controls.target.copy(W([w / 2, d * 0.55, 0.9]));
  orbitCam.position.copy(W([w * 0.15 - 1.2, -Math.max(w, d) * 0.85, Math.max(3.2, h + 1.6)]));
  $("empty").classList.add("hidden");
}

function practical(pos, color = 0xffb46b, intensity = 3, dist = 5) {
  const l = new THREE.PointLight(color, intensity, dist, 1.6);
  l.position.copy(pos); l.userData.base = intensity; practicals.push(l); setGroup.add(l);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffe2b8 }));
  bulb.position.copy(pos); setGroup.add(bulb);
}

function buildProp(p, sc) {
  const n = p.name.toLowerCase();
  const c = W([p.x, p.y, 0]);
  const g = new THREE.Group(); g.position.copy(c); setGroup.add(g);
  const onBackWall = p.y > sc.set.depth - 0.6;
  const wallZ = onBackWall ? -(sc.set.depth - p.y) : 0; // local z of the back wall surface
  if (/flashlight|torch/.test(n)) return; // carried by an actor; rendered as their spotlight
  if (/mirror/.test(n)) {
    const mw = Math.max(0.5, Math.min(p.width, 1.2)), mh = Math.max(0.6, Math.min(p.height, 1.0));
    const y0 = 1.5;
    const mirror = new Reflector(new THREE.PlaneGeometry(mw, mh), { textureWidth: 512, textureHeight: 512, color: 0x8a9196, clipBias: 0.003 });
    mirror.position.set(0, y0, wallZ + 0.07); g.add(mirror);
    g.add(box(mw + 0.08, mh + 0.08, 0.04, M.darkwood, 0, y0, wallZ + 0.045));
  } else if (/door/.test(n) && !/ward|closet/.test(n)) {
    const dw = Math.min(Math.max(p.width, 0.8), 1.1), dh = Math.min(Math.max(p.height, 2.0), 2.3);
    const z = wallZ + 0.03;
    g.add(box(dw, dh, 0.02, M.void, 0, dh / 2, z - 0.005));
    g.add(box(dw - 0.06, dh - 0.04, 0.045, M.darkwood, 0, dh / 2, z + 0.02));
    for (const [px, py, pw, ph] of [[-dw / 2, dh / 2, 0.08, dh], [dw / 2, dh / 2, 0.08, dh], [0, dh + 0.04, dw + 0.16, 0.08]]) g.add(box(pw, ph, 0.08, M.wood, px, py, z + 0.03));
    const knob = new THREE.Mesh(new THREE.SphereGeometry(0.03), M.metal); knob.position.set(dw / 2 - 0.12, 1.0, z + 0.06); g.add(knob);
  } else if (/ward|closet|armoire|cabinet/.test(n)) {
    const bw = p.width, bh = Math.max(p.height, 1.8), bd = Math.max(p.depth, 0.45);
    g.add(box(bw, bh, bd, M.darkwood, 0, bh / 2, 0));
    g.add(box(bw + 0.06, 0.08, bd + 0.06, M.wood, 0, bh + 0.04, 0));
    g.add(box(bw * 0.92, bh * 0.9, 0.02, M.void, 0, bh / 2, bd / 2 + 0.002)); // dark interior showing through the ajar door
    const door = new THREE.Group(); door.position.set(bw / 2 - 0.02, 0, bd / 2 + 0.02); door.rotation.y = 0.45; g.add(door);
    door.add(box(bw / 2 - 0.03, bh * 0.9, 0.035, M.wood, -(bw / 4), bh / 2, 0));
    g.add(box(bw / 2 - 0.03, bh * 0.9, 0.035, M.wood, -bw / 4, bh / 2, bd / 2 + 0.02));
  } else if (/bed/.test(n)) {
    const along = p.depth >= p.width;
    const L = Math.max(p.width, p.depth), Wd = Math.min(p.width, p.depth);
    const bg = new THREE.Group(); bg.rotation.y = along ? 0 : Math.PI / 2; g.add(bg);
    bg.add(box(Wd, 0.32, L, M.darkwood, 0, 0.16, 0));
    bg.add(box(Wd - 0.06, 0.18, L - 0.08, M.linen, 0, 0.41, 0));
    bg.add(box(Wd - 0.02, 0.06, L * 0.62, M.blanket, 0, 0.52, L * 0.18));
    bg.add(box(Wd * 0.7, 0.1, 0.32, M.linen, 0, 0.55, -L / 2 + 0.25));
    bg.add(box(Wd, 0.95, 0.06, M.darkwood, 0, 0.47, -L / 2));
  } else if (/sink|basin/.test(n)) {
    g.add(box(0.18, 0.8, 0.18, M.porcelain, 0, 0.4, 0));
    g.add(box(Math.max(p.width, 0.55), 0.16, Math.max(Math.min(p.depth, 0.5), 0.4), M.porcelain, 0, 0.86, 0));
    g.add(box(0.03, 0.18, 0.03, M.metal, 0, 1.0, -0.12));
  } else if (/tub|bath(?!.*mat)/.test(n)) {
    g.add(box(p.width, 0.58, p.depth, M.porcelain, 0, 0.29, 0));
    g.add(box(p.width - 0.12, 0.02, p.depth - 0.12, new THREE.MeshStandardMaterial({ color: 0x1b2a2e, roughness: 0.1, metalness: 0.2 }), 0, 0.5, 0));
  } else if (/curtain/.test(n)) {
    const cw = Math.max(p.width, p.depth), ch = Math.max(p.height, 1.8);
    const cur = shadowy(new THREE.Mesh(new THREE.PlaneGeometry(cw, ch, 24, 1), M.curtain));
    const pos = cur.geometry.attributes.position; for (let i = 0; i < pos.count; i++) pos.setZ(i, Math.sin(pos.getX(i) * 18) * 0.03); cur.geometry.computeVertexNormals();
    cur.position.y = ch / 2 + 0.2; if (p.depth > p.width) cur.rotation.y = Math.PI / 2; g.add(cur);
  } else if (/rug|carpet|mat\b|runner/.test(n)) {
    const r = new THREE.Mesh(new THREE.PlaneGeometry(p.width, p.depth), M.rug); r.rotation.x = -Math.PI / 2; r.position.y = 0.006; r.receiveShadow = true; g.add(r);
  } else if (/lamp|sconce|light|candle|lantern|bulb/.test(n)) {
    const wallMounted = /sconce|wall/.test(n);
    const y = wallMounted ? 1.75 : Math.max(p.height, 0.5);
    if (!wallMounted) g.add(box(0.06, y - 0.15, 0.06, M.metal, 0, (y - 0.15) / 2, 0));
    const shade = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.18, 20, 1, true), new THREE.MeshStandardMaterial({ color: 0xe8c99a, emissive: 0x8a5a2a, emissiveIntensity: 0.6, side: THREE.DoubleSide }));
    shade.position.set(0, y, wallMounted ? wallZ + 0.15 : 0); g.add(shade);
    practical(new THREE.Vector3(c.x, y - 0.05, c.z + (wallMounted ? wallZ + 0.15 : 0)), 0xffb46b, 2.6, 5);
  } else if (/window/.test(n)) {
    const win = new THREE.Mesh(new THREE.PlaneGeometry(p.width, Math.max(p.height, 1)), new THREE.MeshBasicMaterial({ color: 0x31456e }));
    win.position.set(0, 1.5, wallZ + 0.02); g.add(win);
  } else {
    g.add(box(p.width, p.height, p.depth, /table|night|desk|dresser|shelf|chest|chair/.test(n) ? M.wood : M.darkwood, 0, p.height / 2, 0));
  }
  if (!/rug|carpet|mat\b|runner/.test(n)) { const l = label(p.name, "#d8b98a"); l.position.copy(c).setY(Math.min(Math.max(p.height, 0.4) + 0.25, 2.5)); setGroup.add(l); }
}

/* ================= the cast ================= */
let botTemplate = null, botClips = [];
const botReady = new Promise((resolve) => new GLTFLoader().load("/assets/Xbot.glb", (g) => { botTemplate = g.scene; botClips = g.animations; resolve(); }, undefined, () => resolve()));
const cast = [];

function actorAt(a, t) { // mirrors schemas.Actor.position
  const marks = [[0, a.x, a.y], ...(a.path || []).filter((m) => m.t > 0).map((m) => [m.t, m.x, m.y]).sort((p, q) => p[0] - q[0])];
  if (t <= marks[0][0] || marks.length === 1) return [marks[0][1], marks[0][2]];
  for (let i = 0; i < marks.length - 1; i++) {
    const [t0, x0, y0] = marks[i], [t1, x1, y1] = marks[i + 1];
    if (t <= t1) { const u = (t - t0) / Math.max(t1 - t0, 1e-6); return [x0 + (x1 - x0) * u, y0 + (y1 - y0) * u]; }
  }
  return [marks.at(-1)[1], marks.at(-1)[2]];
}
const isEntity = (a) => a.kind === "entity" || /ghost|figure|entity|spirit|pale|twin|apparition|phantom/i.test(a.name);

async function buildCast(sc) {
  clear(castGroup); cast.length = 0;
  $("loading").classList.remove("hidden");
  await botReady;
  $("loading").classList.add("hidden");
  const text = (state.sceneText + " " + sc.props.map((p) => p.name).join(" ")).toLowerCase();
  let flashlightGiven = !/flashlight|torch/.test(text);
  for (const a of sc.actors) {
    const entity = isEntity(a);
    const holder = new THREE.Group(); castGroup.add(holder);
    let mixer = null; const actions = {};
    if (botTemplate) {
      const model = SkeletonUtils.clone(botTemplate);
      model.scale.setScalar(Math.max(a.height, 0.9) / 1.8);
      model.traverse((o) => {
        if (!o.isMesh) return;
        o.castShadow = !entity; o.receiveShadow = true; o.frustumCulled = false;
        if (entity) o.material = M.ghost;
        else o.material = /joint/i.test(o.material.name) ? M.mannequinJoint : M.mannequin;
      });
      holder.add(model);
      mixer = new THREE.AnimationMixer(model);
      for (const clip of botClips) if (["idle", "walk"].includes(clip.name)) actions[clip.name] = mixer.clipAction(clip);
      actions.idle?.play();
      if (actions.walk) { actions.walk.play(); actions.walk.setEffectiveWeight(0); }
      if (entity && actions.idle) actions.idle.timeScale = 0.25;
    } else { // asset missing: a plain silhouette, never a hard failure
      const model = shadowy(new THREE.Mesh(new THREE.CapsuleGeometry(0.2, Math.max(a.height - 0.4, 0.3), 6, 16), entity ? M.ghost : new THREE.MeshStandardMaterial({ color: 0xb8b4c4 })));
      model.position.y = a.height / 2; holder.add(model);
    }
    if (entity) { const glow = new THREE.PointLight(0x9fe8ff, 1.2, 2.5, 2); glow.position.y = a.height * 0.8; holder.add(glow); }
    if (!entity && !flashlightGiven) {
      const spot = new THREE.SpotLight(0xfff1d6, 18, 9, 0.32, 0.55, 1.4); spot.castShadow = true; spot.shadow.mapSize.set(1024, 1024);
      spot.position.set(0.18, a.height * 0.62, 0.1); spot.target.position.set(0.1, a.height * 0.35, 3); holder.add(spot, spot.target); flashlightGiven = true;
    }
    const tag = label(a.name, entity ? "#bff3ff" : "#ffffff"); tag.position.y = a.height + 0.28; holder.add(tag);
    cast.push({ a, holder, mixer, actions, entity, heading: 0, restHeading: 0, walkW: 0 });
  }
  faceFocus(sc);
}
function faceFocus(sc) { // idle actors face the beat's focus (a prop or another actor); entities face the camera side
  for (const c of cast) {
    const focus = sc.beats.map((b) => b.focus).find((f) => f && f !== c.a.name);
    const target = sc.props.find((p) => p.name === focus) || sc.actors.find((p) => p.name === focus);
    c.restHeading = !c.entity && target ? Math.atan2(target.x - c.a.x, -(target.y - c.a.y)) : 0;
    c.heading = c.restHeading;
  }
}

function poseCast(sceneT, dt) {
  for (const c of cast) {
    const [x, y] = actorAt(c.a, sceneT);
    const [x2, y2] = actorAt(c.a, sceneT + 0.15);
    const vx = x2 - x, vy = y2 - y, speed = Math.hypot(vx, vy) / 0.15;
    c.holder.position.copy(W([x, y, 0]));
    if (c.entity) c.holder.position.y = 0.04 + Math.sin(performance.now() / 700) * 0.03;
    const walking = speed > 0.08 && !c.entity;
    const want = walking ? Math.atan2(vx, -vy) : c.restHeading;
    let dh = want - c.heading; dh = Math.atan2(Math.sin(dh), Math.cos(dh)); c.heading += dh * Math.min(1, dt * 6 + (dt === 0 ? 0 : 0));
    c.holder.rotation.y = c.heading;
    if (c.mixer) {
      const target = walking ? THREE.MathUtils.clamp(speed / 1.1, 0.3, 1) : 0;
      c.walkW = THREE.MathUtils.lerp(c.walkW, target, Math.min(1, dt * 5));
      c.actions.walk?.setEffectiveWeight(c.walkW); c.actions.idle?.setEffectiveWeight(1 - c.walkW);
      if (c.actions.walk) c.actions.walk.timeScale = THREE.MathUtils.clamp(speed / 1.2, 0.5, 1.6);
      c.mixer.update(dt);
    }
  }
  M.ghost.opacity = 0.42 + Math.sin(performance.now() / 260) * 0.06 + (Math.random() < 0.015 ? -0.3 : 0);
}

/* ================= the robot ================= */
const R = {
  body: new THREE.MeshStandardMaterial({ color: 0xd6dade, metalness: 0.35, roughness: 0.42 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x2a2e35, metalness: 0.5, roughness: 0.45 }),
  accent: new THREE.MeshStandardMaterial({ color: 0xff7a1a, metalness: 0.2, roughness: 0.4, emissive: 0x401800 }),
  black: new THREE.MeshStandardMaterial({ color: 0x0c0c0e, metalness: 0.4, roughness: 0.3 }),
  glass: new THREE.MeshStandardMaterial({ color: 0x223355, metalness: 0.9, roughness: 0.05 }),
  hit: new THREE.MeshStandardMaterial({ color: 0xff2d55, emissive: 0xff2d55, emissiveIntensity: 1.6 }),
};
const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 28);
const taperCyl = new THREE.CylinderGeometry(0.85, 1, 1, 28);
const rig = { built: false };
function buildRig(rail) {
  clear(rigGroup);
  const railG = new THREE.Group(); railG.position.copy(W([rail.rail_x_start + 2, rail.rail_y, 0])); rigGroup.add(railG);
  railG.add(box(4.3, 0.05, 0.5, R.dark, 0, 0.025, 0));
  for (const z of [-0.16, 0.16]) railG.add(box(4.3, 0.04, 0.05, R.body, 0, 0.07, z));
  for (let x = -2; x <= 2; x += 0.5) railG.add(box(0.06, 0.03, 0.56, R.black, x, 0.015, 0));
  rig.carriage = box(0.55, 0.1, 0.55, R.dark, 0, 0.13, 0); rigGroup.add(rig.carriage);
  rig.pedestal = shadowy(new THREE.Mesh(unitCyl, R.dark)); rigGroup.add(rig.pedestal);
  rig.links = LINK_NAMES.map((_, i) => { const m = shadowy(new THREE.Mesh(taperCyl, i === 6 ? R.dark : R.body)); m.userData.base = m.material; rigGroup.add(m); return m; });
  rig.housings = Array.from({ length: 6 }, () => {
    const g = new THREE.Group();
    const h = shadowy(new THREE.Mesh(unitCyl, R.dark)); h.scale.set(0.085, 0.19, 0.085); g.add(h);
    for (const y of [-0.065, 0.065]) { const ring = new THREE.Mesh(new THREE.TorusGeometry(0.087, 0.009, 8, 32), R.accent); ring.rotation.x = Math.PI / 2; ring.position.y = y; g.add(ring); }
    g.userData.mesh = h; rigGroup.add(g); return g;
  });
  rig.cam = new THREE.Group();
  rig.cam.add(box(0.15, 0.14, 0.26, R.black, 0, 0.01, -0.1));
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.055, 0.16, 24), R.black); lens.rotation.x = Math.PI / 2; lens.position.z = 0.1; rig.cam.add(lens);
  const glass = new THREE.Mesh(new THREE.CircleGeometry(0.045, 24), R.glass); glass.position.z = 0.181; rig.cam.add(glass);
  rig.cam.add(box(0.22, 0.004, 0.12, R.black, 0, 0.09, 0.2)); // matte box flags
  rig.cam.add(box(0.004, 0.14, 0.12, R.black, 0.11, 0.02, 0.2)); rig.cam.add(box(0.004, 0.14, 0.12, R.black, -0.11, 0.02, 0.2));
  const tally = new THREE.Mesh(new THREE.SphereGeometry(0.014), new THREE.MeshBasicMaterial({ color: 0xff3355 })); tally.position.set(0.06, 0.09, -0.2); rig.cam.add(tally);
  rig.cam.add(box(0.06, 0.05, 0.06, R.dark, 0, -0.08, -0.1));
  rig.cam.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  rigGroup.add(rig.cam);
  if (!rig.path) { rig.path = new THREE.Group(); fxGroup.add(rig.path); }
  rig.built = true;
}
const Y = new THREE.Vector3(0, 1, 0);
function placeCyl(mesh, a, b, r0) {
  const dir = new THREE.Vector3().subVectors(b, a), len = dir.length();
  mesh.visible = len > 1e-3; if (!mesh.visible) return;
  mesh.position.copy(a).addScaledVector(dir, 0.5); mesh.scale.set(r0, len, r0); mesh.quaternion.setFromUnitVectors(Y, dir.normalize());
}
function poseRig(m, t) {
  const f = THREE.MathUtils.clamp(t / m.dt, 0, m.links.length - 1), i = Math.floor(f), j = Math.min(i + 1, m.links.length - 1), u = f - i;
  const lerp = (A, B) => A.map((v, k) => v + (B[k] - v) * u);
  const pts = m.links[i].map((p, k) => W(lerp(p, m.links[j][k])));
  rig.carriage.position.copy(pts[0]).setY(0.13);
  placeCyl(rig.pedestal, pts[0].clone().setY(0.18), pts[1], 0.11);
  const radii = [0.075, 0.068, 0.058, 0.05, 0.05, 0.045, 0.04];
  for (let k = 0; k < 7; k++) placeCyl(rig.links[k], pts[k + 1], pts[k + 2], radii[k]);
  rig.housings.forEach((h, k) => {
    h.position.copy(pts[k + 1]); h.scale.setScalar(1 - k * 0.07);
    if (m.axes?.length) h.quaternion.setFromUnitVectors(Y, W(lerp(m.axes[i][k], m.axes[j][k])).normalize());
  });
  const pos = W(lerp(m.cam_pos[i], m.cam_pos[j])), fwd = W(lerp(m.cam_fwd[i], m.cam_fwd[j])), up = W(lerp(m.cam_up[i], m.cam_up[j]));
  rig.cam.position.copy(pos); rig.cam.up.copy(up); rig.cam.lookAt(pos.clone().add(fwd));
  const lensMm = m.lens_mm[i] + (m.lens_mm[j] - m.lens_mm[i]) * u;
  shotCam.position.copy(pos); shotCam.up.copy(up); shotCam.lookAt(pos.clone().add(fwd));
  shotCam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(SCOPE_SENSOR_H_MM / 2 / lensMm)); shotCam.updateProjectionMatrix();
  return lensMm;
}
function setRigTint(bad) { R.body.emissive.setHex(bad ? 0x3a0008 : 0x000000); }
function highlight(v) {
  if (!rig.built) return;
  for (const m of rig.links) m.material = m.userData.base;
  rig.housings.forEach((h) => (h.userData.mesh.material = R.dark));
  if (!v) return;
  LINK_NAMES.forEach((n, i) => { if (n === v.link) rig.links[i].material = R.hit; });
  if (v.link === "camera") rig.links[6].material = R.hit;
  const ax = AXIS_NAMES.indexOf(v.link); if (ax >= 1) rig.housings[ax - 1].userData.mesh.material = R.hit;
}
function drawPaths() {
  if (!rig.path) return;
  clear(rig.path);
  const plan = state.plan; if (!plan) return;
  const bad = new Set(plan.violations.map((v) => v.shot_id));
  plan.shots.forEach((m, i) => {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(m.cam_pos.map(W)),
      new THREE.LineBasicMaterial({ color: bad.has(m.shot_id) ? 0xff4d6d : 0x7cffb2, transparent: true, opacity: i === state.shotIdx ? 0.95 : 0.22 }));
    rig.path.add(line);
  });
}
function clearanceRing(v, sceneT) {
  const old = fxGroup.getObjectByName("ring"); if (old) fxGroup.remove(old);
  if (!v || v.kind !== "actor") return;
  const actor = state.scene.actors.find((a) => a.name === v.obstacle); if (!actor) return;
  const need = parseFloat((v.detail.match(/needs ([\d.]+) cm/) || [])[1] || "60") / 100;
  const [x, y] = actorAt(actor, sceneT);
  const g = new THREE.Group(); g.name = "ring"; g.position.copy(W([x, y, 0]));
  for (const h of [0.05, 0.9, 1.5]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(need, 0.012, 8, 64), new THREE.MeshBasicMaterial({ color: 0xff4d6d, transparent: true, opacity: 0.85 }));
    ring.rotation.x = Math.PI / 2; ring.position.y = h; g.add(ring);
  }
  fxGroup.add(g);
}

/* ================= views, layout, loop ================= */
function layout() {
  const st = $("stage").getBoundingClientRect(), H = st.height - 62;
  let w = st.width, h = H, left = 0, top = 0;
  bigEl.classList.toggle("cine", state.view === "camera");
  if (state.compare) { // original on the left, our camera view on the right (stacked when narrow)
    const wide = st.width > 900, paneW = wide ? st.width / 2 - 18 : st.width - 24, paneH = wide ? H - 80 : (H - 90) / 2;
    const fit = (aspect) => { let fw = paneW, fh = fw / aspect; if (fh > paneH) { fh = paneH; fw = fh * aspect; } return [fw, fh]; };
    const [yw, yh] = fit(16 / 9);
    [w, h] = fit(2.39);
    const yt = $("cmp-left"); const ytLeft = wide ? 12 + (paneW - yw) / 2 : (st.width - yw) / 2, ytTop = wide ? 52 + (paneH - yh) / 2 : 48;
    Object.assign(yt.style, { left: `${ytLeft}px`, top: `${ytTop}px`, width: `${yw}px`, height: `${yh}px` });
    left = wide ? st.width / 2 + 6 + (paneW - w) / 2 : (st.width - w) / 2; top = wide ? 52 + (paneH - h) / 2 : 48 + yh + 34;
    Object.assign($("cmp-right-label").style, { left: `${left}px`, top: `${top + h + 6}px`, width: `${w}px` });
    Object.assign($("cmp-left-label").style, { left: `${ytLeft}px`, top: `${ytTop + yh + 6}px`, width: `${yw}px` });
  } else if (state.view === "camera") { h = Math.min(H, st.width / 2.39); w = h * 2.39; left = (st.width - w) / 2; top = (H - h) / 2; }
  Object.assign(bigR.domElement.style, { width: `${w}px`, height: `${h}px`, left: `${left}px`, top: `${top}px`, position: "absolute" });
  Object.assign($("hud").style, { left: `${left}px`, width: `${w}px`, top: `${top}px` }); // keep the slate inside the letterboxed frame
  $("rounds").classList.toggle("in-cam", state.view === "camera");
  bigR.setSize(w, h, false); bigC.comp.setSize(w, h);
  pipEl.classList.toggle("rig", state.view === "camera");
  const p = pipEl.getBoundingClientRect(); pipR.setSize(p.width, p.height, false); pipC.comp.setSize(p.width, p.height);
  orbitCam.aspect = state.view === "camera" ? p.width / p.height : w / h; orbitCam.updateProjectionMatrix();
}
addEventListener("resize", layout);
function setView(v) {
  state.view = v; $("view-rig").classList.toggle("active", v === "rig"); $("view-cam").classList.toggle("active", v === "camera");
  controls.disconnect(); controls.connect(v === "rig" ? bigR.domElement : pipR.domElement);
  layout();
}

function renderView(C, cam, cinematic) {
  rigGroup.visible = fxGroup.visible = !cinematic;
  scene3.traverse((o) => { if (o.userData.label) o.visible = !cinematic; });
  C.rp.camera = cam;
  const u = C.grade.uniforms;
  u.time.value = performance.now() / 1000; u.grain.value = cinematic ? 0.075 : 0.02; u.vignette.value = cinematic ? 1.1 : 0.45; u.grade.value = cinematic ? 1 : 0.3;
  C.bloom.strength = cinematic ? 0.6 : 0.3;
  C.comp.render();
  rigGroup.visible = fxGroup.visible = true;
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  const m = state.plan?.shots[state.shotIdx];
  const frozen = now < state.freezeUntil;
  if (m) {
    if (state.playing && !frozen) {
      state.t += dt;
      if (state.t > m.duration) {
        if (state.cut) { nextCutShot(); return; }
        state.t = 0; state.froze.clear(); highlight(null); clearanceRing(null); hideCaption();
      }
    }
    // rejected rounds: stop on each violation so the audience sees exactly what the simulator caught
    if (state.playing && !state.cut && !frozen) for (const v of state.plan.violations) {
      const key = `${v.shot_id}@${v.t}@${v.kind}`;
      if (v.shot_id === m.shot_id && state.t >= v.t && !state.froze.has(key)) {
        state.froze.add(key); state.t = v.t; state.freezeUntil = now + 2400; showViolation(v); break;
      }
    }
    if (!frozen && state.highlight && state.t > state.highlight.t + 0.8) { highlight(null); clearanceRing(null); hideCaption(); state.highlight = null; }
    const t = Math.min(state.t, m.duration), sceneT = (m.start_t || 0) + t;
    const lens = poseRig(m, t);
    poseCast(sceneT, frozen ? 0 : dt);
    updateHud(m, t, sceneT, lens);
    if (state.cut) updateSubtitle(sceneT);
  } else if (state.scene) poseCast(0, dt);
  for (const l of practicals) l.intensity = l.userData.base * (0.85 + Math.random() * 0.15 + (Math.random() < 0.006 ? -0.7 : 0));
  controls.update();
  if (state.view === "rig") { renderView(bigC, orbitCam, false); if (m) renderView(pipC, shotCam, true); }
  else { renderView(bigC, m ? shotCam : orbitCam, !!m); renderView(pipC, orbitCam, false); }
}
layout(); requestAnimationFrame(frame);

/* ================= HUD & captions ================= */
function updateHud(m, t, sceneT, lens) {
  const shot = state.shotList?.shots.find((s) => s.id === m.shot_id);
  const tc = (s) => `00:00:${String(Math.floor(s)).padStart(2, "0")}:${String(Math.floor((s % 1) * 24)).padStart(2, "0")}`;
  const after = shot?.reference_title ? ` · after ${esc(shot.reference_title.split(/ [—–-] /)[0])}` : "";
  const html = `<span><span class="rec">● REC</span> ${esc(m.shot_id)} · ${esc(shot?.technique || "")}${after}</span><span>${tc(sceneT)} · ${lens.toFixed(0)}mm</span>`;
  $("pip-label").innerHTML = state.view === "rig" ? html : `<span>RIG · axis load ${m.peak_joint_speed_pct.toFixed(0)}%</span><span>${esc(m.shot_id)}</span>`;
  const hud = $("hud"); hud.classList.toggle("hidden", state.view !== "camera"); if (state.view === "camera") hud.innerHTML = html;
  $("time").textContent = `${t.toFixed(2)}s`;
  $("scrub").value = Math.round((t / Math.max(m.duration, 1e-6)) * 1000);
}
function showViolation(v) {
  state.highlight = v; highlight(v);
  const m = state.plan.shots[state.shotIdx]; clearanceRing(v, (m.start_t || 0) + v.t);
  const c = $("caption"); c.className = ""; c.innerHTML = `✗ <b>${esc(v.shot_id)} @ ${v.t}s</b> · ${esc(v.detail)}`;
}
function hideCaption() { $("caption").classList.add("hidden"); }
function updateSubtitle(sceneT) {
  const b = state.scene.beats.filter((x) => x.t <= sceneT + 0.05).at(-1); const el = $("subtitle");
  if (b && sceneT - b.t < 4.5) { el.textContent = b.description; el.classList.remove("hidden"); } else el.classList.add("hidden");
  const lastBeat = state.scene.beats.at(-1);
  if (lastBeat && !state.cut.hit && sceneT >= lastBeat.t) { state.cut.hit = true; audio.hit(); }
}

/* ================= shots, rounds, transport ================= */
function selectShot(i) {
  if (!state.plan?.shots[i]) return;
  state.shotIdx = i; state.t = 0; state.froze.clear(); state.freezeUntil = 0; highlight(null); clearanceRing(null); hideCaption();
  document.querySelectorAll("#shot-tabs button").forEach((b, k) => b.classList.toggle("active", k === i));
  drawPaths();
}
function loadPlan(plan, shotList) {
  if (plan.shots.length > 1 && plan.shots.every((m) => !m.start_t)) { // runs saved before shots carried start_t: lay them end to end
    let t0 = 0; for (const m of plan.shots) { m.start_t = t0; t0 += m.duration; }
  }
  state.plan = plan; state.shotList = shotList;
  buildRig(plan.rig);
  const bad = new Set(plan.violations.map((v) => v.shot_id));
  $("shot-tabs").innerHTML = plan.shots.map((m) => `<button class="${bad.has(m.shot_id) ? "bad" : ""}" title="starts at scene t=${(m.start_t || 0).toFixed(1)}s">${esc(m.shot_id)}</button>`).join("");
  document.querySelectorAll("#shot-tabs button").forEach((b, i) => (b.onclick = () => selectShot(i)));
  setRigTint(plan.violations.length > 0);
  selectShot(Math.min(state.shotIdx, plan.shots.length - 1));
  setPlaying(true);
}
function renderRounds() {
  const el = $("rounds"); el.classList.toggle("hidden", state.rounds.length === 0);
  el.innerHTML = `<span class="label">Rig check</span>` + state.rounds.map((r, i) => {
    const n = r.violations.length, name = i === 0 ? "Draft" : `Rev ${i}`;
    return `<button data-i="${i}" class="${n ? "bad" : "good"} ${i === state.roundIdx ? "active" : ""}" title="${n ? `${n} problem(s) caught by the simulator` : "every move passed"}">${name} ${n ? `✗ ${n}` : "✓"}</button>`;
  }).join("");
  el.querySelectorAll("button").forEach((b) => (b.onclick = () => selectRound(+b.dataset.i)));
}
async function selectRound(i, opts = {}) {
  state.roundIdx = i; renderRounds();
  if (!state.roundPlans[i]) { const c = $("caption"); c.className = "good"; c.textContent = `Re-simulating ${i === 0 ? "the draft" : `revision ${i}`} on the arm…`; }
  const p = await roundPlan(i); hideCaption();
  if (!p) return;
  loadPlan(p, p.shot_list || state.rounds[i].shot_list);
  const v = opts.violation;
  if (v) { const k = p.shots.findIndex((m) => m.shot_id === v.shot_id); if (k >= 0) selectShot(k); state.t = Math.max(0, v.t - 1.2); }
  else if (p.violations.length) { const k = p.shots.findIndex((m) => m.shot_id === p.violations[0].shot_id); if (k >= 0) selectShot(k); state.t = Math.max(0, p.violations[0].t - 1.5); }
  else if (i > 0) { const c = $("caption"); c.className = "good"; c.textContent = `✓ Revision ${i}: every move is reachable, within axis limits and clear of the cast`; setTimeout(hideCaption, 3500); }
}
function setPlaying(p) { state.playing = p; $("play").textContent = p ? "❚❚" : "▶"; }
$("play").onclick = () => setPlaying(!state.playing);
$("scrub").oninput = (e) => { const m = state.plan?.shots[state.shotIdx]; if (!m) return; setPlaying(false); state.t = (e.target.value / 1000) * m.duration; state.froze.clear(); highlight(null); hideCaption(); };
$("view-rig").onclick = () => setView("rig");
$("view-cam").onclick = () => setView("camera");
$("swap").onclick = () => setView(state.view === "rig" ? "camera" : "rig");
addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
  if (e.key === "Escape" && state.compare) closeCompare();
  if (e.key === "Escape" && state.cut) stopCut();
  if (e.key === "ArrowRight") selectShot(Math.min(state.shotIdx + 1, (state.plan?.shots.length || 1) - 1));
  if (e.key === "ArrowLeft") selectShot(Math.max(state.shotIdx - 1, 0));
});

/* ================= play the cut ================= */
const audio = (() => {
  let ctx, master, nodes = [];
  return {
    start() {
      ctx = ctx || new AudioContext(); master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);
      master.gain.linearRampToValueAtTime(0.2, ctx.currentTime + 2);
      for (const [f, type] of [[55, "sawtooth"], [55.7, "sawtooth"], [82.4, "sine"]]) {
        const o = ctx.createOscillator(), lp = ctx.createBiquadFilter(); o.type = type; o.frequency.value = f; lp.type = "lowpass"; lp.frequency.value = 240;
        o.connect(lp).connect(master); o.start(); nodes.push(o);
      }
    },
    hit() {
      if (!ctx) return;
      const len = ctx.sampleRate * 1.6, buf = ctx.createBuffer(1, len, ctx.sampleRate), d = buf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
      const src = ctx.createBufferSource(), g = ctx.createGain(); src.buffer = buf; g.gain.value = 0.8; src.connect(g).connect(ctx.destination); src.start();
      const boom = ctx.createOscillator(), bg = ctx.createGain(); boom.frequency.setValueAtTime(90, ctx.currentTime); boom.frequency.exponentialRampToValueAtTime(30, ctx.currentTime + 1.2);
      bg.gain.setValueAtTime(0.8, ctx.currentTime); bg.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5); boom.connect(bg).connect(ctx.destination); boom.start(); boom.stop(ctx.currentTime + 1.6);
    },
    stop() { if (!ctx || !master) return; master.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.6); const ns = nodes; nodes = []; setTimeout(() => ns.forEach((o) => o.stop()), 700); },
  };
})();
async function playCut() {
  const final = state.rounds.length - 1;
  if (state.roundIdx !== final) await selectRound(final);
  const order = state.plan.shots.map((m, i) => [m.start_t || 0, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  state.cut = { order, k: 0, hit: false };
  document.body.classList.add("cut"); setView("camera"); hideCaption();
  selectShot(order[0]); setPlaying(true); audio.start();
  $("cut").textContent = "■ Stop (Esc)";
}
function nextCutShot() {
  state.cut.k += 1;
  if (state.cut.k >= state.cut.order.length) { state.t = state.plan.shots[state.shotIdx].duration; setPlaying(false); setTimeout(stopCut, 1800); return; }
  selectShot(state.cut.order[state.cut.k]);
}
function stopCut() {
  if (!state.cut) return;
  state.cut = null; audio.stop(); document.body.classList.remove("cut"); $("subtitle").classList.add("hidden"); setView("rig"); $("cut").textContent = "▶ Play the scene";
}
$("cut").onclick = () => (state.cut ? stopCut() : playCut());

/* ================= reference vs previz ================= */
async function openCompare(k, shotId) {
  const ref = state.references[k]; if (!ref) return;
  const clip = state.clips?.[k];
  const final = state.rounds.length - 1;
  if (final >= 0 && state.roundIdx !== final) await selectRound(final);
  if (state.cut) stopCut();
  let idx = shotId ? state.plan?.shots.findIndex((m) => m.shot_id === shotId) : -1;
  if (idx < 0 && state.shotList) idx = state.plan.shots.findIndex((m) => refIndexForShot(state.shotList.shots.find((s) => s.id === m.shot_id)) === k);
  const shot = idx >= 0 ? state.shotList.shots.find((s) => s.id === state.plan.shots[idx].shot_id) : null;
  state.compare = { k }; document.body.classList.add("comparing"); $("compare").classList.remove("hidden");
  $("cmp-title").innerHTML = `Where this move comes from: <b>${esc(ref.film)}</b> — ${esc(ref.moment)}`;
  $("cmp-left").innerHTML = clip
    ? `<iframe src="https://www.youtube-nocookie.com/embed/${encodeURIComponent(clip.video_id)}?autoplay=1&rel=0&modestbranding=1&start=${clip.start_s || 0}" title="${esc(clip.title)}" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>`
    : `<div class="cmp-empty">No embeddable clip of this moment was found on YouTube.<br><a href="${esc(ref.url)}" target="_blank" rel="noopener">Read the source article ↗</a></div>`;
  $("cmp-left-label").innerHTML = clip ? `<b>Original</b> · ${esc(clip.title)} · <span class="muted">${esc(clip.channel)}, played from YouTube</span>` : `<b>Original</b> · ${esc(ref.film)}`;
  $("cmp-right-label").innerHTML = shot ? `<b>Ghost Grip</b> · ${esc(shot.id)} ${esc(shot.technique)} · ${shot.keyframes[0].lens_mm.toFixed(0)}mm · planned for the 6-axis rig` : `<b>Ghost Grip</b> · no shot in the final plan cites this reference`;
  if (idx >= 0) selectShot(idx);
  setView("camera"); setPlaying(true);
}
function closeCompare() {
  if (!state.compare) return;
  state.compare = null; $("cmp-left").innerHTML = ""; // removing the iframe stops its audio
  $("compare").classList.add("hidden"); document.body.classList.remove("comparing"); setView("rig");
}
$("cmp-close").onclick = closeCompare;

/* ================= crew chat ================= */
const AGENTS = {
  you: ["🗣️", "You", "director"],
  ad: ["📋", "1st AD", "Parasail · schema-locked JSON"],
  scout: ["🔎", "Reference Scout", "CrewAI agent · You.com search"],
  lib: ["📚", "Technique library", "Parasail BGE-M3 embeddings"],
  dp: ["🎥", "Cinematographer", "Parasail · streamed JSON"],
  rig: ["🦾", "Rig Engineer", "physics simulator · no LLM"],
  director: ["🎬", "Director's notes", "Parasail stream"],
  producer: ["📣", "Producer", "CrewAI agent · One"],
};
function say(agent, html, { id, cls = "" } = {}) {
  let el = id && document.getElementById(id);
  if (!el) {
    el = document.createElement("div"); if (id) el.id = id;
    const [icon, name, sub] = AGENTS[agent];
    el.innerHTML = `<div class="avatar">${icon}</div><div class="bubble"><div class="who">${esc(name)} <small>${esc(sub)}</small></div><div class="body"></div></div>`;
    $("chat").appendChild(el);
  }
  el.className = `msg ${cls}`; el.querySelector(".body").innerHTML = html;
  $("panel").scrollTo({ top: $("panel").scrollHeight, behavior: "smooth" });
  return el;
}
const typing = (agent, text, id) => say(agent, `<span class="typing">${esc(text)}</span>`, { id });
function castLine(sc) {
  return sc.actors.map((a) => {
    const walk = (a.path || []).filter((m) => m.t > 0);
    return `${esc(a.name)}${isEntity(a) ? " 👻" : ""}${walk.length ? ` (walks until ${walk.at(-1).t.toFixed(0)}s)` : ""}`;
  }).join(", ");
}
function sayScene(sc) {
  say("ad", `<p><b>${esc(sc.title)}</b>: ${esc(sc.logline)}</p><p>${sc.beats.length} beats over ${sc.duration_s.toFixed(0)}s · cast: ${castLine(sc)} · ${sc.set.width}×${sc.set.depth} m set</p>`, { id: "m-ad" });
}
function sayRefs(refs) {
  state.references = refs;
  const card = (r, k) => {
    const clip = state.clips ? state.clips[k] : undefined; // undefined = still searching, null = none found
    const thumb = clip ? `<button class="thumb" data-k="${k}" title="Play the original next to our version" style="background-image:url('${esc(clip.thumbnail)}')"><span>▶</span></button>`
      : `<div class="thumb ${clip === null ? "none" : "loading"}">${clip === null ? "no embeddable clip" : "finding clip…"}</div>`;
    const src = clip ? `<span class="src">▶ ${esc(clip.channel)} · verified on YouTube</span>` : "";
    return `<div class="ref">${thumb}<div class="ref-text"><b>${esc(r.film)}</b> — ${esc(r.moment)}<span>${esc(r.technique)}</span>${src}
      <a href="${esc(r.url)}" target="_blank" rel="noopener">source article ↗</a></div></div>`;
  };
  const el = say("scout", `<p>Found ${refs.length} real reference shots:</p><div class="refs">${refs.map(card).join("")}</div>`, { id: "m-scout" });
  el.querySelectorAll("button.thumb").forEach((b) => (b.onclick = () => openCompare(+b.dataset.k)));
}
function refIndexForShot(shot) {
  if (!shot) return -1;
  const byUrl = state.references.findIndex((r) => r.url === shot.reference_url);
  if (byUrl >= 0) return byUrl;
  const t = (shot.reference_title || "").toLowerCase();
  return state.references.findIndex((r) => t.includes(r.film.replace(/\s*\(\d{4}\)/, "").toLowerCase()));
}
function sayTechs(ts) {
  say("lib", `<p>Closest camera techniques for these beats:</p><div class="chips">${ts.map((t) => `<span class="chip" title="${esc(t.effect || "")}"><b>${(t.score ?? 0).toFixed(2)}</b> ${esc(t.name)}</span>`).join("")}</div>`, { id: "m-lib" });
}
function sayShotList(r, sl) {
  const rows = sl.shots.map((s) => {
    const k = refIndexForShot(s);
    const ref = k >= 0 ? `<button class="refbtn" data-k="${k}" data-shot="${esc(s.id)}" title="Watch the original this move is based on">▶ ref</button>` : "";
    return `<li data-shot="${esc(s.id)}" data-round="${r}"><b>${esc(s.id)}</b> ${esc(s.technique)} · ${[...new Set(s.keyframes.map((k) => k.lens_mm.toFixed(0)))].join("/")}mm · ${s.keyframes.at(-1).t.toFixed(1)}s @ ${(s.start_t || 0).toFixed(0)}s ${ref}</li>`;
  }).join("");
  const el = say("dp", `<p>${r === 0 ? "Draft" : `Revision ${r}`}: ${esc(sl.director_notes)}</p><ul class="shots">${rows}</ul>`, { id: `m-dp-${r}` });
  el.querySelectorAll("button.refbtn").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); openCompare(+b.dataset.k, b.dataset.shot); }));
  el.querySelectorAll("li").forEach((li) => (li.onclick = async () => { await selectRound(+li.dataset.round); const k = state.plan.shots.findIndex((m) => m.shot_id === li.dataset.shot); if (k >= 0) selectShot(k); }));
}
function sayRig(r, plan) {
  const gaps = (plan.shots || []).map((m) => m.min_actor_gap_m).filter((g) => g != null);
  const gap = gaps.length ? ` · closest approach to the cast ${(Math.min(...gaps) * 100).toFixed(0)} cm` : "";
  if (!plan.violations.length) return say("rig", `<p>✓ Every move is reachable, within axis limits and clear of the cast${gap}.</p>`, { id: `m-rig-${r}`, cls: "good" });
  const items = plan.violations.map((v, k) => `<div class="viol"><span class="what">✗ <b>${esc(v.shot_id)} @ ${v.t}s</b> ${esc(v.detail)}</span><span class="fix">→ ${esc(v.suggestion)}</span><button data-k="${k}">▶ show me</button></div>`).join("");
  const el = say("rig", `<p>Rejected: ${plan.violations.length} problem${plan.violations.length > 1 ? "s" : ""} the real rig would hit.</p>${items}`, { id: `m-rig-${r}`, cls: "bad" });
  el.querySelectorAll("button[data-k]").forEach((b) => (b.onclick = () => { if (state.view !== "rig") setView("rig"); selectRound(r, { violation: plan.violations[+b.dataset.k] }); }));
}
async function roundPlan(i) {
  if (!state.roundPlans[i] && state.runId) {
    const r = await fetch(`/api/runs/${state.runId}/rounds/${i}`);
    if (r.ok) state.roundPlans[i] = await r.json();
  }
  return state.roundPlans[i];
}
async function renderStats() {
  if (!state.rounds.length) return;
  const finalRound = state.rounds.at(-1);
  const caught = state.rounds.reduce((n, r) => n + r.violations.length, 0);
  const finalPlan = await roundPlan(state.rounds.length - 1); // stats describe the approved plan, never a rejected draft
  if (!finalPlan) return;
  const gaps = finalPlan.shots.map((m) => m.min_actor_gap_m).filter((g) => g != null);
  const peak = Math.max(...finalPlan.shots.map((m) => m.peak_joint_speed_pct));
  const ok = !finalRound.violations.length;
  $("stats").innerHTML = [
    `<span class="${ok ? "good" : "bad"}"><b>${ok ? "✓ safe to shoot" : "✗ open issues"}</b></span>`,
    `<span><b>${state.rounds.length - 1}</b>revision${state.rounds.length === 2 ? "" : "s"}</span>`,
    `<span class="${caught ? "bad" : ""}"><b>${caught}</b>problem${caught === 1 ? "" : "s"} caught before set</span>`,
    gaps.length ? `<span><b>${(Math.min(...gaps) * 100).toFixed(0)} cm</b>closest approach to the cast</span>` : "",
    `<span><b>${peak.toFixed(0)}%</b>peak axis load</span>`,
    state.elapsed ? `<span><b>${state.elapsed.toFixed(0)} s</b>to plan</span>` : "",
  ].join("");
  $("stats").classList.remove("hidden"); document.body.classList.add("has-stats"); layout();
}

/* ================= live run ================= */
function resetAll() {
  $("chat").innerHTML = ""; for (const id of ["dispatch-card", "error", "sheet", "replay-badge", "stats", "rounds"]) $(id).classList.add("hidden");
  document.body.classList.remove("has-stats");
  closeCompare();
  Object.assign(state, { runId: null, rounds: [], roundPlans: {}, roundIdx: 0, plan: null, shotList: null, shotIdx: 0, t: 0, tokens: {}, elapsed: null, references: [], clips: null });
  $("shot-tabs").innerHTML = ""; $("cut").disabled = true; hideCaption(); layout();
}
const RUNNING = {
  scene: () => ["ad", "Breaking the scene down into a set, cast and beats", "m-ad"],
  research: () => ["scout", "Searching the web for real reference shots", "m-scout"],
  retrieve: () => ["lib", "Matching beats to camera techniques", "m-lib"],
  compose: (ev) => ["dp", ev.round ? "Revising the moves after the rig engineer's notes" : "Designing the shot list", `m-dp-${ev.round || 0}`],
  robot: (ev) => ["rig", "Simulating every move on the 6-axis arm", `m-rig-${ev.round || 0}`],
  notes: () => ["director", "Briefing the crew", "m-notes"],
  produce: () => ["producer", "Delivering the call sheet through One", "m-producer"],
};
function handle(ev) {
  switch (ev.type) {
    case "step": if (ev.status === "running" && RUNNING[ev.step]) typing(...RUNNING[ev.step](ev)); break;
    case "token": {
      state.tokens[ev.channel] = (state.tokens[ev.channel] || "") + ev.text;
      if (ev.channel === "notes") say("director", `<p>${esc(state.tokens.notes)}</p>`, { id: "m-notes" });
      else if (ev.channel.startsWith("compose")) {
        const r = ev.channel.split("-")[1]; const el = document.querySelector(`#m-dp-${r} .typing`);
        if (el) el.textContent = `Writing ${r === "0" ? "the shot list" : "the revision"} · ${state.tokens[ev.channel].length} chars`;
      }
      break;
    }
    case "scene": state.scene = ev; buildSet(ev); buildCast(ev); sayScene(ev); break;
    case "references": state.clips = null; sayRefs(ev.references); break;
    case "clips": state.clips = ev.clips; sayRefs(state.references); break;
    case "techniques": sayTechs(ev.techniques); break;
    case "shot_list": state.pendingShotList = ev; sayShotList(ev.round, ev); break;
    case "robot_plan":
      state.rounds[ev.round] = { violations: ev.violations, shot_list: state.pendingShotList };
      state.roundPlans[ev.round] = { ...ev, shot_list: state.pendingShotList };
      sayRig(ev.round, ev); renderRounds(); selectRound(ev.round); break;
    case "done":
      state.runId = ev.run_id; state.elapsed = ev.elapsed_s; history.replaceState(null, "", `?run=${ev.run_id}`);
      $("dispatch-card").classList.remove("hidden"); $("export").href = `/api/runs/${ev.run_id}/program.csv`; $("cut").disabled = false;
      renderStats(); refreshRuns(); break;
    case "one_result": say("producer", `<p>${esc(ev.report).replace(/\n/g, "<br>")}</p>`, { id: "m-producer", cls: "good" }); break;
    case "dispatched": $("send").disabled = false; break;
    case "error": $("error").textContent = ev.message; $("error").classList.remove("hidden"); $("roll").disabled = false; $("send").disabled = false; break;
  }
}
function subscribe(channel, onEnd) {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${channel}`);
  ws.onmessage = (m) => handle(JSON.parse(m.data));
  ws.onclose = () => onEnd && onEnd();
}
$("roll").onclick = async () => {
  const text = $("scene-text").value.trim();
  resetAll(); state.sceneText = text; history.replaceState(null, "", location.pathname);
  say("you", `<p>${esc(text)}</p>`);
  $("roll").disabled = true; $("roll").textContent = "● Rolling…";
  try {
    const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene_text: text }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    subscribe((await r.json()).channel, () => { $("roll").disabled = false; $("roll").textContent = "● Roll camera"; });
  } catch (e) { handle({ type: "error", message: String(e.message || e) }); $("roll").textContent = "● Roll camera"; }
};

/* ================= replay ================= */
async function replay(id) {
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}`);
  if (!r.ok) { handle({ type: "error", message: "Saved run not found." }); return; }
  const s = await r.json();
  resetAll(); $("replay-badge").classList.remove("hidden");
  state.runId = s.id; state.sceneText = s.scene_text; state.scene = s.scene; state.elapsed = s.elapsed_s || null;
  $("scene-text").value = s.scene_text; $("scene-pick").value = "";
  say("you", `<p>${esc(s.scene_text)}</p>`); sayScene(s.scene); sayRefs(s.references); sayTechs(s.techniques);
  buildSet(s.scene); await buildCast(s.scene);
  state.rounds = s.rounds.map((rd) => ({ violations: rd.violations, shot_list: rd.shot_list }));
  const lastRound = s.rounds.length - 1;
  s.rounds.forEach((rd, i) => { sayShotList(rd.round, rd.shot_list); sayRig(rd.round, i === lastRound ? s.plan : { violations: rd.violations, shots: [] }); });
  say("director", `<p>${esc(s.notes)}</p>`);
  fetch(`/api/runs/${s.id}/clips`).then((r) => (r.ok ? r.json() : null)).then((c) => { if (c && state.runId === s.id) { state.clips = c; sayRefs(state.references); } });
  renderRounds();
  const firstBad = s.rounds.findIndex((rd) => rd.violations.length);
  await selectRound(firstBad >= 0 ? firstBad : lastRound);
  $("dispatch-card").classList.remove("hidden"); $("export").href = `/api/runs/${s.id}/program.csv`; $("cut").disabled = false;
  renderStats();
}

/* ================= dispatch via One ================= */
function shootStart() { return $("when").value ? new Date($("when").value).toISOString() : new Date().toISOString(); }
$("preview-sheet").onclick = async () => {
  if (!state.runId) return;
  const r = await fetch(`/api/runs/${state.runId}/call-sheet?shoot_start=${encodeURIComponent(shootStart())}`);
  $("sheet").textContent = (await r.json()).text; $("sheet").classList.toggle("hidden");
};
$("send").onclick = async () => {
  if (!state.runId) return;
  $("send").disabled = true;
  const r = await fetch(`/api/runs/${state.runId}/dispatch`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ to: $("to").value.trim(), slack_channel: $("slack").value.trim(), shoot_start: shootStart() }) });
  if (!r.ok) { $("send").disabled = false; handle({ type: "error", message: (await r.json()).detail }); return; }
  subscribe((await r.json()).channel, () => ($("send").disabled = false));
};

/* ================= boot ================= */
async function refreshRuns() {
  const runs = await (await fetch("/api/runs")).json();
  $("run-pick").innerHTML = `<option value="">Replay a saved run…</option>` + runs.map((r) => `<option value="${esc(r.id)}">${esc(r.title)} · ${r.feasible ? "✓" : "✗"} · ${r.revisions} rev</option>`).join("");
}
(async () => {
  const health = await (await fetch("/api/health")).json();
  $("status").innerHTML = [["You.com", health.youcom], ["Parasail", health.parasail], ["One", health.one]]
    .map(([n, ok]) => `<span class="pill ${ok ? "ok" : "off"}">${n} ${ok ? "●" : "○"}</span>`).join("") + `<span class="pill">${esc(health.model)}</span>`;
  if (health.call_sheet_to) $("to").value = health.call_sheet_to;
  if (health.slack_channel) $("slack").value = health.slack_channel;
  const tomorrow = new Date(Date.now() + 86400000); tomorrow.setHours(9, 0, 0, 0);
  $("when").value = new Date(tomorrow - tomorrow.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  const scenes = await (await fetch("/api/scenes")).json();
  $("scene-pick").innerHTML = `<option value="">Example scenes…</option>` + scenes.map((s, i) => `<option value="${i}">${esc(s.name.replace(/^\d-/, "").replace(/-/g, " "))}</option>`).join("");
  $("scene-pick").onchange = (e) => { if (e.target.value !== "") $("scene-text").value = scenes[e.target.value].text; };
  if (scenes[0]) { $("scene-text").value = scenes[0].text; $("scene-pick").value = "0"; }
  $("run-pick").onchange = (e) => e.target.value && (history.replaceState(null, "", `?run=${e.target.value}`), replay(e.target.value));
  await refreshRuns();
  const q = new URLSearchParams(location.search).get("run");
  if (q) replay(q);
})();
