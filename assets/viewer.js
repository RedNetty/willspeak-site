// The WillSpeak case in 3D. One model, drawn in one of two places on the page:
//
//   'hero'   at the top of the page. The case turns slowly and leans toward the
//            pointer. Holding it lifts the parts apart; letting go closes it.
//   'bench'  in the device section. Drag to turn it. The parts come together as
//            the build steps scroll past, or follow the slider.
//
// The geometry is model.json, the assembled model exported from the case's CAD
// source (printed parts, the Raspberry Pi 5, the SSD board, the fan and the
// power bank). Nothing here changes a part's shape: parts only move whole,
// straight up, and the lid key slides out of the case wall.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

// What each part rests on, and how far it lifts off it at full spread, in mm.
// A part's height is its own lift plus the lift of everything under it.
const STACK = {
  tray: { on: null, lift: 0 },
  bank: { on: 'tray', lift: 26 },
  plate: { on: 'bank', lift: 20 },
  case: { on: 'plate', lift: 26 },
  key: { on: 'case', lift: 0, slide: 36 },
  pi: { on: 'case', lift: 60 },
  spacers: { on: 'case', lift: 60 },
  hat: { on: 'pi', lift: 31 },
  fan: { on: 'hat', lift: 18 },
  lid: { on: 'fan', lift: 31 },
};
const ORDER = ['tray', 'bank', 'plate', 'case', 'key', 'pi', 'spacers', 'hat', 'fan', 'lid'];   // each part after what it rests on

// The parts that settle during each build step on the page, in the order the steps are written.
export const BUILD_STEPS = [['bank'], ['pi', 'spacers', 'hat'], ['fan', 'lid', 'key'], ['plate', 'case']];

// The parts as the page lists them, top to bottom.
export const PARTS = [
  { id: 'lid', name: 'Lid', kind: 'printed', swatch: '#2F5BEA' },
  { id: 'fan', name: 'Fan', kind: 'bought', swatch: '#34373E' },
  { id: 'hat', name: 'NVMe SSD board', kind: 'bought', swatch: '#2F6B4F' },
  { id: 'spacers', name: 'Spacers (4)', kind: 'printed', swatch: '#2B2E35' },
  { id: 'pi', name: 'Raspberry Pi 5, 8 GB', kind: 'bought', swatch: '#2E8B57' },
  { id: 'case', name: 'Case body', kind: 'printed', swatch: '#2B2E35' },
  { id: 'key', name: 'Lid key', kind: 'printed', swatch: '#2B2E35' },
  { id: 'plate', name: 'Rail plate', kind: 'printed', swatch: '#2B2E35' },
  { id: 'bank', name: '65 W power bank', kind: 'bought', swatch: '#4A4F5C' },
  { id: 'tray', name: 'Battery tray', kind: 'printed', swatch: '#2B2E35' },
];

// The three ways the bench draws the model.
export const VIEWS = ['solid', 'xray', 'lines'];

// Printed parts wear the colours of the real print: black, with a blue lid.
const PRINT_BLACK = 0x2B2E35, PRINT_BLUE = 0x2F5BEA;
const COLOR = {
  pcb: 0x2E8B57, hatpcb: 0x3A3F4B, ssd: 0x2F6B4F, label: 0xD5D9E0,
  silver: 0xC4C8CE, gold: 0xD4AF4A, brass: 0xC29A45, black: 0x34373E, beige: 0xE4D6B5,
  usb3: 0x2F5BEA, ribbon: 0xD08A3C, bank: 0x4A4F5C, display: 0x8FD3F4,
  ledg: 0x37C866, leda: 0xF2A93B, ledb: 0x2F5BEA,
};
const METAL = new Set(['silver', 'gold', 'brass']);
const PRINTED = new Set(['print1', 'mech']);
const EDGE_ON_DARK = 0x8C95A6, EDGE_ON_LIGHT = 0x1B1E24;

const HOME = new THREE.Vector3(0.80, 1.0, 0.52).normalize();   // the side that shows the ports and reads the lid's name
const HOME_TURN = Math.atan2(HOME.y, HOME.x), HOME_RISE = Math.asin(HOME.z);
const KEY_LIGHT = new THREE.Vector3(0.35, 0.9, 1.25).multiplyScalar(800);
const FILL_LIGHT = new THREE.Vector3(-0.9, 0.2, 0.5).multiplyScalar(800);
const UP = new THREE.Vector3(0, 0, 1);

const HERO_SPREAD = 0.82;      // how far the parts lift while the hero is held
const HERO_CLOSE = 0.86;       // the closed case stands this much nearer than a loose fit; it backs off as the parts lift
const HERO_AIM_DROP = 0.13;    // closed, the camera aims this share of the model's height under its middle, which centres it on screen
const HERO_TURN_RATE = 0.22;   // radians a second
const HERO_LEAN = { turn: 0.5, rise: 0.2 };   // radians at the edge of the screen
const SCROLL_SPREAD = 0.8;     // how far the parts start apart before the build steps
const SETTLE = 0.0004;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const smooth = (t) => { const u = clamp(t, 0, 1); return u * u * (3 - 2 * u); };

function bytes(b64) {
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u.buffer;
}

// A mesh is stored as 16-bit positions with a scale and an offset, plus indices.
function decode(m) {
  const q = new Int16Array(bytes(m.p)), pos = new Float32Array(q.length);
  for (let i = 0; i < q.length; i += 3) {
    pos[i] = q[i] * m.s + m.o[0];
    pos[i + 1] = q[i + 1] * m.s + m.o[1];
    pos[i + 2] = q[i + 2] * m.s + m.o[2];
  }
  const ib = bytes(m.i);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(m.i32 ? new Uint32Array(ib) : new Uint16Array(ib), 1));
  g.computeVertexNormals();
  return g;
}

function hasWebGL() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
  } catch (e) {
    return false;
  }
}

/**
 * Load the model and build the viewer. Nothing is drawn until mount() puts it in a stage.
 * @param {string} modelUrl   where model.json is
 * @param {object} look       { paper, line }: the page's paper colour and line colour, for the line drawing
 * @param {boolean} calm      true when the visitor asked for reduced motion
 * @returns {Promise<object|null>}  the viewer, or null when the browser cannot draw it
 */
export async function createViewer(modelUrl, look = {}, calm = false) {
  if (!hasWebGL()) return null;
  const data = await (await fetch(modelUrl)).json();

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  const pixelRatio = () => Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pixelRatio());
  const canvas = renderer.domElement;
  canvas.className = 'model';
  canvas.setAttribute('role', 'img');
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  // The page scrolls under the model. Only a pinch (which arrives as ctrl + wheel) zooms.
  canvas.addEventListener('wheel', (event) => { if (!event.ctrlKey) event.stopImmediatePropagation(); }, { capture: true, passive: true });

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xC9D0DB, 2.3));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7);
  scene.add(key, fill);

  const camera = new THREE.PerspectiveCamera(32, 1, 1, 6000);
  camera.up.copy(UP);                           // the model is Z up
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.rotateSpeed = 0.9;
  controls.enabled = false;

  // ---- the parts
  const paperFace = new THREE.MeshBasicMaterial({
    color: new THREE.Color(look.paper || '#E8E5D2'), polygonOffset: true, polygonOffsetFactor: 1.5, polygonOffsetUnits: 1.5, side: THREE.DoubleSide,
  });
  const lineColor = new THREE.Color(look.line || '#2247C7');
  const parts = {};         // id -> { group, faces: [{ mesh, own }], edge, edgeColor, box }
  for (const o of data.objects) {
    if (!STACK[o.id]) continue;
    const group = new THREE.Group();
    const dark = o.id !== 'lid' && o.meshes.every((m) => PRINTED.has(m.c));
    const edge = new THREE.LineBasicMaterial({ transparent: true });
    const faces = [];
    for (const m of o.meshes) {
      const printed = PRINTED.has(m.c);
      const color = printed ? (o.id === 'lid' ? PRINT_BLUE : PRINT_BLACK) : (COLOR[m.c] ?? 0x999999);
      const own = new THREE.MeshStandardMaterial({
        color, roughness: METAL.has(m.c) ? 0.42 : (printed ? 0.72 : 0.86), metalness: METAL.has(m.c) ? 0.35 : 0,
        polygonOffset: true, polygonOffsetFactor: 1.5, polygonOffsetUnits: 1.5, side: THREE.DoubleSide, transparent: true,
      });
      const geometry = decode(m);
      const mesh = new THREE.Mesh(geometry, own);
      group.add(mesh, new THREE.LineSegments(new THREE.EdgesGeometry(geometry, printed ? 28 : 35), edge));
      faces.push({ mesh, own });
    }
    scene.add(group);
    parts[o.id] = { group, faces, edge, edgeColor: dark ? EDGE_ON_DARK : EDGE_ON_LIGHT, box: new THREE.Box3().setFromObject(group) };
  }
  const ids = ORDER.filter((id) => parts[id]);

  // ---- state
  let stage = null, mode = null, ui = null;
  let visible = false, frameId = 0, lastTime = 0;
  let held = false, turn = 0;
  const lean = new THREE.Vector2(), leanGoal = new THREE.Vector2();
  const bench = { follow: true, steps: BUILD_STEPS.map(() => 0), amount: 0.62, view: 'solid', chosen: null, framed: false };
  const benchSide = HOME.clone();
  const spread = {}, goal = {};
  for (const id of ids) { spread[id] = 0; goal[id] = 0; }

  const box = new THREE.Box3(), partBox = new THREE.Box3(), centre = new THREE.Vector3(), size = new THREE.Vector3();
  const side = new THREE.Vector3(), lightAt = new THREE.Vector3(), aim = new THREE.Vector3();

  // ---- where the parts are
  function setGoals() {
    for (const id of ids) {
      if (mode === 'hero') {
        goal[id] = held ? HERO_SPREAD : 0;
      } else if (bench.follow) {
        const step = BUILD_STEPS.findIndex((group) => group.includes(id));
        goal[id] = step < 0 ? 0 : SCROLL_SPREAD * (1 - smooth((bench.steps[step] - 0.12) / 0.7));
      } else {
        goal[id] = bench.amount;
      }
    }
  }

  function snapToGoals() {
    for (const id of ids) spread[id] = goal[id];
  }

  // Move each part's spread toward its goal. Returns true while anything is still moving.
  function ease(dt) {
    const rate = calm ? 1 : 1 - Math.exp(-dt * (mode === 'hero' ? 7 : 12));
    let moving = false;
    for (const id of ids) {
      const gap = goal[id] - spread[id];
      if (Math.abs(gap) < SETTLE) { spread[id] = goal[id]; continue; }
      spread[id] += gap * rate;
      moving = true;
    }
    return moving;
  }

  function place() {
    const lift = {};
    box.makeEmpty();
    for (const id of ids) {
      const rule = STACK[id];
      lift[id] = (rule.on ? lift[rule.on] ?? 0 : 0) + rule.lift * spread[id];
      const part = parts[id];
      part.group.position.set(0, (rule.slide || 0) * spread[id], lift[id]);
      box.union(partBox.copy(part.box).translate(part.group.position));
    }
    box.getCenter(centre);
    box.getSize(size);
  }

  // Distance at which a sphere around the model fills the view, whichever way the model is turned.
  function fitDistance() {
    const radius = size.length() / 2;
    const vertical = THREE.MathUtils.degToRad(camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
    return radius / Math.sin(Math.min(vertical, horizontal)) * 1.04;
  }

  function aimLights(angle) {
    key.position.copy(centre).add(lightAt.copy(KEY_LIGHT).applyAxisAngle(UP, angle));
    fill.position.copy(centre).add(lightAt.copy(FILL_LIGHT).applyAxisAngle(UP, angle));
  }

  // The hero camera circles the model and leans toward the pointer. The lights turn with it.
  function poseHero() {
    const around = HOME_TURN + turn + lean.x * HERO_LEAN.turn;
    const rise = clamp(HOME_RISE - lean.y * HERO_LEAN.rise, 0.14, 1.2);
    side.set(Math.cos(rise) * Math.cos(around), Math.cos(rise) * Math.sin(around), Math.sin(rise));
    const open = clamp(Math.max(...ids.map((id) => spread[id])) / HERO_SPREAD, 0, 1);
    aim.copy(centre);
    aim.z -= size.z * HERO_AIM_DROP * (1 - open);
    camera.position.copy(aim).addScaledVector(side, fitDistance() * (HERO_CLOSE + (1 - HERO_CLOSE) * open));
    camera.lookAt(aim);
    aimLights(around - HOME_TURN);
  }

  // The bench camera keeps the side the visitor turned it to and steps back far enough to fit.
  function poseBench(direction) {
    if (direction) side.copy(direction).normalize();
    else side.copy(camera.position).sub(controls.target).normalize();
    const distance = fitDistance();
    controls.target.copy(centre);
    camera.position.copy(centre).addScaledVector(side, distance);
    controls.minDistance = distance * 0.35;
    controls.maxDistance = distance * 1.8;
    aimLights(0);
    controls.update();
  }

  // ---- how the parts are drawn
  function applyLook() {
    const view = mode === 'hero' ? 'solid' : bench.view;
    const chosen = mode === 'hero' ? null : bench.chosen;
    for (const id of ids) {
      const part = parts[id];
      const picked = !chosen || id === chosen;
      const seeThrough = view === 'xray' ? !(chosen && picked) : (view === 'solid' && !picked);
      for (const face of part.faces) {
        face.mesh.material = view === 'lines' ? paperFace : face.own;
        face.own.opacity = seeThrough ? (view === 'xray' ? 0.16 : 0.14) : 1;
        face.own.depthWrite = !seeThrough;
      }
      part.edge.color.set(view === 'lines' ? lineColor : part.edgeColor);
      part.edge.opacity = view === 'lines' ? (picked ? 1 : 0.2) : (view === 'xray' ? (picked ? 0.7 : 0.3) : 0.5);
      part.edge.visible = view !== 'solid' || picked;
    }
  }

  // ---- drawing, only while something moves
  function requestDraw() {
    if (frameId || !visible || !stage) return;
    frameId = requestAnimationFrame(draw);
  }

  function draw(now) {
    frameId = 0;
    const dt = Math.min((now - lastTime) / 1000 || 0, 0.05);
    lastTime = now;
    let moving = ease(dt);
    place();
    if (mode === 'hero') {
      if (!calm) {
        turn += dt * HERO_TURN_RATE;
        lean.lerp(leanGoal, 1 - Math.exp(-dt * 5));
        moving = true;
      }
      poseHero();
    } else {
      if (moving) poseBench();
      moving = controls.update() || moving;
    }
    renderer.render(scene, camera);
    if (moving) requestDraw();
  }

  function resize() {
    if (!stage) return;
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setPixelRatio(pixelRatio());
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    place();
    if (mode === 'hero') poseHero(); else poseBench();
    renderer.render(scene, camera);
    requestDraw();
  }

  controls.addEventListener('change', requestDraw);
  controls.addEventListener('start', () => ui && ui.hint && ui.hint.classList.add('gone'));
  const sized = new ResizeObserver(resize);
  const seen = new IntersectionObserver((entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) { lastTime = performance.now(); requestDraw(); }
  });

  // ---- the bench's controls
  function showBenchState() {
    if (!ui) return;
    if (ui.follow) ui.follow.setAttribute('aria-pressed', String(bench.follow));
    for (const button of ui.views ? ui.views.querySelectorAll('button[data-view]') : []) {
      button.setAttribute('aria-pressed', String(button.dataset.view === bench.view));
    }
    for (const chip of ui.chips ? ui.chips.querySelectorAll('button[data-part]') : []) {
      chip.setAttribute('aria-pressed', String(chip.dataset.part === bench.chosen));
    }
    const part = PARTS.find((p) => p.id === bench.chosen);
    if (ui.about) ui.about.textContent = part ? `${part.name}: ${part.kind === 'printed' ? '3D-printed' : 'bought part'}` : '';
    showSlider();
  }

  // While the parts follow the page, the slider shows how far apart they are on the whole.
  function showSlider() {
    if (!ui || !ui.slider || !bench.follow) return;
    const open = bench.steps.reduce((sum, done) => sum + (1 - smooth((done - 0.12) / 0.7)), 0) / bench.steps.length;
    ui.slider.value = String(Math.round(open * SCROLL_SPREAD * 100));
  }

  function changeBench(change) {
    Object.assign(bench, change);
    showBenchState();
    if (mode !== 'bench') return;
    setGoals();
    applyLook();
    requestDraw();
    if (!frameId) { place(); poseBench(); renderer.render(scene, camera); }
  }

  /**
   * Wire the bench's controls. Call once.
   * @param {object} controlsUi  { slider, follow, reset, views, chips, about, hint, label }
   */
  function bind(controlsUi) {
    ui = controlsUi;
    for (const part of PARTS) {
      if (!parts[part.id]) continue;
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.dataset.part = part.id;
      chip.setAttribute('aria-pressed', 'false');
      chip.style.setProperty('--c', part.swatch);
      chip.innerHTML = '<i></i>';
      chip.append(part.name);
      chip.addEventListener('click', () => changeBench({ chosen: bench.chosen === part.id ? null : part.id }));
      ui.chips.appendChild(chip);
    }
    ui.slider.addEventListener('input', () => changeBench({ follow: false, amount: Number(ui.slider.value) / 100 }));
    ui.follow.addEventListener('click', () => changeBench({ follow: !bench.follow, amount: Number(ui.slider.value) / 100 }));
    for (const button of ui.views.querySelectorAll('button[data-view]')) {
      button.addEventListener('click', () => changeBench({ view: VIEWS.includes(button.dataset.view) ? button.dataset.view : 'solid' }));
    }
    ui.reset.addEventListener('click', () => {
      changeBench({ follow: true, view: 'solid', chosen: null });
      if (mode !== 'bench') return;
      snapToGoals();
      place();
      poseBench(HOME);
      requestDraw();
    });
    showBenchState();
  }

  // ---- the public face
  /**
   * Put the model in a stage and draw it there in the given mode ('hero' or 'bench').
   * The canvas moves; the model, its state and the WebGL context stay.
   */
  function mount(nextStage, nextMode) {
    if (stage === nextStage) return;
    if (mode === 'bench') benchSide.copy(camera.position).sub(controls.target).normalize();
    stage = nextStage;
    mode = nextMode;
    held = false;
    lean.set(0, 0);
    leanGoal.set(0, 0);
    stage.appendChild(canvas);
    canvas.setAttribute('aria-label', mode === 'hero'
      ? 'Three-dimensional model of the WillSpeak case, turning slowly.'
      : 'Three-dimensional model of the WillSpeak case. Drag to turn it.');
    canvas.tabIndex = mode === 'bench' ? 0 : -1;
    canvas.style.touchAction = mode === 'bench' ? 'none' : 'pan-y';
    controls.enabled = mode === 'bench';
    setGoals();
    snapToGoals();
    applyLook();
    place();
    if (mode === 'bench') {
      poseBench(bench.framed ? benchSide : HOME);
      bench.framed = true;
    }
    sized.disconnect();
    sized.observe(stage);
    seen.disconnect();
    seen.observe(stage);
    visible = true;
    lastTime = performance.now();
    resize();
    stage.dataset.ready = 'true';
  }

  /** Lift the hero's parts apart (true) or let them close (false). */
  function hold(on) {
    if (mode !== 'hero' || held === on) return;
    held = on;
    setGoals();
    requestDraw();
  }

  /** Lean the hero toward a point: x and y run from -1 to 1 across the screen, 0 0 is straight on. */
  function point(x, y) {
    leanGoal.set(clamp(x, -1, 1), clamp(y, -1, 1));
  }

  /** How far each build step has scrolled past, 0 to 1 each. The bench's parts settle to match. */
  function follow(steps) {
    bench.steps = BUILD_STEPS.map((_, i) => clamp(Number(steps[i]) || 0, 0, 1));
    if (!bench.follow) return;
    showSlider();
    if (mode !== 'bench') return;
    setGoals();
    requestDraw();
  }

  /**
   * A still picture of the closed case from the hero's starting side, as a PNG data URL.
   * For the build tool that makes the hero's placeholder image; call it before mount().
   */
  function snapshot(width, height) {
    renderer.setPixelRatio(1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    mode = 'hero';
    turn = 0;
    lean.set(0, 0);
    for (const id of ids) { spread[id] = 0; goal[id] = 0; }
    applyLook();
    place();
    poseHero();
    renderer.render(scene, camera);
    const picture = canvas.toDataURL('image/png');
    mode = null;
    renderer.setPixelRatio(pixelRatio());
    return picture;
  }

  return { canvas, mount, bind, hold, point, follow, snapshot, get mode() { return mode; } };
}
