// The moving waveform behind the device at the top of the page.
//
// It is a drawing of what the device does, made from arithmetic: uneven, broken
// bars travel in from the left, pass behind the device and leave on the right
// as even, rounded bars. It is decoration. No recording of anyone's speech is
// behind it. Bars near the pointer swell, so the wave follows the visitor.

const PITCH = 10;           // px from one bar to the next
const SPEED = 26;           // px a second, left to right
const EDGE_FADE = 90;       // px over which the bars fade at the screen edges
const GAP_FADE = 70;        // px over which they fade next to the device
const REACH = 90;           // px around the pointer where bars swell

// A repeatable number from 0 to 1 for a whole-number seed.
function hash(n) {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}

// Smooth noise from 0 to 1 along a line.
function noise(x) {
  const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f);
  return hash(i) * (1 - u) + hash(i + 1) * u;
}

// Height of bar k on the way in, 0 to 1: bursts and dropouts, with every bar off by its own amount.
function unclear(k) {
  const burst = noise(k * 0.09 + 40);
  const dropout = noise(k * 0.031 + 7) < 0.34 ? 0.12 : 1;
  const jag = 0.3 + 0.7 * hash(k * 3 + 1);
  return Math.min(1, (0.1 + 0.9 * burst * burst) * dropout * jag * 1.5);
}

// Height of bar k on the way out, 0 to 1: even syllables under a slow swell.
function clear(k) {
  const syllable = 0.5 - 0.5 * Math.cos(k * 0.62);
  const swell = 0.62 + 0.38 * Math.sin(k * 0.085 + 1.3);
  return 0.16 + 0.84 * syllable * swell;
}

/**
 * Draw the waveform on a canvas and keep it moving while it is on screen.
 * @param {HTMLCanvasElement} canvas  stretched across the hero by the stylesheet
 * @param {object} options
 * @param {() => number} options.gap     half the width, in px, to keep clear around the middle for the device
 * @param {string} options.ink           colour of the bars on the way in
 * @param {string} options.accent        colour of the bars on the way out
 * @param {boolean} options.calm         true when the visitor asked for reduced motion: one still frame
 * @returns {{ point: (x: number|null, y?: number) => void }}  point() tells the wave where the pointer is, in page px, or null when it leaves
 */
export function startWave(canvas, { gap, ink, accent, calm }) {
  const context = canvas.getContext('2d');
  let width = 0, height = 0, visible = true, frameId = 0, startTime = 0;
  const pointer = { x: 0, y: 0, pull: 0, goal: 0 };

  function size() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function bar(x, half, thickness, round) {
    if (round) {
      context.beginPath();
      context.roundRect(x - thickness / 2, height / 2 - half, thickness, half * 2, thickness / 2);
      context.fill();
    } else {
      context.fillRect(x - thickness / 2, height / 2 - half, thickness, half * 2);
    }
  }

  function paint(seconds) {
    context.clearRect(0, 0, width, height);
    if (!width || !height) return;
    const middle = width / 2, clearHalf = gap(), tallest = height * 0.3;
    const travel = seconds * SPEED;
    const first = Math.floor(-travel / PITCH) - 1, count = Math.ceil(width / PITCH) + 2;
    for (let n = 0; n < count; n++) {
      const k = first + n;
      const x = k * PITCH + travel;
      const fromMiddle = Math.abs(x - middle);
      if (fromMiddle < clearHalf) continue;
      const fade = Math.min(1, (fromMiddle - clearHalf) / GAP_FADE, x / EDGE_FADE, (width - x) / EDGE_FADE);
      if (fade <= 0) continue;
      const incoming = x < middle;
      const near = pointer.pull * Math.exp(-(((x - pointer.x) / REACH) ** 2));
      const half = Math.max(1.5, tallest * (incoming ? unclear(k) : clear(k)) * (1 + 0.9 * near));
      if (incoming) {
        context.globalAlpha = 0.5 * fade;
        context.fillStyle = ink;
        bar(x + (hash(k * 5 + 2) - 0.5) * 4, half, 2 + Math.round(hash(k * 7 + 3) * 2), false);
      } else {
        context.globalAlpha = 0.9 * fade;
        context.fillStyle = accent;
        bar(x, half, 4, true);
      }
    }
    context.globalAlpha = 1;
  }

  function frame(now) {
    frameId = 0;
    if (!startTime) startTime = now;
    pointer.pull += (pointer.goal - pointer.pull) * 0.12;
    paint(calm ? 0 : (now - startTime) / 1000);
    if (visible && (!calm || Math.abs(pointer.goal - pointer.pull) > 0.01)) request();
  }

  function request() {
    if (!frameId) frameId = requestAnimationFrame(frame);
  }

  new ResizeObserver(() => { size(); request(); }).observe(canvas);
  new IntersectionObserver((entries) => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible) request();
  }).observe(canvas);
  size();
  request();

  return {
    point(pageX, pageY) {
      if (pageX === null) {
        pointer.goal = 0;
      } else {
        const box = canvas.getBoundingClientRect();
        pointer.x = pageX - (box.left + window.scrollX);
        pointer.y = pageY - (box.top + window.scrollY);
        pointer.goal = pointer.y > -40 && pointer.y < height + 40 ? 1 : 0;
      }
      request();
    },
  };
}
