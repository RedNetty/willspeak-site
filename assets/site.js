// What the page does once it has loaded:
//
//   - marks the section in view in the top bar
//   - opens every folded part before the page is printed
//   - runs the waveform behind the device at the top
//   - loads the 3D model, shows it at the top of the page (hold to take it
//     apart) and hands the same model to the device section when that comes
//     into view, where the parts settle as the build steps scroll past
//
// Everything is readable without any of it: until the model has loaded, and in
// a browser that cannot draw it, a still picture stands in both places.
import { startWave } from './wave.js';

const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const tokens = getComputedStyle(document.documentElement);
const token = (name) => tokens.getPropertyValue(name).trim();
const byId = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const HOLD_TEXT = { false: 'Hold to take it apart', true: 'Release to close it' };
const TOUCH_HOLD_DELAY = 140;   // ms a finger must rest before it counts as a hold, so a scroll that starts on the model does not open it

/** Mark the top bar's link for the section that is in the middle of the screen. */
function markSectionInView() {
  const links = new Map([...document.querySelectorAll('.bar nav a[href^="#"]')].map((a) => [a.getAttribute('href').slice(1), a]));
  if (!('IntersectionObserver' in window)) return;
  const spy = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links.values()) link.removeAttribute('aria-current');
      links.get(entry.target.id)?.setAttribute('aria-current', 'true');
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  for (const id of links.keys()) {
    const section = byId(id);
    if (section) spy.observe(section);
  }
}

/** Printing shows everything: open each folded part first, and put them back afterwards. */
function unfoldForPrint() {
  let opened = [];
  window.addEventListener('beforeprint', () => {
    opened = [...document.querySelectorAll('details.fold:not([open]), .chart details:not([open])')];
    for (const fold of opened) fold.open = true;
  });
  window.addEventListener('afterprint', () => {
    for (const fold of opened) fold.open = false;
    opened = [];
  });
}

/** Close an open toolbar menu on a click outside it or on Escape. */
function closeMenusFromOutside() {
  const close = (except) => {
    for (const menu of document.querySelectorAll('details.menu[open]')) if (menu !== except) menu.open = false;
  };
  document.addEventListener('click', (event) => close(event.target.closest('details.menu')));
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(null); });
}

/** Start the waveform behind the hero's device. Returns its handle, or null without a canvas. */
function startHeroWave(stage) {
  const canvas = byId('wave');
  if (!canvas || !canvas.getContext) return null;
  return startWave(canvas, {
    gap: () => stage.clientHeight * 0.22,      // the wave changes behind the device, where the model hides the join
    ink: token('--ink'),
    accent: token('--accent'),
    calm,
  });
}

/**
 * Work out how far each build step has scrolled past, and tell `report` on every change.
 * A step runs from 0, when its top reaches a line low on the screen, to 1, when it has
 * travelled up to the model (or by its own height, when the model sits beside it).
 */
function watchBuildSteps(report) {
  const build = byId('build'), stuck = byId('viewer');
  const items = [...document.querySelectorAll('#build-steps > li')];
  if (!build || !stuck || !items.length) return;
  build.classList.add('live');
  let queued = false;
  function measure() {
    queued = false;
    const model = stuck.getBoundingClientRect(), list = items[0].parentElement.getBoundingClientRect();
    const beside = list.left >= model.right - 1;
    const top = beside ? 0 : clamp(model.bottom, 0, window.innerHeight * 0.75);
    const line = top + (window.innerHeight - top) * (beside ? 0.62 : 0.72);
    let current = 0;
    const done = items.map((item, index) => {
      const box = item.getBoundingClientRect();
      const share = clamp((line - box.top) / Math.min(box.height, line - top), 0, 1);
      if (share > 0) current = index;
      return share;
    });
    items.forEach((item, index) => item.classList.toggle('is-on', index === current));
    report(done);
  }
  const queue = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(measure);
  };
  window.addEventListener('scroll', queue, { passive: true });
  window.addEventListener('resize', queue);
  measure();
}

/** Wire the hero: holding the model or its button lifts the parts apart, and the model leans toward the pointer. */
function wireHero(viewer, hero, stage, button, wave) {
  const label = button.querySelector('span');
  let pressed = false, waiting = 0;
  const press = (on) => {
    window.clearTimeout(waiting);
    if (pressed === on) return;
    pressed = on;
    viewer.hold(on);
    button.setAttribute('aria-pressed', String(on));
    label.textContent = HOLD_TEXT[on];
  };
  stage.addEventListener('pointerdown', (event) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    if (event.pointerType === 'touch') waiting = window.setTimeout(() => press(true), TOUCH_HOLD_DELAY);
    else press(true);
  });
  for (const type of ['pointerup', 'pointercancel', 'blur']) window.addEventListener(type, () => press(false));
  button.addEventListener('keydown', (event) => {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    event.preventDefault();
    if (!event.repeat) press(true);
  });
  button.addEventListener('keyup', () => press(false));
  button.addEventListener('blur', () => press(false));

  hero.addEventListener('pointermove', (event) => {
    if (event.pointerType === 'touch') return;
    const box = stage.getBoundingClientRect();
    viewer.point((event.clientX / window.innerWidth) * 2 - 1, (event.clientY - (box.top + box.height / 2)) / (window.innerHeight / 2));
    if (wave) wave.point(event.pageX, event.pageY);
  });
  hero.addEventListener('pointerleave', () => {
    viewer.point(0, 0);
    if (wave) wave.point(null);
  });
}

/** Load the 3D model and keep it in whichever of the two stages is on screen. */
async function startModel(wave) {
  const hero = byId('hero'), heroStage = byId('hero-stage'), benchStage = byId('stage');
  let latestSteps = [];
  let viewer = null;
  watchBuildSteps((done) => {
    latestSteps = done;
    if (viewer) viewer.follow(done);
  });
  try {
    const { createViewer } = await import('./viewer.js');
    viewer = await createViewer(new URL('./model.json', import.meta.url).href, { paper: token('--paper'), line: token('--accent-text') }, calm);
  } catch (error) {
    console.warn('The 3D model did not start; the pictures stay.', error);
  }
  if (!viewer) return;

  viewer.bind({
    slider: byId('explode'), follow: byId('follow'), reset: byId('reset'), views: byId('views'),
    chips: byId('chips'), about: byId('about'), hint: byId('hint'),
  });
  viewer.follow(latestSteps);
  wireHero(viewer, hero, heroStage, byId('hold'), wave);

  const show = {
    hero() {
      viewer.mount(heroStage, 'hero');
      const still = byId('hero-still');
      if (still) {
        still.classList.add('gone');
        window.setTimeout(() => still.remove(), 450);
      }
      byId('hold').hidden = false;
    },
    bench() {
      viewer.mount(benchStage, 'bench');
      byId('fallback')?.remove();
      byId('toolbar').hidden = false;
      byId('hint').hidden = false;
    },
  };
  if (!('IntersectionObserver' in window)) {
    show.hero();
    return;
  }
  // The model goes to whichever stage is on or near the screen. The results section lies between
  // the two, so they are never both in view; if a very tall screen shows both, the hero keeps it.
  const near = new Set();
  const watch = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) near.add(entry.target);
      else near.delete(entry.target);
    }
    if (near.has(heroStage)) show.hero();
    else if (near.has(benchStage)) show.bench();
  }, { rootMargin: '300px 0px' });
  watch.observe(heroStage);
  watch.observe(benchStage);
}

markSectionInView();
unfoldForPrint();
closeMenusFromOutside();
startModel(startHeroWave(byId('hero-stage')));
