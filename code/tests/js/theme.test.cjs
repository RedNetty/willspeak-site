const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const script = fs.readFileSync(path.join(__dirname, '../../static/common.js'), 'utf8');

function page({ saved = null, dark = false, storageFails = false, legacyMedia = false } = {}) {
  const attributes = new Map(), documentEvents = new Map(), windowEvents = new Map();
  const changes = [], selectors = [];
  const media = { matches: dark };
  media[legacyMedia ? 'addListener' : 'addEventListener'] = (...args) => changes.push(args.at(-1));
  const root = {
    style: {},
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: name => attributes.delete(name),
  };
  const badge = { textContent: '', hidden: true };
  const label = { textContent: 'Review' };
  const icon = { textContent: 'review icon' };
  const navAttributes = new Map([['href', '/review']]);
  const review = {
    children: [icon, label, badge], textContent: '',
    getAttribute: name => navAttributes.get(name),
    setAttribute: (name, value) => navAttributes.set(name, value),
    querySelector: selector => selector === '[data-nav-badge]' ? badge : label,
  };
  let stored = saved;
  const context = vm.createContext({
    document: {
      documentElement: root, readyState: 'loading',
      addEventListener(name, callback) {
        if (!documentEvents.has(name)) documentEvents.set(name, []);
        documentEvents.get(name).push(callback);
      },
      querySelectorAll: selector => selector === '[data-theme-select]' ? selectors :
        selector === '.app-nav a, header nav a' ? [review] : [],
    },
    localStorage: {
      getItem() { if (storageFails) throw Error('blocked'); return stored; },
      setItem(key, value) { if (storageFails) throw Error('blocked'); assert.equal(key, 'willspeak-theme'); stored = value; },
    },
    matchMedia: () => media,
    addEventListener: (name, callback) => windowEvents.set(name, callback),
  });
  vm.runInContext(script, context);
  if (!storageFails && ['light', 'dark'].includes(saved)) assert.equal(attributes.get('data-theme'), saved);
  const controls = [0, 1].map(() => ({
    value: '', events: new Map(),
    addEventListener(name, callback) { this.events.set(name, callback); },
  }));
  selectors.push(...controls);
  documentEvents.get('DOMContentLoaded').forEach(callback => callback());
  return {
    attributes, root, controls, badge, review, icon, label, navAttributes,
    stored: () => stored,
    choose(value) { controls[0].value = value; controls[0].events.get('change')(); },
    systemDark(value) { media.matches = value; changes.forEach(fn => fn({ matches: value })); },
    storage(value, key = 'willspeak-theme') { windowEvents.get('storage')({ key, newValue: value }); },
    badgeCount(totals) { context.totals = totals; vm.runInContext('navBadge(totals)', context); },
  };
}

test('saved explicit theme applies before page controls exist and persists to the next page', () => {
  const p = page({ saved: 'dark' });
  assert.equal(p.attributes.get('data-theme'), 'dark');
  assert.equal(p.root.style.colorScheme, 'dark');
  assert.deepEqual(p.controls.map(select => select.value), ['dark', 'dark']);
  p.choose('light');
  assert.equal(p.attributes.get('data-theme'), 'light');
  assert.equal(p.stored(), 'light');
  assert.deepEqual(p.controls.map(select => select.value), ['light', 'light']);
  assert.equal(page({ saved: p.stored() }).attributes.get('data-theme'), 'light');
});

test('system preference follows OS changes while an explicit preference stays fixed', () => {
  const p = page();
  assert.equal(p.attributes.has('data-theme'), false);
  assert.equal(p.root.style.colorScheme, 'light');
  p.systemDark(true);
  assert.equal(p.root.style.colorScheme, 'dark');
  p.choose('light');
  p.systemDark(false); p.systemDark(true);
  assert.equal(p.attributes.get('data-theme'), 'light');
  assert.equal(p.root.style.colorScheme, 'light');
  p.choose('system');
  assert.equal(p.attributes.has('data-theme'), false);
  assert.equal(p.root.style.colorScheme, 'dark');
  assert.equal(p.stored(), 'system');
});

test('blocked storage, invalid saved values and older media listeners remain usable', () => {
  const p = page({ storageFails: true, dark: true, legacyMedia: true });
  assert.equal(p.root.style.colorScheme, 'dark');
  p.choose('light');
  assert.equal(p.root.style.colorScheme, 'light');
  p.choose('system'); p.systemDark(false);
  assert.equal(p.root.style.colorScheme, 'light');
  assert.equal(page({ saved: 'invalid' }).attributes.has('data-theme'), false);
});

test('theme changes in another page update controls and clearing storage restores system', () => {
  const p = page({ saved: 'light', dark: true });
  p.storage('dark', 'unrelated');
  assert.equal(p.attributes.get('data-theme'), 'light');
  p.storage('dark');
  assert.equal(p.attributes.get('data-theme'), 'dark');
  assert.deepEqual(p.controls.map(select => select.value), ['dark', 'dark']);
  p.storage(null, null);
  assert.equal(p.attributes.has('data-theme'), false);
  assert.equal(p.root.style.colorScheme, 'dark');
});

test('pending review badge preserves navigation icon and label and describes the count', () => {
  const p = page();
  const children = [...p.review.children];
  p.badgeCount({ pending: 3 });
  assert.equal(p.badge.textContent, '3');
  assert.equal(p.badge.hidden, false);
  assert.equal(p.navAttributes.get('aria-label'), 'Review, 3 takes waiting');
  assert.equal(p.label.textContent, 'Review');
  assert.deepEqual(p.review.children, children);
  p.badgeCount({ pending: 1 });
  assert.equal(p.navAttributes.get('aria-label'), 'Review, 1 take waiting');
  p.badgeCount({ pending: 0 });
  assert.equal(p.badge.hidden, true);
  assert.equal(p.badge.textContent, '');
  assert.equal(p.navAttributes.get('aria-label'), 'Review');
  p.badgeCount({ pending: 'invalid' });
  assert.equal(p.badge.hidden, true);
});

test('legacy review links without a separate badge still show the count', () => {
  const p = page();
  p.review.querySelector = () => null;
  p.badgeCount({ pending: 2 });
  assert.equal(p.review.textContent, 'Review (2)');
  p.badgeCount(null);
  assert.equal(p.review.textContent, 'Review');
});
