// Lab notebook: a tooltip for each mark in the charts, on hover and on keyboard focus.
// The charts are complete without it: each mark carries its own <title>, and every chart has its table.
const tip = document.createElement('div');
tip.className = 'nb-tip';
tip.setAttribute('role', 'tooltip');
tip.hidden = true;
document.body.append(tip);

for (const mark of document.querySelectorAll('.nb-svg [data-tip]')) {
  mark.querySelector('title')?.remove();
  mark.setAttribute('aria-label', mark.dataset.tip);
  mark.setAttribute('role', 'img');
}

function show(mark) {
  tip.textContent = mark.dataset.tip;
  tip.hidden = false;
  const box = mark.getBoundingClientRect();
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const x = Math.min(Math.max(8, box.left + box.width / 2 - w / 2), window.innerWidth - w - 8);
  const above = box.top - h - 8;
  tip.style.left = `${x}px`;
  tip.style.top = `${above > 8 ? above : box.bottom + 8}px`;
}
const hide = () => { tip.hidden = true; };

// On a narrow screen the top bar scrolls sideways; bring the current page's item into view.
const nav = document.querySelector('.pill nav');
const here = nav?.querySelector('[aria-current="true"]');
if (nav && here && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = here.offsetLeft + here.offsetWidth - nav.clientWidth + 12;
const markOf = (event) => event.target.closest?.('.nb-svg [data-tip]');

document.addEventListener('pointerover', (event) => { const m = markOf(event); if (m) show(m); });
document.addEventListener('pointerout', (event) => { if (markOf(event)) hide(); });
document.addEventListener('focusin', (event) => { const m = markOf(event); if (m) show(m); else hide(); });
document.addEventListener('focusout', hide);
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') hide(); });
window.addEventListener('scroll', hide, { passive: true });
