const top = document.createElement('div');
const bottom = document.createElement('div');
top.className = 'veil veil-top';
bottom.className = 'veil veil-bottom';
for (const v of [top, bottom]) {
  v.setAttribute('aria-hidden', 'true');
  document.body.append(v);
}

let queued = false;
function update() {
  queued = false;
  const y = window.scrollY;
  const end = document.documentElement.scrollHeight - window.innerHeight;
  top.classList.toggle('off', y < 40);
  bottom.classList.toggle('off', end - y < 40);
}
function queue() {
  if (!queued) {
    queued = true;
    requestAnimationFrame(update);
  }
}
window.addEventListener('scroll', queue, { passive: true });
window.addEventListener('resize', queue);
update();
