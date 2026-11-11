const root = document.querySelector(".mm");
if (root) init(root);

function make(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text) n.textContent = text;
  return n;
}

function copy(tag, cls, from) {
  const n = make(tag, cls);
  for (const c of from.childNodes) n.append(c.cloneNode(true));
  return n;
}

function labelled(label, from) {
  const p = copy("p", "", from);
  p.prepend(make("span", "mm-lb", label), " ");
  return p;
}

function init(root) {
  const panel = root.querySelector(".mm-panel");
  const buttons = [...root.querySelectorAll("button[data-part]")];
  if (!panel || !buttons.length) return;
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let current = null;

  root.classList.add("mm-live");
  for (const b of buttons) b.setAttribute("aria-pressed", "false");

  function ids(b) {
    return (b.dataset.in || "").split(" ").filter(Boolean);
  }

  function show(id) {
    if (id === current) return true;
    const body = root.querySelector(`tbody[data-part="${CSS.escape(id)}"]`);
    if (!body) return false;
    current = id;
    const chosen = buttons.find((b) => b.dataset.part === id);
    const linked = chosen ? ids(chosen) : [];
    for (const b of buttons) {
      b.setAttribute("aria-pressed", String(b === chosen));
      b.classList.toggle("mm-in", linked.includes(b.dataset.part) || ids(b).includes(id));
    }

    const th = body.querySelector("th");
    const head = make("div", "mm-ph");
    head.append(make("p", "mm-k", body.dataset.kicker), make("h3", "", th.querySelector(".mm-pn").textContent));
    const facts = th.querySelector(".mm-pf");
    if (facts) head.append(make("p", "", facts.textContent));
    const rows = [...body.rows];
    const open = rows.filter((r) => r.dataset.type === "untested").length;
    const done = rows.length - open;
    const parts = [];
    if (done) parts.push(done === 1 ? "1 test" : `${done} tests`);
    if (open) parts.push(`${open} not tried`);
    const count = parts.join(", ");
    head.append(make("p", "mm-count", count));
    const back = make("button", "mm-back", "Back to the map");
    back.type = "button";
    back.addEventListener("click", () => {
      chosen.scrollIntoView({ block: "center", behavior: calm ? "auto" : "smooth" });
      chosen.focus({ preventScroll: true });
    });
    head.append(back);

    const list = make("ol", "mm-exps");
    for (const r of rows) {
      const li = make("li");
      li.dataset.type = r.dataset.type;
      li.append(copy("p", "mm-x-w", r.querySelector(".mm-w")));
      li.append(labelled("Result", r.querySelector(".mm-r")));
      li.append(labelled("Verdict", r.querySelector(".mm-v")));
      const note = r.querySelector(".mm-n");
      if (note.textContent.trim()) li.append(copy("p", "mm-x-n", note));
      list.append(li);
    }
    panel.replaceChildren(head, list);
    panel.hidden = false;
    return true;
  }

  function reveal() {
    const box = panel.getBoundingClientRect();
    if (box.top > innerHeight * 0.7 || box.bottom < 0) panel.scrollIntoView({ block: "start", behavior: calm ? "auto" : "smooth" });
  }

  root.addEventListener("focusin", (e) => {
    const b = e.target.closest("button[data-part]");
    if (b && b.matches(":focus-visible")) show(b.dataset.part);
  });
  root.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-part]");
    if (b && show(b.dataset.part)) reveal();
  });
}
