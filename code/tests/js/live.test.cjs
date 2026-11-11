const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const staticDir = path.join(__dirname, '../../static');
const html = fs.readFileSync(path.join(staticDir, 'live.html'), 'utf8');
const script = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).join('\n');

async function page() {
  const elements = new Map();
  for (const match of html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)) {
    elements.set('#' + match[1], {
      value: /\bvalue="([^"]+)"/.exec(match[0])?.[1] || '',
      textContent: '', className: '', style: {}, hidden: false,
      classList: { toggle() {} }, setAttribute() {}, focus() {}, addEventListener() {},
      replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; },
      append(child) { this.children.push(child); }, children: [],
    });
  }
  elements.set('#care .grid2', {children: []});
  let now = 0;
  const timers = new Map(), calls = [], spoken = [], confirmations = [];
  const settings = { stop_phrase: 'purple elephant', sleep_phrase: 'voice go to sleep',
    wake_phrase: 'voice wake up', speak_delay_s: 0, stop_text_sim: 0.85 };
  const localVoice = { voiceURI: 'local', name: 'Local', lang: 'en-US', localService: true, default: true };
  const remoteVoice = { voiceURI: 'cloud', name: 'Cloud', lang: 'en-US', localService: false };
  const context = vm.createContext({
    console, Blob, FormData, Float32Array, ArrayBuffer, DataView,
    Option: function(label, value) { this.label = label; this.value = value; },
    document: { querySelector: selector => elements.get(selector), querySelectorAll: () => [], addEventListener() {} },
    localStorage: { getItem: () => null, setItem() {} },
    navigator: { mediaDevices: { getUserMedia: () => new Promise(() => {}) } },
    performance: { now: () => now },
    setTimeout(fn) { const id = Symbol(); timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); }, setInterval() {},
    speechSynthesis: { speaking: false, getVoices: () => [remoteVoice, localVoice],
      cancel() {}, speak(utterance) { spoken.push(utterance); } },
    SpeechSynthesisUtterance: function(text) { this.text = text; },
    fetch: async (url, options) => {
      calls.push({ url, options });
      if (url === '/api/live/confirm') confirmations.push(options.body.get('clip_id') + (options.body.get('stage') === 'end' ? ':end' : ''));
      return { ok: true, json: async () => url.endsWith('/settings') ? settings :
        url.endsWith('/status') ? { asr_installed: true, asr_ready: true, awake: true } :
        url.endsWith('/board') ? { phrases: [] } : {} };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(staticDir, 'common.js'), 'utf8'), context);
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  const run = code => vm.runInContext(code, context);
  run('audioReady = true; asrReady = true; ctx = {sampleRate:16000};');
  return { context, run, elements, calls, spoken, confirmations, timers,
    advance(ms) { now += ms; },
    output(text, id = 1, delay = 0) {
      context.item = { res: { text, clip_id: id, latency_ms: 1, model: 'test' },
        decision: { text, delay_s: delay }, cut: false };
      run('outputQueue.push(item); drainOutput();');
    } };
}

test('uses local voices, confirms when playback starts and reports when it ends', async () => {
  const p = await page();
  assert.deepEqual(Array.from(p.run('voices'), v => v.voiceURI), ['local']);
  p.output('hello'); p.run('drainOutput()');
  assert.equal(p.spoken.length, 1);
  assert.equal(p.confirmations.length, 0);
  p.spoken[0].onstart();
  assert.deepEqual(p.confirmations, ['1']);
  // End of playback is reported so the retraction window counts from it.
  p.spoken[0].onend();
  assert.deepEqual(p.confirmations, ['1', '1:end']);
});

test('failed playback is not confirmed and reports attention', async () => {
  const p = await page();
  p.output('hello'); p.run('drainOutput()'); p.spoken[0].onerror();
  assert.equal(p.confirmations.length, 0);
  assert.equal(p.elements.get('#state').textContent, 'Voice needs attention');
});

test('waits for ongoing recording and decoding before speaking', async () => {
  const p = await page();
  p.run('recording = true'); p.output('first'); p.output('second', 2);
  assert.equal(p.spoken.length, 0);
  p.run('recording = false; requestsPending = 1; drainOutput()');
  assert.equal(p.spoken.length, 0);
  p.run('requestsPending = 0; drainOutput(); drainOutput()');
  assert.equal(p.spoken[0].text, 'first');
  p.spoken[0].onend(); p.advance(550); p.run('drainOutput(); drainOutput()');
  assert.equal(p.spoken[1].text, 'second');
});

test('stop cancels a delayed sentence without confirming it', async () => {
  const p = await page();
  p.output('wrong words', 1, 2);
  p.context.fetch = async () => ({ ok: true, json: async () => ({decision: {action: 'stop'}}) });
  await p.run('transcribe(new Blob(), {generation:audioGeneration, test:false})');
  p.advance(3000); p.run('drainOutput()');
  assert.equal(p.spoken.length, 0);
  assert.equal(p.run('pending'), null);
  assert.equal(p.elements.get('#hyptext').className, 'cancelled');
});

test('late stop test remains a preview after its capture window', async () => {
  const p = await page();
  p.run("$('#testsay').onclick(); beginUtterance(); chunks=[new Float32Array(16000)]; testing=false;");
  p.context.fetch = async (_, options) => {
    assert.equal(options.body.get('preview_only'), 'true');
    return { ok: true, json: async () => ({text: 'purple elephant', decision: {action:'stop', by:'text'}}) };
  };
  await p.run('endUtterance()');
  assert.match(p.elements.get('#testres').textContent, /STOP PHRASE detected/);
  assert.equal(p.spoken.length, 0);
});

test('decode requests stay in capture order and discard stale responses', async () => {
  const p = await page();
  const releases = [];
  p.context.fetch = () => new Promise(resolve => releases.push(resolve));
  const first = p.run('recording=true; chunks=[new Float32Array(16000)]; endUtterance()');
  const second = p.run('recording=true; chunks=[new Float32Array(16000)]; endUtterance()');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(releases.length, 1);
  p.run('microphoneFailed()');
  releases[0]({ ok: true, json: async () => ({text:'stale', decision:{action:'speak',text:'stale',delay_s:0}}) });
  await Promise.all([first, second]);
  assert.equal(releases.length, 1);
  assert.equal(p.spoken.length, 0);
  p.run('resumeListening()');
  assert.equal(p.elements.get('#state').textContent, 'Microphone needs attention');
});

test('lazy recognizer can accept its first utterance', async () => {
  const p = await page();
  p.context.fetch = async () => ({ok:true,json:async () => ({asr_installed:true, asr_ready:false, asr_eager:false})});
  await p.run('asrReady=false; refreshStatus()');
  assert.equal(p.run('asrReady'), true);
});

test('missing local voice does not fall back to a cloud voice', async () => {
  const p = await page();
  p.context.speechSynthesis.getVoices = () => [{voiceURI:'cloud',lang:'en-US',localService:false}];
  p.run('loadVoices(); speak("hello")');
  assert.equal(p.spoken.length, 0);
  assert.equal(p.elements.get('#state').textContent, 'Voice needs setup');
});

test('HTML escaping handles prompt text and quoted attributes', async () => {
  const p = await page();
  assert.equal(p.run('escapeHtml(`<img src=x onerror="bad">&\'`)'), '&lt;img src=x onerror=&quot;bad&quot;&gt;&amp;&#39;');
});

function blocks(p, db, n) { for (let i = 0; i < n; i++) { p.run(`vadTick(${db})`); p.advance(43); } }

test('gate behaves as the room gate until he has been heard', async () => {
  const p = await page();
  p.run('noiseDb = -55');
  assert.equal(p.run('gateThreshold(-39)'), -39);
  blocks(p, -36, 3);
  assert.equal(p.run('recording'), true);
});

test('a spoken sentence teaches the gate his level', async () => {
  const p = await page();
  p.context.item = { res: { text: 'hello', clip_id: 7, latency_ms: 1, model: 'test', qc: { speech_dbfs: -20 } },
    decision: { text: 'hello', delay_s: 0 }, cut: false };
  p.run('outputQueue.push(item); drainOutput(); drainOutput();');
  p.spoken[0].onstart();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(Array.from(p.run('gate.record'), r => r.db), [-20]);
  assert.equal(p.run('gateThreshold(-39)'), -32);
});

test('a quieter voice in the room no longer opens the gate, his does', async () => {
  const p = await page();
  p.run('noiseDb = -55; learnLevel(1, -20)');
  blocks(p, -36, 5);
  assert.equal(p.run('recording'), false);
  blocks(p, -26, 3);
  assert.equal(p.run('recording'), true);
});

test('Not him removes a clip from the record and New session clears it', async () => {
  const p = await page();
  p.run('learnLevel(1, -20); learnLevel(2, -40); forgetLevel("2")');
  assert.deepEqual(Array.from(p.run('gate.record'), r => r.id), [1]);
  p.run('resetGate()');
  assert.equal(p.run('hisLevel()'), null);
});

test('record keeps the last twenty sentences and uses their median', async () => {
  const p = await page();
  for (let i = 0; i < 25; i++) p.run(`learnLevel(${i}, ${-30 + i})`);
  assert.equal(p.run('gate.record.length'), 20);
  assert.equal(p.run('hisLevel()'), -30 + 14.5);
});

test('the leak lowers his level only after three quiet minutes, and learning ends it', async () => {
  const p = await page();
  p.run("noiseDb = -55; learnLevel(1, -20); gate.lastOpen = performance.now(); setState('listening', listeningText())");
  blocks(p, -35, 3);
  assert.equal(p.run('gate.leakDb'), 0);
  p.advance(180000);
  blocks(p, -45, 1); blocks(p, -35, 3);
  assert.equal(p.run('gate.leakDb'), 1);
  assert.match(p.elements.get('#state').textContent, /Having trouble hearing him today/);
  p.run('learnLevel(2, -30)');
  assert.equal(p.run('gate.leakDb'), 0);
});

test('the leak never takes the gate below the room gate', async () => {
  const p = await page();
  p.run('learnLevel(1, -20); gate.leakDb = 40');
  assert.equal(p.run('gateThreshold(-39)'), -39);
});

test('the gate level an utterance opened at is sent for the server log', async () => {
  const p = await page();
  let sent = null;
  p.context.fetch = async (url, options) => {
    if (url === '/api/live/transcribe') sent = options.body.get('gate_dbfs');
    return { ok: true, json: async () => ({ text: '', decision: { action: 'ignore' } }) };
  };
  p.run('noiseDb = -55; learnLevel(1, -20)');
  blocks(p, -26, 3);
  await p.run('chunks = [new Float32Array(16000)]; endUtterance()');
  assert.equal(sent, '-32.0');
});
