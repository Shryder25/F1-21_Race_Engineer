// The HUD's tyre/fuel model card renders [low, expected, high] ranges from model_ranges.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ai_engineer/static/app.js'), 'utf8');
const elements = new Map();
const stub = () => new Proxy({
    style: {}, dataset: {}, innerHTML: '', textContent: '', value: '',
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, addEventListener() {}, setAttribute() {}, querySelectorAll: () => [],
    querySelector: () => null, getContext: () => new Proxy({}, { get: () => () => {} }),
    getBoundingClientRect: () => ({ width: 100, height: 100 }),
}, { get: (target, key) => (key in target ? target[key] : undefined),
     set: (target, key, value) => { target[key] = value; return true; } });
const element = id => { if (!elements.has(id)) elements.set(id, stub()); return elements.get(id); };
let snapshot = {
    connected: true, speed: 200, rpm: 10000, gear: 5, throttle: 1, brake: 0,
    tyre_surface_temps: [95, 95, 95, 95], tyre_inner_temps: [95, 95, 95, 95],
    tyre_pressures: [22, 22, 23, 23], tyre_wear: [10, 11, 9, 10], track_length: 4318,
    speed_trace_best: [], speed_trace_current: [], corner_analysis: [], voice_log: [],
    race_report: [], race_control: [], opponent_threats: [],
    model_ranges: {
        basis: 'track', fit_laps: 42, stint_laps: 6, simulated_fit: true,
        wear_per_lap_pct: [2.1, 2.4, 2.7], wear_in_5_laps_pct: [21.0, 22.5, 24.1],
        deg_s_per_lap: [0.04, 0.06, 0.08], pit_ideal_lap: [17, 19, 21], pit_latest_lap: [22, 24, null],
        finish_without_stop_prob: 0.12, fuel_per_lap_kg: [1.62, 1.7, 1.78], fuel_margin_laps: [-0.4, 0.6, 1.5],
    },
};
const context = vm.createContext({
    document: { getElementById: element, createElement: () => stub(), querySelectorAll: () => [],
                addEventListener() {}, body: stub() },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    Chart: class { constructor(ctx, config) { Object.assign(this, config); } update() {} },
    fetch: async () => ({ ok: true, json: async () => snapshot }), setInterval() {}, setTimeout() {},
    console, Math, Number, JSON, Date, Object, Array, String,
});
vm.runInContext(source, context);
(async () => {
    await vm.runInContext('fetchTelemetry()', context);
    const html = element('modelList').innerHTML;
    assert.equal(element('modelCard').style.display, '');
    assert.match(html, /WEAR \/ LAP/);
    assert.match(html, /2\.40% \(2\.10–2\.70\)/);
    assert.match(html, /L19 \(17–21\)/);
    assert.match(html, /L24 \(22–flag\)/);
    assert.match(html, /12%/);
    assert.match(html, /0\.6 laps \(-0\.4–1\.5\)/);
    assert.match(element('modelBasis').textContent, /fitted here, 48 laps, sim/);
    snapshot = { ...snapshot, model_ranges: null };
    await vm.runInContext('fetchTelemetry()', context);
    assert.equal(element('modelCard').style.display, 'none');
    console.log('HUD tyre/fuel model card shows ranges: PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
