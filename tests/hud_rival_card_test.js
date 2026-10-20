// The HUD's race strategy card lists nearby rivals' predicted next stop and tyre.
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
    race_strategy: {
        best: { stops: 1, compounds: ['MEDIUM', 'HARD'], stint_laps: [20, 30], pit_laps: [20] },
        alternatives: [], undercut_on_car_ahead: null, last_call: null,
        rivals: { nearby: [
            { name: 'Hamilton', position: 2, compound: 'MEDIUM', tyre_age: 14, will_stop: true,
              stop_lap: 22, stop_lap_range: [21, 22, 23], next_compound: 'HARD' },
            { name: null, position: 4, compound: 'SOFT', tyre_age: 9, will_stop: true,
              stop_lap: 18, stop_lap_range: [15, 18, 21], next_compound: 'MEDIUM' },
            { name: 'Norris', position: 5, compound: 'HARD', tyre_age: 3, will_stop: false,
              stop_lap: null, stop_lap_range: [null, null, null], next_compound: null },
            { name: 'Sainz', position: 6, compound: 'MEDIUM', tyre_age: 20, will_stop: true,
              stop_lap: 49, stop_lap_range: [47, 49, null], next_compound: 'SOFT' },
        ] },
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
    const html = element('strategyList').innerHTML;
    assert.equal(element('strategyCard').style.display, '');
    assert.match(html, /P2 Hamilton M14/);
    assert.match(html, /L22 → H/);
    assert.match(html, /P4 S9/);
    assert.match(html, /L18 \(15-21\) → M/);
    assert.match(html, /P5 Norris H3[\s\S]*NO STOP/);
    assert.match(html, /L49\+ → S/);
    snapshot = { ...snapshot, race_strategy: { ...snapshot.race_strategy, rivals: null } };
    await vm.runInContext('fetchTelemetry()', context);
    assert.doesNotMatch(element('strategyList').innerHTML, /Hamilton/);
    console.log('HUD strategy card shows rival stop predictions: PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
