// The HUD's race strategy card shows the plan we are on (Plan A) and a fresh Plan B call.
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
    snapshot.race_strategy = { ...snapshot.race_strategy, rivals: null, replan: {
        plan_a: { label: 'One-stop (M-H)', stops: 1, compounds: ['MEDIUM', 'HARD'], pit_laps: [24] },
        status: 'watching', call_active: false, last_call: null,
        challenger: { label: 'Two-stop (M-H-S)', time_gain_s: 4.2, position_gain: 0.3, p_better: 0.7,
                      clears_margin: true },
    } };
    await vm.runInContext('fetchTelemetry()', context);
    let html = element('strategyList').innerHTML;
    assert.match(html, /PLAN A ONE-STOP \(M-H\) · B\? TWO-STOP \(M-H-S\) \+4\.2s/);
    assert.match(html, /BOX L24/);
    assert.match(html, /MODEL 1-STOP/);
    assert.doesNotMatch(html, /plan-b-call/);
    snapshot.race_strategy.replan = { ...snapshot.race_strategy.replan, status: 'switched', challenger: null,
        call_active: true, last_call: { lap: 22, message: 'Safety car. Plan B, Plan B. Box this lap for hards. Box, box.' } };
    await vm.runInContext('fetchTelemetry()', context);
    html = element('strategyList').innerHTML;
    assert.match(html, /plan-b-call[\s\S]*PLAN B · L22[\s\S]*Box this lap for hards/);
    assert.ok(html.indexOf('PLAN B') < html.indexOf('PLAN A'));
    snapshot.race_strategy.replan = null;
    await vm.runInContext('fetchTelemetry()', context);
    html = element('strategyList').innerHTML;
    assert.doesNotMatch(html, /PLAN A|PLAN B|MODEL/);
    assert.match(html, /1-STOP/);
    console.log('HUD strategy card shows Plan A and the Plan B call: PASS');
})().catch(error => { console.error(error); process.exit(1); });
