// The Track Database page shows a qualifying run plan and setup tips when the session is
// qualifying, and follows the game into qualifying.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ai_engineer/static/app.js'), 'utf8');
const elements = new Map();
const events = [];
const stub = () => new Proxy({
    style: {}, dataset: {}, innerHTML: '', textContent: '', value: '',
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, addEventListener() {}, setAttribute() {}, querySelectorAll: () => [],
    querySelector: () => null, getContext: () => new Proxy({}, { get: () => () => {} }),
    getBoundingClientRect: () => ({ width: 100, height: 100 }),
    dispatchEvent(event) { events.push(event.type); },
}, { get: (target, key) => (key in target ? target[key] : undefined),
     set: (target, key, value) => { target[key] = value; return true; } });
const element = id => { if (!elements.has(id)) elements.set(id, stub()); return elements.get(id); };
const rows = [stub(), stub()];
let snapshot = { connected: true, speed_trace_best: [], speed_trace_current: [], corner_analysis: [],
                 voice_log: [], race_report: [], race_control: [], opponent_threats: [] };
const context = vm.createContext({
    document: { getElementById: element, createElement: () => stub(),
                querySelectorAll: sel => (sel === '.strategy-row' ? rows : []),
                addEventListener() {}, body: stub() },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    Event: class { constructor(type) { this.type = type; } },
    Chart: class { constructor(ctx, config) { Object.assign(this, config); } update() {} },
    fetch: async () => ({ ok: true, json: async () => snapshot }), setInterval() {}, setTimeout() {},
    console, Math, Number, JSON, Date, Object, Array, String,
});
vm.runInContext(source, context);

const setup = {
    aerodynamics: { front_wing: 8, rear_wing: 8 }, transmission: { diff_on_throttle: 72, diff_off_throttle: 57 },
    suspension_geometry: { front_camber: -3.5, rear_camber: -1.9, front_toe: 0.1, rear_toe: 0.25 },
    suspension: { front_suspension: 7, rear_suspension: 6, front_anti_roll: 9, rear_anti_roll: 7,
                  front_ride_height: 2, rear_ride_height: 3 },
    brakes: { brake_pressure: 100, brake_bias: 56 },
    tyres: { front_left_pressure: 23.8, front_right_pressure: 23.2, rear_left_pressure: 22, rear_right_pressure: 22 },
};
context.data = {
    track_name: 'Silverstone Circuit', country: 'UK', weather: 'dry', session: 'qualifying', setup,
    downforce_level: 'Medium', strategy: { race_laps: 52, notes: ['race note'] },
    qualifying: { tyre: 'SOFT', push_laps_per_set: 1, out_lap: 'Normal out-lap.',
                  session_timing: 'Go late.', notes: ['quali note'], ai: { reasoning: 'Track evolves.' } },
};
vm.runInContext('renderSetupData(data)', context);
assert.equal(element('strategyTitle').textContent, 'QUALIFYING PLAN');
assert.equal(element('strategyLaps').textContent, 'ONE LAP');
assert.equal(element('qualiPlan').style.display, '');
assert.equal(rows[0].style.display, 'none');
assert.equal(element('qualiTyre').textContent, 'Soft');
assert.equal(element('qualiPushLaps').textContent, 1);
assert.equal(element('qualiOutLap').textContent, 'Normal out-lap.');
assert.equal(element('strategyAi').textContent, 'DeepSeek: Track evolves.');
assert.match(element('setupConditions').textContent, /DRY · QUALIFYING/);
assert.match(element('tipTransmission').textContent, /^Quali/);

context.data = { ...context.data, session: 'race', qualifying: null };
vm.runInContext('renderSetupData(data)', context);
assert.equal(element('strategyTitle').textContent, 'RACE STRATEGY');
assert.equal(element('qualiPlan').style.display, 'none');
assert.equal(rows[0].style.display, '');
assert.equal(element('strategyLaps').textContent, '52 LAPS');
assert.doesNotMatch(element('tipTransmission').textContent, /^Quali/);

(async () => {
    snapshot = { ...snapshot, session_mode: 'qualifying', session_label: 'Q1' };
    await vm.runInContext('fetchTelemetry()', context);
    assert.equal(element('sessionSelect').value, 'qualifying');
    assert.deepEqual(events, ['change']);
    element('sessionSelect').value = 'race';                // driver overrides it
    await vm.runInContext('fetchTelemetry()', context);
    assert.equal(element('sessionSelect').value, 'race');
    console.log('HUD Track Database shows the qualifying plan and follows the game into qualifying: PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
