// Header pill shows the real track status (yellow, SC, VSC, red, blue), and the sector boxes
// keep sector 3 on screen: the last lap's split shows (dimmed) until this lap's is driven.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ai_engineer/static/app.js'), 'utf8');
const elements = new Map();
const stub = () => {
    const target = {
        style: {}, dataset: {}, innerHTML: '', textContent: '', value: '', className: '',
        appendChild() {}, addEventListener() {}, dispatchEvent() {}, setAttribute() {}, querySelectorAll: () => [],
        querySelector: () => null, getContext: () => new Proxy({}, { get: () => () => {} }),
        getBoundingClientRect: () => ({ width: 100, height: 100 }),
    };
    target.classList = {
        add: (...names) => { target.className = [target.className, ...names].filter(Boolean).join(' '); },
        remove() {}, toggle() {}, contains: name => target.className.split(' ').includes(name),
    };
    return new Proxy(target, { get: (t, key) => (key in t ? t[key] : undefined),
                               set: (t, key, value) => { t[key] = value; return true; } });
};
const element = id => { if (!elements.has(id)) elements.set(id, stub()); return elements.get(id); };
const context = vm.createContext({
    document: { getElementById: element, createElement: () => stub(), querySelectorAll: () => [],
                addEventListener() {}, body: stub() },
    window: { addEventListener() {}, devicePixelRatio: 1, requestAnimationFrame() {} },
    Chart: class { constructor(ctx, config) { Object.assign(this, config); } update() {} resize() {} },
    fetch: async () => ({ ok: true, json: async () => ({}) }), setInterval() {}, setTimeout() {},
    Event: class { constructor(type) { this.type = type; } },
    console, Math, Number, JSON, Date, Object, Array, String, Set,
});
vm.runInContext(source, context);
const render = vm.runInContext('renderTelemetry', context);
const base = { connected: true, tyre_surface_temps: [95, 95, 95, 95], tyre_pressures: [22, 22, 23, 23],
               speed_trace_best: [], speed_trace_current: [], corner_analysis: [], voice_log: [] };
const pill = () => [element('safetyCarPill').className, element('safetyCarText').textContent];

// Track status pill
render({ ...base, track_status: { state: 'YELLOW', label: 'YELLOW FLAG · S2' } }, null);
assert.deepEqual(pill(), ['safety-car-pill sc-yellow', 'YELLOW FLAG · S2']);
render({ ...base, track_status: { state: 'RED', label: 'RED FLAG' } }, null);
assert.deepEqual(pill(), ['safety-car-pill sc-red', 'RED FLAG']);
render({ ...base, track_status: { state: 'SC', label: 'SAFETY CAR DEPLOYED' } }, null);
assert.deepEqual(pill(), ['safety-car-pill sc-full', 'SAFETY CAR DEPLOYED']);
render({ ...base, track_status: { state: 'BLUE', label: 'BLUE FLAG' } }, null);
assert.deepEqual(pill(), ['safety-car-pill sc-blue', 'BLUE FLAG']);
render({ ...base, track_status: { state: 'CLEAR', label: 'TRACK CLEAR' } }, null);
assert.deepEqual(pill(), ['safety-car-pill sc-none', 'TRACK CLEAR']);
render({ ...base, safety_car: 'VSC' }, null);                      // older server: safety_car only
assert.deepEqual(pill(), ['safety-car-pill sc-vsc', 'VIRTUAL SAFETY CAR']);

// Just crossed the line: all three of the last lap's sectors show, S3 included
const lastLap = { last_lap_sectors_ms: { 1: 28000, 2: 29000, 3: 25450 },
                  last_lap_sector_refs_ms: { 1: 28100, 2: 28900, 3: 25600 } };
render({ ...base, current_sectors_ms: {}, personal_sectors_ms: { 1: 28000, 2: 28900, 3: 25450 }, ...lastLap }, null);
assert.equal(element('secTime3').textContent, '25.450');
assert.equal(element('secDelta3').textContent, '-0.15');
assert.equal(element('sectorBox3').className, 'sector-box sec-prev sec-purple');
assert.equal(element('secTime2').textContent, '29.000');
assert.equal(element('sectorBox2').className, 'sector-box sec-prev sec-yellow');

// S1 done on the new lap: S1 is live, S2/S3 still the last lap's
render({ ...base, current_sectors_ms: { 1: 27900 }, personal_sectors_ms: { 1: 28000, 2: 28900, 3: 25450 },
         ...lastLap }, null);
assert.equal(element('secTime1').textContent, '27.900');
assert.equal(element('sectorBox1').className, 'sector-box sec-purple');
assert.equal(element('secTime3').textContent, '25.450');

// Nothing yet this session
render({ ...base, current_sectors_ms: {}, personal_sectors_ms: {}, last_lap_sectors_ms: {} }, null);
assert.equal(element('secTime3').textContent, '--.---');
console.log('hud track status + sector 3: ok');
