// The live track map draws the stored circuit (closed loop) and pit lane, keeps the car on
// it, and shows a flashback badge when the game rewinds.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ai_engineer/static/app.js'), 'utf8');

const calls = [];
const ctx = new Proxy({}, {
    get: (target, key) => (key in target ? target[key]
        : (...args) => { calls.push([key, ...args]); }),
    set: (target, key, value) => { calls.push(['set', key, value]); target[key] = value; return true; },
});
const stub = () => new Proxy({
    style: {}, dataset: {}, innerHTML: '', textContent: '', value: '', width: 400, height: 300,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, addEventListener() {}, setAttribute() {}, querySelectorAll: () => [],
    querySelector: () => null, getContext: () => ctx, parentElement: null,
    getBoundingClientRect: () => ({ width: 400, height: 300 }),
}, { get: (target, key) => (key in target ? target[key] : undefined),
     set: (target, key, value) => { target[key] = value; return true; } });
const elements = new Map();
const element = id => { if (!elements.has(id)) elements.set(id, stub()); return elements.get(id); };
element('trackCanvas').parentElement = stub();
const context = vm.createContext({
    document: { getElementById: element, createElement: () => stub(), querySelectorAll: () => [],
                addEventListener() {}, body: stub() },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    Chart: class { constructor(c, config) { Object.assign(this, config); } update() {} },
    fetch: async () => ({ ok: true, json: async () => ({}) }), setInterval() {}, setTimeout() {},
    console, Math, Number, JSON, Date, Object, Array, String,
});
vm.runInContext(source, context);
vm.runInContext('initCanvas()', context);

const outline = [[0, 0], [100, 0], [100, 100], [0, 100]];
const cars = [[10, 0, 1, 2, 0, 0], [100, 50, 8, 3, 0, 0], [50, 0, 0, 1, 1, 0]];
const snapshot = { cars, track_outline: outline, pit_lane: [[0, -10], [50, -10]], current_lap_trail: [[0, 0], [50, 0]],
                   track_map: [], track_map_source: 'stored', car_x: 50, car_z: 0, flashback_count: 0,
                   braking_zones: [], ers_zones: [] };
context.snapshot = snapshot;
vm.runInContext('updateMapState(snapshot); redrawTrackMap();', context);
const texts = calls.filter(c => c[0] === 'fillText').map(c => c[1]);
assert.ok(texts.includes('TRACK MAP'), 'stored outline is labelled as the track map');
assert.ok(calls.some(c => c[0] === 'closePath'), 'stored circuit is drawn as a closed loop');
assert.ok(calls.some(c => c[0] === 'set' && c[1] === 'strokeStyle' && /255, 176, 0/.test(c[2])),
          'pit lane is drawn with amber edges');
const widths = calls.filter(c => c[0] === 'set' && c[1] === 'lineWidth').map(c => c[2]);
assert.ok(Math.max(...widths) >= 12, 'the circuit is drawn as a wide road');
assert.ok(!texts.some(t => /FLASHBACK/.test(t)));
const fills = calls.filter(c => c[0] === 'set' && c[1] === 'fillStyle').map(c => c[2]);
assert.ok(fills.includes('#dc0000') && fills.includes('#ff8700'), 'other cars are drawn in team colours');

// A partly mapped circuit with a hole: the pen lifts instead of drawing a chord.
calls.length = 0;
context.snapshot = { track_outline: [], cars: [], current_lap_trail: [],
                     track_map: [[0, 0], [10, 0], null, [50, 50], [60, 50]] };
vm.runInContext('updateMapState(snapshot); redrawTrackMap();', context);
assert.equal(calls.filter(c => c[0] === 'moveTo').length >= 2, true);

// A flashback (count goes up) shows the badge; the rewound trail replaces the old one.
calls.length = 0;
context.snapshot = { ...snapshot, track_map: [], flashback_count: 1, current_lap_trail: [[0, 0], [20, 0]] };
vm.runInContext('updateMapState(snapshot); redrawTrackMap();', context);
assert.ok(calls.some(c => c[0] === 'fillText' && /FLASHBACK/.test(c[1])));
assert.deepEqual(vm.runInContext('mapState.trail', context), [[0, 0], [20, 0]]);

// No stored outline yet: the live trail is drawn instead, not closed.
calls.length = 0;
context.snapshot = { track_outline: [], pit_lane: [], track_map: [[0, 0], [10, 10], [20, 5]],
                     current_lap_trail: [[0, 0], [10, 10], [20, 5]], track_map_source: 'recording' };
vm.runInContext('updateMapState(snapshot); redrawTrackMap();', context);
assert.ok(calls.some(c => c[0] === 'fillText' && c[1] === 'MAPPING CIRCUIT...'));
assert.ok(!calls.some(c => c[0] === 'closePath'));
console.log('ok');
