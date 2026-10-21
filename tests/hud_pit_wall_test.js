// Pit Wall tab: timing tower, gap chart, tyre stints and strategy odds render from the
// pushed state, and live WebSocket messages merge into the HUD state (deltas only redraw
// what changed).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ai_engineer/static/app.js'), 'utf8');
const elements = new Map();
const stub = () => new Proxy({
    style: {}, dataset: {}, innerHTML: '', textContent: '', value: '',
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, addEventListener() {}, dispatchEvent() {}, setAttribute() {}, querySelectorAll: () => [],
    querySelector: () => null, getContext: () => new Proxy({}, { get: () => () => {} }),
    getBoundingClientRect: () => ({ width: 100, height: 100 }),
}, { get: (target, key) => (key in target ? target[key] : undefined),
     set: (target, key, value) => { target[key] = value; return true; } });
const element = id => { if (!elements.has(id)) elements.set(id, stub()); return elements.get(id); };
const charts = [];
const frames = [];
const context = vm.createContext({
    document: { getElementById: element, createElement: () => stub(), querySelectorAll: () => [],
                addEventListener() {}, body: stub() },
    window: { addEventListener() {}, devicePixelRatio: 1, requestAnimationFrame: fn => frames.push(fn) },
    Chart: class { constructor(ctx, config) { Object.assign(this, config); this.updates = 0; charts.push(this); }
                   update() { this.updates += 1; } resize() {} },
    fetch: async () => ({ ok: true, json: async () => ({}) }), setInterval() {}, setTimeout() {},
    Event: class { constructor(type) { this.type = type; } },
    console, Math, Number, JSON, Date, Object, Array, String, Set,
});
vm.runInContext(source, context);
vm.runInContext('initPitWallCharts()', context);
const [gapChart, oddsChart] = charts;
const flush = () => { while (frames.length) frames.shift()(); };

const tower = [
    { index: 0, position: 1, name: 'Verstappen', team_id: 2, is_player: false, gap_s: 0, laps_down: 0,
      interval_s: null, last_lap_ms: 64400, best_lap_ms: 63900, fastest_lap: true, compound: 'M',
      tyre_age: 5, stops: 1, pit: 0, penalties_s: 0, status: 'RUN' },
    { index: 4, position: 2, name: null, team_id: 0, is_player: true, gap_s: 3.259, laps_down: 0,
      interval_s: 0.812, last_lap_ms: 64750, best_lap_ms: 64550, fastest_lap: false, compound: 'H',
      tyre_age: 4, stops: 1, pit: 0, penalties_s: 0, status: 'RUN' },
    { index: 7, position: 3, name: '<b>Gamer</b>', team_id: 9, is_player: false, gap_s: 95.1, laps_down: 1,
      interval_s: 91.8, last_lap_ms: 70000, best_lap_ms: 69000, fastest_lap: false, compound: 'S',
      tyre_age: 12, stops: 0, pit: 1, penalties_s: 0, status: 'RUN' },
    { index: 9, position: 20, name: 'Latifi', team_id: 3, is_player: false, gap_s: null, laps_down: 0,
      interval_s: null, last_lap_ms: null, best_lap_ms: null, fastest_lap: false, compound: null,
      tyre_age: null, stops: 0, pit: 0, penalties_s: 0, status: 'DNF' },
];
const odds = {
    lap: 20, laps_to_go: 32, runs: 300, recommended: '1-stop M-H',
    p_win: 0.12, p_podium: 0.46, p_points: 0.97, mean_position: 3.6, position_range: [2, 3, 6],
    position_probs: [0.12, 0.2, 0.14, 0.2, 0.1, 0.24],
    plans: [
        { label: '1-stop M-H', stops: 1, compounds: ['MEDIUM', 'HARD'], pit_laps: [24], p_best: 0.64,
          p_win: 0.12, p_podium: 0.46, p_points: 0.97, mean_position: 3.6, position_range: [2, 3, 6],
          time_vs_best_s: [0, 0, 3.1] },
        { label: '2-stop M-H-S', stops: 2, compounds: ['MEDIUM', 'HARD', 'SOFT'], pit_laps: [22, 40], p_best: 0.36,
          p_win: 0.08, p_podium: 0.38, p_points: 0.95, mean_position: 4.1, position_range: [2, 4, 7],
          time_vs_best_s: [0, 4.2, 9] },
    ],
    field: [], p_safety_car: 0.31, p_vsc: 0.22, p_rain: 0, pit_loss_s: [20.1, 21.4, 24.0],
    history: [{ lap: 15, p_win: 0.08, p_podium: 0.4, p_points: 0.95 },
              { lap: 20, p_win: 0.12, p_podium: 0.46, p_points: 0.97 }],
};
const full = {
    type: 'full', version: 10,
    state: {
        connected: true, speed: 280, rpm: 11000, gear: 7, throttle: 1, brake: 0, session_mode: 'race',
        lap_num: 20, total_laps: 52, position: 2, track_length: 4318,
        tyre_surface_temps: [95, 95, 95, 95], tyre_inner_temps: [95, 95, 95, 95],
        tyre_pressures: [22, 22, 23, 23], tyre_wear: [10, 11, 9, 10],
        speed_trace_best: [], speed_trace_current: [], corner_analysis: [], voice_log: [],
        race_report: [], race_control: [], opponent_threats: [],
        race_strategy: { best: { stops: 1, compounds: ['HARD', 'SOFT'], stint_laps: [10, 23], pit_laps: [29] },
                         alternatives: [], undercut_on_car_ahead: null, last_call: null, rivals: null },
        timing_tower: tower,
        gap_history: [
            { index: 0, name: 'Verstappen', team_id: 2, is_player: false, points: [[1, 0], [2, 0]] },
            { index: 4, name: null, team_id: 0, is_player: true, points: [[1, 1.2], [2, 2.5]] },
            { index: 7, name: 'Gamer', team_id: 2, is_player: false, points: [[1, 3.0], [2, 6.1]] },
        ],
        tyre_stints: [
            { index: 0, name: 'Verstappen', team_id: 2, is_player: false, position: 1,
              stints: [{ compound: 'S', start_lap: 1, end_lap: 3, open: false },
                       { compound: 'M', start_lap: 4, end_lap: 20, open: true }] },
            { index: 4, name: null, team_id: 0, is_player: true, position: 2,
              stints: [{ compound: 'M', start_lap: 1, end_lap: 12, open: false },
                       { compound: 'H', start_lap: 13, end_lap: 20, open: true }] },
        ],
        strategy_odds: odds,
    },
    status: { connected: true, data_age_seconds: 0.02 },
};

vm.runInContext('applyLiveMessage', context)(full);
flush();

// Timing tower
let html = element('pwTowerBody').innerHTML;
assert.match(html, /VER[\s\S]*LEADER/);
assert.match(html, /pw-player[\s\S]*YOU[\s\S]*\+3\.259[\s\S]*\+0\.812/);
assert.match(html, /pw-int-close">\+0\.812/);
assert.match(html, /\+1 LAP/);
assert.match(html, /pw-inpit">PIT/);
assert.match(html, /pw-fastest">1:03\.900/);
assert.match(html, /pw-out[\s\S]*DNF/);
assert.match(html, /&lt;b&gt;Gamer/);                    // gamertags are escaped
assert.doesNotMatch(html, /<b>Gamer/);
assert.equal(element('pwLapBadge').textContent, 'LAP 20 / 52');

// Gap chart: one line per car, player last and thickest, teammate dashed
assert.equal(gapChart.data.datasets.length, 3);
assert.equal(gapChart.data.datasets[2].label, 'YOU');
assert.equal(gapChart.data.datasets[2].borderWidth, 3);
assert.equal(JSON.stringify(gapChart.data.datasets[0].data), '[{"x":1,"y":0},{"x":2,"y":0}]');
assert.equal(JSON.stringify(gapChart.data.datasets[1].borderDash), '[5,3]');   // Gamer shares Verstappen's team
assert.equal(gapChart.options.scales.y.reverse, true);
assert.equal(element('gapChartEmpty').style.display, 'none');

// Stints: real sets plus our planned ones (dashed), positioned by lap
html = element('pwStints').innerHTML;
assert.match(html, /compound-soft[\s\S]*width:5\.7\d*%[\s\S]*S3/);
assert.match(html, /compound-medium\s+pw-open/);
assert.match(html, /pw-plan[^"]*"[^>]*left:38\.4\d*%;width:17\.3\d*%/);   // H laps 21-29
assert.match(html, /compound-soft pw-plan/);                             // then S laps 30-52
assert.match(html, /pw-now/);

// Strategy odds
assert.equal(element('pwOddsBody').style.display, '');
assert.equal(element('pwOddsEmpty').style.display, 'none');
assert.equal(element('pwWin').innerHTML, '12%');
assert.equal(element('pwPodium').innerHTML, '46%');
assert.equal(element('pwPoints').innerHTML, '97%');
assert.match(element('pwWinTrend').innerHTML, /▲ 4 pts<\/span> since L15/);
assert.equal(element('pwExpected').innerHTML, 'P3');
assert.match(element('pwPlans').innerHTML, /pw-best[\s\S]*1-STOP[\s\S]*64%[\s\S]*box L24/);
assert.match(element('pwPlans').innerHTML, /2-STOP[\s\S]*36%[\s\S]*\+4\.2s/);
assert.match(element('pwDist').innerHTML, /pw-podium[\s\S]*pw-current/);
assert.match(element('pwFactors').innerHTML, /SC 31%[\s\S]*PIT LOSS 21\.4s/);
assert.equal(JSON.stringify(oddsChart.data.datasets[0].data), '[{"x":15,"y":8},{"x":20,"y":12}]');
assert.match(element('pwOddsMeta').textContent, /300 simulated races from lap 20/);

// A delta only redraws what changed: a speed update leaves the tower and charts alone.
element('pwTowerBody').innerHTML = 'untouched';
const gapUpdates = gapChart.updates;
vm.runInContext('applyLiveMessage', context)({ type: 'delta', version: 11, state: { speed: 300 },
                                               status: { connected: true, data_age_seconds: 0.01 } });
flush();
assert.equal(element('pwTowerBody').innerHTML, 'untouched');
assert.equal(gapChart.updates, gapUpdates);
assert.equal(element('speedDisplay').textContent, 300);
assert.equal(vm.runInContext('hudState.lap_num', context), 20);         // merged, not replaced

// Several messages before a frame: one draw with all their keys.
vm.runInContext('applyLiveMessage', context)({ type: 'delta', version: 12, state: { strategy_odds: null }, status: {} });
vm.runInContext('applyLiveMessage', context)({ type: 'delta', version: 13, state: { timing_tower: [] }, status: {} });
assert.equal(frames.length, 1);
flush();
assert.equal(element('pwOddsBody').style.display, 'none');
assert.match(element('pwTowerBody').innerHTML, /Waiting for the field/);

// A new full state (reconnect) replaces everything: keys it lacks are gone.
vm.runInContext('applyLiveMessage', context)({ type: 'full', version: 1, state: { speed: 10 }, status: {} });
flush();
assert.equal(vm.runInContext('hudState.lap_num', context), undefined);
assert.match(element('pwStints').innerHTML, /No tyre data yet/);

console.log('Pit wall tab and live push client: PASS');
