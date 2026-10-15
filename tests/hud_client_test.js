const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../ai_engineer/web_hud.py'), 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const element = id => {
    if (!elements.has(id)) elements.set(id, {
        style: {}, classList: { add() {} }, appendChild() {}, getContext() { return {}; }
    });
    return elements.get(id);
};
let snapshot = {
    connected: true, speed: 200, rpm: 10000, gear: 5, throttle: 1, brake: 0,
    tyre_surface_temps: [95, 95, 95, 95], tyre_pressures: [22, 22, 23, 23],
    best_lap_time_ms: 90000, delta_ms: 100, track_length: 6000,
    speed_trace_best: [[0, 100], [10, 120]],
    speed_trace_current: [[2000, 200], [2015, 210]], corner_analysis: []
};
const context = vm.createContext({
    document: { getElementById: element, createElement: () => element(Symbol()), querySelectorAll: () => [] },
    Chart: class { constructor(ctx, config) { Object.assign(this, config); } update() {} },
    fetch: async () => ({ json: async () => snapshot }), setInterval() {}, console
});
vm.runInContext(script, context);
(async () => {
    await vm.runInContext('fetchTelemetry()', context);
    const chart = vm.runInContext('speedChart', context);
    assert.equal(chart.options.scales.x.type, 'linear');
    assert.equal(chart.data.datasets[0].data[0].x, 0);
    assert.equal(chart.data.datasets[1].data[0].x, 2000);
    assert.equal(chart.data.datasets[1].data[1].x, 2015);
    snapshot = { ...snapshot, best_lap_time_ms: 0, speed_trace_best: [], speed_trace_current: [[100, 180]] };
    await vm.runInContext('fetchTelemetry()', context);
    assert.equal(chart.data.datasets[0].data.length, 0);
    assert.equal(chart.data.datasets[1].data[0].x, 100);
    snapshot = { ...snapshot, corner_analysis: [{name: 'Abbey', delta: null,
        current_apex: 240, best_apex: 275, brake_point: null,
        reference_source: 'track guide', advice: 'Guide speed only.'}] };
    await vm.runInContext('fetchTelemetry()', context);
    assert.match(element('cornerList').innerHTML, /Guide only/);
    assert.match(element('cornerList').innerHTML, /Not observed/);
    assert.doesNotMatch(element('cornerList').innerHTML, /nulls/);
    snapshot = { ...snapshot, corner_analysis: [] };
    await vm.runInContext('fetchTelemetry()', context);
    assert.equal(element('cornerList').innerHTML, '');
    snapshot = { ...snapshot, voice_log: [{ time: '12:00', message: 'Old track call', status: 'spoken' }] };
    await vm.runInContext('fetchTelemetry()', context);
    assert.match(element('voiceList').innerHTML, /Old track call/);
    snapshot = { ...snapshot, voice_log: [] };
    await vm.runInContext('fetchTelemetry()', context);
    assert.equal(element('voiceList').innerHTML, '');
    console.log('HUD speed traces retain independent distances and clear previous baselines: PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });
