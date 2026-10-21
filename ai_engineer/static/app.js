// Escape text that comes from the game (e.g. online player names) before using innerHTML.
function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

/**
 * F1 2021 AI Race Engineer — Client Controller & Telemetry Engine
 * ==============================================================
 * Manages:
 *   1. Live telemetry pushed over a WebSocket (polling fallback) & UI rendering
 *      (Timing Tower, Tyres, Pedals)
 *   2. Canvas Live Track Map with X/Z coordinates, car dot, braking & ERS zones
 *   3. 24-Track Database & AI Setup Advisor
 *   4. Latency audio cue & team radio voice command dispatching
 *   5. Corner Delta & Speed Trace Chart
 */

// Global state
let currentTrackId = null;   // set from the game's session packet (or the dropdown)
let lastKnownCoords = [];
let trackCanvas, trackCtx;
let speedChart = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCanvas();
    initSpeedChart();
    initSetupAdvisor();
    initVoiceCommands();
    buildRevLeds();
    initPitWallCharts();

    // Live state: pushed by the server over a WebSocket; polls only while that is down.
    startLiveFeed();
});

// ──────────────────────────────────────────────
// 1. Tab Switching
// ──────────────────────────────────────────────
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            tabs.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.classList.add('active');

            // Trigger canvas redraw or chart update on tab switch
            if (btn.dataset.tab === 'tab-live') {
                resizeCanvas();
                redrawTrackMap();
            } else if (btn.dataset.tab === 'tab-corners' && speedChart) {
                speedChart.resize();
            } else if (btn.dataset.tab === 'tab-pitwall') {
                if (gapChart) gapChart.resize();
                if (oddsChart) oddsChart.resize();
            }
        });
    });
}

// ──────────────────────────────────────────────
// 2. LED Rev Lights Bar
// ──────────────────────────────────────────────
const TOTAL_LEDS = 18;
function buildRevLeds() {
    const bar = document.getElementById('revLedBar');
    if (!bar) return;
    bar.innerHTML = '';
    for (let i = 0; i < TOTAL_LEDS; i++) {
        const led = document.createElement('div');
        led.className = 'rev-led';
        bar.appendChild(led);
    }
}

function updateRevLeds(rpm, maxRpm = 13500) {
    const leds = document.querySelectorAll('.rev-led');
    if (!leds.length) return;
    const idleRpm = 4000;
    const pct = Math.max(0, Math.min(1, (rpm - idleRpm) / (maxRpm - idleRpm)));
    const activeCount = Math.floor(pct * TOTAL_LEDS);

    leds.forEach((led, idx) => {
        led.className = 'rev-led';
        if (idx < activeCount) {
            if (idx < TOTAL_LEDS * 0.45) {
                led.classList.add('on-green');
            } else if (idx < TOTAL_LEDS * 0.80) {
                led.classList.add('on-red');
            } else {
                led.classList.add('on-purple');
            }
        }
    });
}

// ──────────────────────────────────────────────
// 3. Live Track Map Canvas (Feature 3)
// ──────────────────────────────────────────────
// The circuit outline is stored per track on the server (track_maps.py), so the whole
// track is drawn from the first frame and only the car moves. Until a track has been
// mapped, the live trail of the current lap is drawn instead (the whole lap, nothing
// is erased). A flashback cuts the trail back to where the game rewound to.
let lastCarX = 0, lastCarZ = 0;
let lastBrakingZones = [];
let lastErsZones = [];
const mapState = {
    outline: [],          // whole circuit [[x, z], ...]
    pitLane: [],          // pit lane path [[x, z], ...]
    cars: [],             // every car [x, z, team, position, isPlayer, inPit]
    trail: [],            // this lap's driven points
    source: 'recording',  // 'stored' once the outline is known
    inPit: false,
    flashbackCount: null,
    flashbackUntil: 0,
};

function initCanvas() {
    trackCanvas = document.getElementById('trackCanvas');
    if (!trackCanvas) return;
    trackCtx = trackCanvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    document.getElementById('btnFitMap')?.addEventListener('click', () => {
        redrawTrackMap(true);
    });

    document.getElementById('btnClearTrace')?.addEventListener('click', () => {
        // Clear locally so the button feels instant, then tell the server: it also forgets
        // the stored outline for this track so the next clean lap maps it again.
        lastKnownCoords = [];
        lastBrakingZones = [];
        lastErsZones = [];
        mapState.outline = [];
        mapState.pitLane = [];
        mapState.trail = [];
        mapState.source = 'recording';
        redrawTrackMap();
        fetch('/api/reset_map', { method: 'POST' }).catch(() => {});
    });
}

function resizeCanvas() {
    if (!trackCanvas) return;
    const rect = trackCanvas.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
        trackCanvas.width = Math.round(rect.width);
        trackCanvas.height = Math.round(rect.height);
    }
}

// Take the track-map keys of one telemetry snapshot into mapState.
function updateMapState(data, nowMs = Date.now()) {
    if (Array.isArray(data.track_outline)) mapState.outline = data.track_outline;
    if (Array.isArray(data.pit_lane)) mapState.pitLane = data.pit_lane;
    if (Array.isArray(data.cars)) mapState.cars = data.cars;
    if (Array.isArray(data.current_lap_trail)) mapState.trail = data.current_lap_trail;
    if (Array.isArray(data.track_map)) lastKnownCoords = data.track_map;
    if (data.track_map_source) mapState.source = data.track_map_source;
    mapState.inPit = Boolean(data.pit_status);
    if (data.flashback_count !== undefined) {
        if (mapState.flashbackCount !== null && data.flashback_count > mapState.flashbackCount) {
            mapState.flashbackUntil = nowMs + 2500;
        }
        mapState.flashbackCount = data.flashback_count;
    }
    if (data.car_x !== undefined) lastCarX = data.car_x;
    if (data.car_z !== undefined) lastCarZ = data.car_z;
    if (data.braking_zones) lastBrakingZones = data.braking_zones;
    if (data.ers_zones) lastErsZones = data.ers_zones;
}

// What the map draws: the stored outline (closed loop) if known, else the live trail.
function mapBase() {
    if (mapState.outline.length >= 2) return { line: mapState.outline, closed: true };
    const trail = lastKnownCoords.length >= 2 ? lastKnownCoords : mapState.trail;
    return { line: trail, closed: false };
}

function redrawTrackMap(forceFit = false) {
    // Keep the canvas the size of its panel, so the whole circuit always fits on screen.
    const rect = trackCanvas?.parentElement?.getBoundingClientRect?.();
    if (rect && rect.width > 0 && rect.height > 0 &&
            (Math.round(rect.width) !== trackCanvas.width || Math.round(rect.height) !== trackCanvas.height)) {
        resizeCanvas();
    }
    drawTrackMap(mapBase().line, lastCarX, lastCarZ, lastBrakingZones, lastErsZones, forceFit);
}

function tracePath(points, toCanvas, closed) {
    // A null entry is a hole in a partly mapped circuit: lift the pen there.
    trackCtx.beginPath();
    let penDown = false;
    for (const p of points) {
        if (!p) { penDown = false; continue; }
        const [cx, cz] = toCanvas(p[0], p[1]);
        if (penDown) trackCtx.lineTo(cx, cz); else trackCtx.moveTo(cx, cz);
        penDown = true;
    }
    if (closed) trackCtx.closePath();
}

// F1 2021 team ids (Participants packet) to team colours, for the car dots.
const TEAM_COLOURS = {
    0: '#00d2be', 1: '#dc0000', 2: '#1e41ff', 3: '#005aff', 4: '#006f62',
    5: '#0090ff', 6: '#4e7c9b', 7: '#b6babd', 8: '#ff8700', 9: '#900000',
};

// Draw a path as a road: light edges with dark tarmac between them.
function strokeRoad(points, toCanvas, closed, widthPx, edgeColour, tarmacColour) {
    tracePath(points, toCanvas, closed);
    trackCtx.strokeStyle = edgeColour;
    trackCtx.lineWidth = widthPx;
    trackCtx.stroke();
    trackCtx.strokeStyle = tarmacColour;
    trackCtx.lineWidth = Math.max(1, widthPx - 4);
    trackCtx.stroke();
}

function drawTrackMap(points, carX, carZ, brakingZones = [], ersZones = [], forceFit = false) {
    if (!trackCtx || !trackCanvas) return;
    const width = trackCanvas.width;
    const height = trackCanvas.height;
    const closed = points === mapState.outline && points.length >= 2;

    // Clear background
    trackCtx.fillStyle = '#05070b';
    trackCtx.fillRect(0, 0, width, height);

    if (!points || points.length < 2) {
        trackCtx.fillStyle = '#64748b';
        trackCtx.font = "bold 13px 'Barlow Condensed', sans-serif";
        trackCtx.textAlign = 'center';
        trackCtx.fillText('MAPPING CIRCUIT... WAITING FOR CARS ON TRACK', width / 2, height / 2);
        return;
    }

    // Bounding box over the whole circuit and pit lane, so the full track always fits
    // the panel and the map does not jump around.
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    const pitLane = mapState.pitLane || [];
    for (const p of closed ? points.concat(pitLane) : points) {
        if (!p) continue;
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minZ) minZ = p[1];
        if (p[1] > maxZ) maxZ = p[1];
    }

    const rangeX = (maxX - minX) || 1;
    const rangeZ = (maxZ - minZ) || 1;
    const padding = 34;

    const bottomPadding = 56;           // room for the X/Z and G-force pills
    const scaleX = (width - padding * 2) / rangeX;
    const scaleZ = (height - padding - bottomPadding) / rangeZ;
    const scale = Math.min(scaleX, scaleZ);

    const offsetX = (width - rangeX * scale) / 2;
    const offsetZ = bottomPadding + (height - padding - bottomPadding - rangeZ * scale) / 2;

    const toCanvas = (x, z) => [
        offsetX + (x - minX) * scale,
        height - (offsetZ + (z - minZ) * scale) // Flip Z for correct orientation
    ];

    // A real track is about 14 m wide; keep it wide enough to read and hold the car dots.
    const roadPx = Math.max(12, Math.min(22, 14 * scale));
    const carRadius = Math.max(3, roadPx * 0.3);

    trackCtx.lineCap = 'round';
    trackCtx.lineJoin = 'round';

    // 1. Pit lane: a narrower road with amber edges, under the circuit
    if (closed && pitLane.length >= 2) {
        strokeRoad(pitLane, toCanvas, false, roadPx * 0.7, 'rgba(255, 176, 0, 0.8)', '#11151d');
    }

    // 2. Circuit: wide road with white edges
    strokeRoad(points, toCanvas, closed, roadPx, '#c9d1dc', '#141a24');

    // 3. This lap's driven line inside the road (a flashback cuts it back)
    if (closed && mapState.trail.length >= 2) {
        trackCtx.strokeStyle = 'rgba(0, 210, 190, 0.45)';
        trackCtx.lineWidth = 2;
        tracePath(mapState.trail, toCanvas, false);
        trackCtx.stroke();
    }

    // 4. ERS deployment and braking zones, as small marks inside the road
    for (const [zones, colour] of [[ersZones, '#00d2be'], [brakingZones, '#e10600']]) {
        if (!zones || !zones.length) continue;
        trackCtx.fillStyle = colour;
        for (const pt of zones) {
            const [zx, zz] = toCanvas(pt[0], pt[1]);
            trackCtx.beginPath();
            trackCtx.arc(zx, zz, 1.8, 0, Math.PI * 2);
            trackCtx.fill();
        }
    }

    // 5. Start / finish line: a white bar across the road
    const first = points.find(p => p);
    const next = points.slice(1).find(p => p) || first;
    const [sfx, sfz] = toCanvas(first[0], first[1]);
    const [nx, nz] = toCanvas(next[0], next[1]);
    const along = Math.atan2(nz - sfz, nx - sfx);
    const half = roadPx / 2 + 2;
    trackCtx.strokeStyle = '#ffffff';
    trackCtx.lineWidth = 3;
    trackCtx.beginPath();
    trackCtx.moveTo(sfx - Math.sin(along) * half, sfz + Math.cos(along) * half);
    trackCtx.lineTo(sfx + Math.sin(along) * half, sfz - Math.cos(along) * half);
    trackCtx.stroke();

    // 6. Every car, in its team colour; the player on top with a white ring
    const cars = mapState.cars || [];
    let player = null;
    for (const car of cars) {
        if (car[4]) { player = car; continue; }
        const [cx, cz] = toCanvas(car[0], car[1]);
        trackCtx.fillStyle = TEAM_COLOURS[car[2]] || '#9ca3af';
        trackCtx.beginPath();
        trackCtx.arc(cx, cz, carRadius, 0, Math.PI * 2);
        trackCtx.fill();
    }
    if (!player && (carX || carZ)) player = [carX, carZ, -1, 0, 1, mapState.inPit ? 1 : 0];
    if (player) {
        const [px, pz] = toCanvas(player[0], player[1]);
        const carColour = player[5] ? '#ffb000' : (TEAM_COLOURS[player[2]] || '#ffffff');
        trackCtx.strokeStyle = '#ffffff';
        trackCtx.lineWidth = 2;
        trackCtx.beginPath();
        trackCtx.arc(px, pz, carRadius + 4, 0, Math.PI * 2);
        trackCtx.stroke();
        trackCtx.fillStyle = carColour;
        trackCtx.shadowColor = '#ffffff';
        trackCtx.shadowBlur = 8;
        trackCtx.beginPath();
        trackCtx.arc(px, pz, carRadius + 1, 0, Math.PI * 2);
        trackCtx.fill();
        trackCtx.shadowBlur = 0;
    }

    // 8. Status: stored map or still mapping, and a flashback badge
    trackCtx.font = "bold 12px 'Barlow Condensed', sans-serif";
    trackCtx.textAlign = 'left';
    trackCtx.fillStyle = closed ? '#00d2be' : '#64748b';
    trackCtx.fillText(closed ? 'TRACK MAP' : 'MAPPING CIRCUIT...', 12, 20);
    if (Date.now() < mapState.flashbackUntil) {
        trackCtx.textAlign = 'right';
        trackCtx.fillStyle = '#ffb000';
        trackCtx.fillText('⟲ FLASHBACK', width - 12, 20);
    }
}

// ──────────────────────────────────────────────
// 4. Speed Trace Chart (Chart.js)
// ──────────────────────────────────────────────
function initSpeedChart() {
    const ctx = document.getElementById('speedChart');
    if (!ctx || typeof Chart === 'undefined') return;   // never let a missing chart library kill the HUD
    speedChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Session Best Lap',
                    data: [],
                    borderColor: '#b138dd',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Current Lap',
                    data: [],
                    borderColor: '#00d2be',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Distance (m)', color: '#64748b' },
                    grid: { color: 'rgba(39, 49, 66, 0.4)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    title: { display: true, text: 'Speed (km/h)', color: '#64748b' },
                    grid: { color: 'rgba(39, 49, 66, 0.4)' },
                    ticks: { color: '#64748b' },
                    min: 0,
                    max: 360
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    });
}

// ──────────────────────────────────────────────
// 5. Setup Advisor Engine (Feature 5)
// ──────────────────────────────────────────────
async function initSetupAdvisor() {
    const trackSelect = document.getElementById('trackSelect');
    const weatherSelect = document.getElementById('weatherSelect');
    const balanceSelect = document.getElementById('balanceSelect');
    const distanceSelect = document.getElementById('distanceSelect');
    const sessionSelect = document.getElementById('sessionSelect');
    const btnConsult = document.getElementById('btnConsultAI');

    if (!trackSelect) return;

    // Load 24 tracks from backend
    try {
        const res = await fetch('/api/tracks');
        if (res.ok) {
            const tracks = await res.json();
            trackSelect.innerHTML = '';
            tracks.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t.id;
                opt.textContent = `${t.name} (${t.country})`;
                trackSelect.appendChild(opt);
            });
            // Follow the game if it has already told us which circuit this is.
            if (currentTrackId !== null && [...trackSelect.options].some(o => o.value === String(currentTrackId))) {
                trackSelect.value = currentTrackId;
            }
        }
    } catch (e) {
        console.warn('Could not load tracks list from API:', e);
    }

    const reload = () => loadTrackSetup(trackSelect.value, weatherSelect.value);
    [trackSelect, weatherSelect, balanceSelect, distanceSelect, sessionSelect].forEach(el => el?.addEventListener('change', reload));
    // Race distance means nothing for a one-lap qualifying setup.
    const syncDistance = () => { if (distanceSelect) distanceSelect.disabled = sessionSelect?.value === 'qualifying'; };
    sessionSelect?.addEventListener('change', syncDistance);
    syncDistance();

    btnConsult?.addEventListener('click', async () => {
        btnConsult.disabled = true;
        btnConsult.innerHTML = '<span>⏳</span> DEEPSEEK ANALYSING...';
        showLatencyCue(true);
        try {
            await requestTrackAnalysis(true);
        } finally {
            btnConsult.disabled = false;
            btnConsult.innerHTML = '<span class="btn-icon">🧠</span> RE-ANALYSE WITH DEEPSEEK';
            showLatencyCue(false);
        }
    });

    // Initial load
    reload();
}

// Current Track Database selection; responses for an older selection are ignored.
function analysisParams(trackId, weather) {
    const val = (id, fallback) => document.getElementById(id)?.value || fallback;
    return {
        track_id: parseInt(trackId ?? val('trackSelect', '7'), 10),
        weather: weather ?? val('weatherSelect', 'dry'),
        handling_issue: val('balanceSelect', 'balanced'),
        race_fraction: parseFloat(val('distanceSelect', '1')),
        session: val('sessionSelect', 'race'),
    };
}

function sameParams(a, b) {
    return a.track_id === b.track_id && a.weather === b.weather &&
        a.handling_issue === b.handling_issue && a.race_fraction === b.race_fraction &&
        a.session === b.session;
}

async function loadTrackSetup(trackId, weather) {
    const params = analysisParams(trackId, weather);
    try {
        const qs = new URLSearchParams(params).toString();
        const res = await fetch(`/api/setup?${qs}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!sameParams(params, analysisParams())) return;
        renderSetupData(data);
        // Instant data-model numbers are on screen; now let DeepSeek analyse the track.
        if (data.analysis_source !== 'deepseek' && data.ai_available) {
            requestTrackAnalysis(false);
        }
    } catch (e) {
        console.error('Error fetching track setup:', e);
    }
}

async function requestTrackAnalysis(force) {
    const params = analysisParams();
    setAnalysisSource('busy', 'DeepSeek analysing…');
    try {
        const res = await fetch(force ? '/api/setup/consult' : '/api/track/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!res.ok) return;
        const data = await res.json();
        if (sameParams(params, analysisParams())) renderSetupData(data);
    } catch (err) {
        console.error('Track analysis error:', err);
        setAnalysisSource('model', 'Data model (DeepSeek unreachable)');
    }
}

function setAnalysisSource(kind, text, title) {
    const pill = document.getElementById('analysisSource');
    if (!pill) return;
    pill.className = `t-pill source-pill ${kind}`;
    pill.textContent = text;
    pill.title = title || '';
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text ?? '';
}

function stintBar(container, plan) {
    if (!container) return;
    container.innerHTML = '';
    if (!plan || !plan.stints) {
        container.textContent = '—';
        return;
    }
    const total = plan.stints.reduce((sum, s) => sum + s.laps, 0) || 1;
    plan.stints.forEach(s => {
        const seg = document.createElement('div');
        seg.className = `stint c-${s.compound}`;
        seg.style.flex = String(s.laps / total);
        seg.textContent = `${s.compound[0]} ${s.laps}`;
        seg.title = `${s.compound} — ${s.laps} laps`;
        container.appendChild(seg);
    });
    container.title = `${plan.stops}-stop, ~${plan.time_loss_s}s lost to tyres and stops`;
}

const DEMAND_LABELS = {
    abrasion: 'Abrasion', traction: 'Traction', braking: 'Braking',
    lateral: 'Lateral load', tyre_stress: 'Tyre stress', downforce: 'Downforce',
};

function renderSetupData(data) {
    const s = data.setup;
    if (!s) return;
    const wet = data.weather === 'wet';
    const quali = data.session === 'qualifying';

    // Header banner
    setText('setupTrackTitle', `${data.track_name} — ${data.country}`);
    setText('setupCharacteristics', data.characteristics || '');
    setText('setupDownforce', `Downforce: ${data.downforce_level}`);
    if (data.length_m) setText('setupLength', `${Number(data.length_m).toLocaleString()} m`);
    if (data.turns) setText('setupTurns', `${data.turns} Turns`);
    setText('setupTyreWear', `Tyre Degradation: ${data.tyre_degradation || '—'}`);
    setText('setupConditions', `${wet ? 'WET' : 'DRY'}${quali ? ' · QUALIFYING' : ''}`);
    if (data.analysis_source === 'deepseek') {
        setAnalysisSource('ai', 'DeepSeek analysis');
    } else if (data.ai_error) {
        setAnalysisSource('model', 'Data model (DeepSeek offline)', data.ai_error);
    } else {
        setAnalysisSource('model', 'Data model');
    }

    // A. Tyre degradation
    const level = data.tyre_degradation || '—';
    const badge = document.getElementById('degBadge');
    if (badge) {
        badge.textContent = level.toUpperCase();
        badge.className = `deg-badge deg-${level.toLowerCase().replace(/\s+/g, '-')}`;
    }
    const deg = data.degradation || {};
    const body = document.getElementById('degTableBody');
    if (body) {
        body.innerHTML = '';
        Object.entries(deg.rates_s_per_lap || {}).forEach(([compound, rate]) => {
            const tr = document.createElement('tr');
            const name = document.createElement('td');
            const dot = document.createElement('span');
            dot.className = `tyre-dot c-${compound}`;
            name.appendChild(dot);
            name.appendChild(document.createTextNode(compound.charAt(0) + compound.slice(1).toLowerCase()));
            const r = document.createElement('td');
            r.textContent = `+${Number(rate).toFixed(3)} s`;
            const life = document.createElement('td');
            life.textContent = `${(deg.life_laps || {})[compound] ?? '—'} laps`;
            tr.append(name, r, life);
            body.appendChild(tr);
        });
    }
    const limiting = deg.limiting_tyre || (deg.limiting_axle ? `${deg.limiting_axle} axle` : '');
    setText('degReasoning', `${deg.reasoning || ''}${limiting ? ` Limiting: ${limiting}.` : ''}`);

    // B. Strategy (race) or run plan (qualifying)
    const strat = data.strategy || {};
    const plan = data.qualifying || {};
    setText('strategyTitle', quali ? 'QUALIFYING PLAN' : 'RACE STRATEGY');
    document.querySelectorAll('.strategy-row').forEach(row => { row.style.display = quali ? 'none' : ''; });
    const planEl = document.getElementById('qualiPlan');
    if (planEl) planEl.style.display = quali ? '' : 'none';
    let ai = strat.ai || {};
    if (quali) {
        const qai = plan.ai || {};
        const tyre = qai.tyre || plan.tyre || '—';
        setText('strategyLaps', 'ONE LAP');
        setText('qualiTyre', tyre.charAt(0).toUpperCase() + tyre.slice(1).toLowerCase());
        setText('qualiPushLaps', qai.push_laps_per_set ?? plan.push_laps_per_set ?? '—');
        setText('qualiOutLap', qai.out_lap || plan.out_lap || '');
        setText('qualiTiming', plan.session_timing || '');
        setText('strategyAi', qai.reasoning ? `DeepSeek: ${qai.reasoning}` : '');
    } else {
        setText('strategyLaps', strat.race_laps ? `${strat.race_laps} LAPS` : '—');
        stintBar(document.getElementById('strategyBest'), strat.best);
        stintBar(document.getElementById('strategyAlt'), strat.alternative);
        setText('strategyAi', ai.plan ? `DeepSeek: ${ai.plan}${ai.reasoning ? ' — ' + ai.reasoning : ''}` : '');
    }
    const notes = document.getElementById('strategyNotes');
    if (notes) {
        notes.innerHTML = '';
        ((quali ? plan.notes : strat.notes) || []).forEach(n => {
            const li = document.createElement('li');
            li.textContent = n;
            notes.appendChild(li);
        });
    }

    // C. Track demands and real-race facts
    const bars = document.getElementById('demandBars');
    if (bars) {
        bars.innerHTML = '';
        Object.entries(DEMAND_LABELS).forEach(([key, label]) => {
            const value = (data.demands || {})[key];
            if (value === undefined) return;
            const row = document.createElement('div');
            row.className = 'demand-row';
            row.innerHTML = '<span class="p-name"></span><div class="demand-track"><div class="demand-fill"></div></div><span class="demand-val"></span>';
            row.children[0].textContent = label;
            row.querySelector('.demand-fill').style.width = `${value * 20}%`;
            row.children[2].textContent = `${value}/5`;
            bars.appendChild(row);
        });
    }
    const facts = data.race_facts || {};
    const factsEl = document.getElementById('raceFacts');
    if (factsEl) {
        factsEl.innerHTML = '';
        [
            ['Pit loss', `${facts.pit_loss_s ?? '—'} s`],
            ['Safety Car', `${Math.round((facts.sc_probability ?? 0) * 100)}% of races`],
            ['Rain', `${Math.round((facts.rain_probability ?? 0) * 100)}% of races`],
            ['Overtaking', `${facts.overtaking ?? '—'}/5 hard`],
        ].forEach(([k, v]) => {
            const span = document.createElement('span');
            span.textContent = `${k}: `;
            const b = document.createElement('b');
            b.textContent = v;
            span.appendChild(b);
            factsEl.appendChild(span);
        });
    }
    setText('dataSource', data.calibration && data.calibration.length
        ? `Calibrated from ${data.calibration.length} real races (FastF1): ${data.calibration.slice(-3).join(', ')}…`
        : 'Built from Pirelli circuit ratings and real race history. Run tools/build_real_race_calibration.py to calibrate from real race timing.');

    // Setup values
    setText('valFrontWing', s.aerodynamics?.front_wing ?? '—');
    setText('valRearWing', s.aerodynamics?.rear_wing ?? '—');
    setText('valDiffOn', (s.transmission?.diff_on_throttle ?? '—') + '%');
    setText('valDiffOff', (s.transmission?.diff_off_throttle ?? '—') + '%');
    const deg2 = v => (v === undefined ? '—' : Number(v).toFixed(2) + '°');
    setText('valFrontCamber', deg2(s.suspension_geometry?.front_camber));
    setText('valRearCamber', deg2(s.suspension_geometry?.rear_camber));
    setText('valFrontToe', deg2(s.suspension_geometry?.front_toe));
    setText('valRearToe', deg2(s.suspension_geometry?.rear_toe));
    setText('valSuspension', `${s.suspension?.front_suspension ?? '—'} / ${s.suspension?.rear_suspension ?? '—'}`);
    setText('valArb', `${s.suspension?.front_anti_roll ?? '—'} / ${s.suspension?.rear_anti_roll ?? '—'}`);
    setText('valRideHeight', `${s.suspension?.front_ride_height ?? '—'} / ${s.suspension?.rear_ride_height ?? '—'}`);
    setText('valBrakePressure', (s.brakes?.brake_pressure ?? '—') + '%');
    setText('valBrakeBias', (s.brakes?.brake_bias ?? '—') + '%');
    const psi = v => (v === undefined ? '—' : Number(v).toFixed(1) + ' PSI');
    setText('valFrontRightPsi', psi(s.tyres?.front_right_pressure));
    setText('valFrontLeftPsi', psi(s.tyres?.front_left_pressure));
    setText('valRearPsi', psi(s.tyres?.rear_left_pressure));

    // Per-category reasoning (DeepSeek), with conditions-aware defaults
    const tips = data.setup_notes || {};
    const dflt = quali ? {
        aerodynamics: wet ? 'Quali (wet): a touch more front wing for turn-in on one lap.'
            : `Quali: wings trimmed for one lap at ${String(data.downforce_level || '').toLowerCase()} downforce.`,
        transmission: wet ? 'Quali (wet): diff kept open for traction.' : 'Quali: on-throttle diff locked up for drive out of slow corners.',
        suspension_geometry: wet ? '' : 'Quali: maximum useful camber and sharper front toe; wear does not matter for one lap.',
        suspension: wet ? 'Quali (wet): car kept soft and high over standing water.' : 'Quali: stiffer and lower; low fuel lets the car ride closer to the ground.',
        brakes: wet ? 'Quali (wet): gentle brake pressure, bias rearward against lock-ups.' : 'Quali: full brake pressure, bias a click forward for the lighter car.',
        tyres: wet ? '' : 'Quali: lower pressures for a bigger contact patch on one lap.',
    } : {
        aerodynamics: wet ? 'Wet: extra wing for grip on a low-grip surface.' : `Aero matched to ${String(data.downforce_level || '').toLowerCase()} downforce demand.`,
        transmission: wet ? 'Wet: more open on-throttle diff to stop wheelspin.' : 'Higher off-throttle helps trail-braking stability.',
        suspension_geometry: '',
        suspension: wet ? 'Wet: softer springs and bars, higher ride over standing water.' : '',
        brakes: wet ? 'Wet: lower pressure and bias rearward to avoid front lock-ups.' : 'Slight forward bias prevents rear lockups into heavy chicanes.',
        tyres: '',
    };
    setText('tipAero', tips.aerodynamics || dflt.aerodynamics);
    setText('tipTransmission', tips.transmission || dflt.transmission);
    setText('tipGeometry', tips.suspension_geometry || dflt.suspension_geometry);
    setText('tipSuspension', tips.suspension || dflt.suspension);
    setText('tipBrakes', tips.brakes || dflt.brakes);
    setText('tipTyres', tips.tyres || dflt.tyres);

    // Briefing and key corners
    setText('setupBriefingText', data.engineer_advice || data.ai_consultation || s.notes || 'Setup loaded.');
    const corners = document.getElementById('keyCorners');
    if (corners) {
        corners.innerHTML = '';
        (data.key_corners || []).forEach(c => {
            const li = document.createElement('li');
            li.textContent = c;
            corners.appendChild(li);
        });
    }
}

// ──────────────────────────────────────────────
// 6. Voice Commands & Latency Audio Cue
// ──────────────────────────────────────────────
function initVoiceCommands() {
    const chips = document.querySelectorAll('.cmd-chip');
    chips.forEach(btn => {
        btn.addEventListener('click', () => {
            transmitCommand(btn.dataset.cmd);
        });
    });

    const sendBtn = document.getElementById('btnSendCmd');
    const input = document.getElementById('customCmdInput');
    sendBtn?.addEventListener('click', () => {
        if (input && input.value.trim()) {
            transmitCommand(input.value.trim());
            input.value = '';
        }
    });

    input?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
            transmitCommand(input.value.trim());
            input.value = '';
        }
    });
}

function showLatencyCue(active) {
    const badge = document.getElementById('checkingBadge');
    if (badge) {
        if (active) badge.classList.add('active');
        else badge.classList.remove('active');
    }
}

async function transmitCommand(text) {
    showLatencyCue(true);
    try {
        await fetch('/api/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
    } catch (err) {
        console.error('Radio transmit error:', err);
    } finally {
        setTimeout(() => showLatencyCue(false), 2200);
    }
}

// ──────────────────────────────────────────────
// 7. Telemetry Polling & Dashboard Loop
// ──────────────────────────────────────────────
function fmtTime(ms) {
    if (!ms || ms <= 0) return '0:00.000';
    const totalSecs = ms / 1000;
    const mins = Math.floor(totalSecs / 60);
    const secs = (totalSecs % 60).toFixed(3);
    return `${mins}:${secs.padStart(6, '0')}`;
}

function fmtDelta(deltaMs) {
    if (deltaMs === null || deltaMs === undefined) return '---';
    const sign = deltaMs > 0 ? '+' : '-';
    return `${sign}${(Math.abs(deltaMs) / 1000).toFixed(3)}s`;
}

let lastSetupSessionMode = null;

// ──────────────────────────────────────────────
// Live feed: WebSocket push, with polling only as a fallback
// ──────────────────────────────────────────────
// The server sends the whole state once, then only the keys that changed. They are merged
// into hudState and drawn at most once per animation frame, and only the panels whose keys
// changed are rebuilt.
const hudState = {};
let pendingChanges = null;
let renderQueued = false;
let pollTimer = null;
let socketRetryMs = 1000;

function startLiveFeed() {
    if (typeof WebSocket === 'undefined') { startPolling(); return; }
    connectSocket();
}

function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(fetchTelemetry, 150);
    fetchTelemetry();
}

function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
}

function connectSocket() {
    let socket;
    try {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${proto}//${location.host}/ws`);
    } catch (err) {
        startPolling();
        return;
    }
    socket.onopen = () => {
        socketRetryMs = 1000;
        stopPolling();
        const beacon = document.getElementById('statusBeacon');
        if (beacon) beacon.title = 'Live push (WebSocket)';
    };
    socket.onmessage = event => {
        try { applyLiveMessage(JSON.parse(event.data)); } catch (err) { console.error('[HUD] bad message', err); }
    };
    socket.onclose = () => {
        const beacon = document.getElementById('statusBeacon');
        if (beacon) beacon.title = 'Polling (WebSocket reconnecting)';
        startPolling();
        setTimeout(connectSocket, socketRetryMs);
        socketRetryMs = Math.min(socketRetryMs * 2, 10000);
    };
    socket.onerror = () => socket.close();
}

function applyLiveMessage(msg) {
    const state = msg.state || {};
    const status = msg.status || {};
    if (!pendingChanges) pendingChanges = new Set();
    if (msg.type === 'full') {
        Object.keys(hudState).forEach(k => delete hudState[k]);
        pendingChanges.add('*');
    }
    Object.assign(hudState, state, status);
    Object.keys(state).forEach(k => pendingChanges.add(k));
    Object.keys(status).forEach(k => pendingChanges.add(k));
    if (!renderQueued) {
        renderQueued = true;
        (window.requestAnimationFrame || (fn => setTimeout(fn, 16)))(flushRender);
    }
}

function flushRender() {
    renderQueued = false;
    const changed = pendingChanges;
    pendingChanges = null;
    if (!changed) return;
    renderTelemetry(hudState, changed.has('*') ? null : changed);
}

function showDisconnected() {
    const beacon = document.getElementById('statusBeacon');
    const statusText = document.getElementById('statusText');
    if (beacon) beacon.classList.remove('connected');
    if (statusText) statusText.innerText = 'DISCONNECTED';
}

async function fetchTelemetry() {
    let data;
    try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        data = await res.json();
    } catch (err) {
        showDisconnected();
        return;
    }
    renderTelemetry(data, null);
}

const TRACK_STATUS_CLASSES = {
    RED: 'sc-red', SC: 'sc-full', VSC: 'sc-vsc', FORMATION: 'sc-vsc',
    YELLOW: 'sc-yellow', BLUE: 'sc-blue', CLEAR: 'sc-none',
};

// The track status to show; older servers only send the safety_car string.
function trackStatus(data) {
    const status = data.track_status;
    if (status && TRACK_STATUS_CLASSES[status.state]) return status;
    if (data.safety_car === 'SAFETY CAR') return { state: 'SC', label: 'SAFETY CAR DEPLOYED' };
    if (data.safety_car === 'VSC') return { state: 'VSC', label: 'VIRTUAL SAFETY CAR' };
    return { state: 'CLEAR', label: 'TRACK CLEAR' };
}

// One sector box: this lap's time if driven, otherwise the last lap's (previous: true).
function sectorSplit(data, s) {
    const current = (data.current_sectors_ms || {})[s];
    if (current && current > 0) {
        return { timeMs: current, bestMs: (data.personal_sectors_ms || {})[s], previous: false };
    }
    const last = (data.last_lap_sectors_ms || {})[s];
    if (last && last > 0) {
        return { timeMs: last, bestMs: (data.last_lap_sector_refs_ms || {})[s], previous: true };
    }
    return null;
}

// Draw the HUD from a state snapshot. changed: the keys that changed since the last draw
// (null = draw everything).
function renderTelemetry(data, changed) {
    const dirty = (...keys) => !changed || keys.some(k => changed.has(k));
    try {
        // 1. Connection Beacon
        const beacon = document.getElementById('statusBeacon');
        const statusText = document.getElementById('statusText');
        if (data.connected) {
            beacon?.classList.add('connected');
            if (statusText) statusText.innerText = 'LIVE TELEMETRY';
        } else {
            beacon?.classList.remove('connected');
            if (statusText) statusText.innerText = 'WAITING FOR UDP';
        }

        // 2. Track & Circuit Metadata
        if (data.track_name) {
            const trackEl = document.getElementById('trackNameDisplay');
            if (trackEl) trackEl.textContent = data.track_name.toUpperCase();
        }
        if (data.track_length) {
            const lenEl = document.getElementById('trackLengthDisplay');
            if (lenEl) lenEl.textContent = `${data.track_length.toLocaleString()} m`;
        }
        // Short layouts (e.g. Silverstone Short) report their parent circuit's setup data.
        const liveTrack = data.setup_track_id !== undefined ? data.setup_track_id : data.track_id;
        if (liveTrack !== undefined && liveTrack >= 0 && liveTrack !== currentTrackId) {
            currentTrackId = liveTrack;
            const sel = document.getElementById('trackSelect');
            // Only follow the game onto circuits that are in the database (the API would
            // otherwise answer with its Silverstone default under the wrong heading).
            if (sel && [...sel.options].some(o => o.value === String(currentTrackId))) {
                sel.value = currentTrackId;
                const weather = document.getElementById('weatherSelect');
                loadTrackSetup(currentTrackId, weather ? weather.value : 'dry');
            }
        }

        // 2b. Session type: Free Practice / Qualifying / Race / Time Trial
        const mode = data.session_mode || 'unknown';
        // Follow the game into qualifying or the race once per session change, then leave
        // the Track Database selector to the driver.
        if ((mode === 'qualifying' || mode === 'race') && mode !== lastSetupSessionMode) {
            lastSetupSessionMode = mode;
            const sessionSel = document.getElementById('sessionSelect');
            if (sessionSel && sessionSel.value !== mode) {
                sessionSel.value = mode;
                sessionSel.dispatchEvent(new Event('change'));
            }
        }
        const badge = document.getElementById('sessionBadge');
        if (badge) {
            badge.textContent = (data.session_label || '--').toUpperCase();
            badge.dataset.mode = mode;
        }
        document.body.dataset.sessionMode = mode;
        document.querySelectorAll('[data-modes]').forEach(el => {
            const visible = mode === 'unknown' || el.dataset.modes.split(' ').includes(mode);
            el.style.display = visible ? '' : 'none';
        });

        // 3. Track status pill: red flag, SC, VSC, yellow, blue or clear
        const scPill = document.getElementById('safetyCarPill');
        const scText = document.getElementById('safetyCarText');
        if (scPill && scText) {
            const status = trackStatus(data);
            scPill.className = 'safety-car-pill ' + TRACK_STATUS_CLASSES[status.state];
            scText.textContent = status.label;
        }

        // 4. Timing Tower
        const posBadge = document.getElementById('posBadge');
        if (posBadge) posBadge.textContent = `P${data.position || 1}`;

        const lapCounter = document.getElementById('lapCounter');
        if (lapCounter) {
            const total = data.total_laps ? ` / ${data.total_laps}` : '';
            lapCounter.textContent = `LAP ${data.lap_num || 1}${total}`;
        }

        // Live Delta
        const deltaDisplay = document.getElementById('liveDeltaDisplay');
        if (deltaDisplay) {
            deltaDisplay.textContent = fmtDelta(data.delta_ms);
            deltaDisplay.className = 'row-value delta-val';
            if (data.delta_ms !== null && data.delta_ms !== undefined) {
                deltaDisplay.classList.add(data.delta_ms <= 0 ? 'delta-ahead' : 'delta-behind');
            }
        }

        const curLapEl = document.getElementById('currentLapTimeDisplay');
        if (curLapEl) curLapEl.textContent = fmtTime(data.lap_time_ms);

        const lastLapEl = document.getElementById('lastLapTimeDisplay');
        if (lastLapEl) lastLapEl.textContent = fmtTime(data.last_lap_time_ms);

        const bestLapEl = document.getElementById('bestLapTimeDisplay');
        if (bestLapEl) bestLapEl.textContent = fmtTime(data.best_lap_time_ms);

        const paceEl = document.getElementById('racePaceDisplay');
        if (paceEl) paceEl.textContent = data.race_pace_ms ? fmtTime(data.race_pace_ms) : '--:--.---';

        // Gaps
        const gapAhead = document.getElementById('gapAheadDisplay');
        if (gapAhead) {
            gapAhead.textContent = data.gap_ahead_s !== null && data.gap_ahead_s !== undefined
                ? `${(data.car_ahead_name || 'INTERVAL').toUpperCase()} +${data.gap_ahead_s.toFixed(1)}s` : 'INTERVAL --';
        }
        const gapBehind = document.getElementById('gapBehindDisplay');
        if (gapBehind) {
            gapBehind.textContent = data.gap_behind_s !== null && data.gap_behind_s !== undefined
                ? `${(data.car_behind_name || 'INTERVAL').toUpperCase()} -${data.gap_behind_s.toFixed(1)}s` : 'INTERVAL --';
        }

        // Pit Window
        const pitEl = document.getElementById('pitWindowDisplay');
        if (pitEl) {
            if (data.pit_window) {
                pitEl.textContent = `Target Lap ${data.pit_window[0]} (Latest Lap ${data.pit_window[1]})`;
            } else {
                pitEl.textContent = 'Calculating strategy...';
            }
        }

        // Nearby opponent threats (physics-model gap/pace scoring)
        const threatsCard = document.getElementById('threatsCard');
        const threatsList = document.getElementById('threatsList');
        if (threatsCard && threatsList && dirty('opponent_threats')) {
            const threats = data.opponent_threats || [];
            threatsCard.style.display = threats.length ? '' : 'none';
            threatsList.innerHTML = threats.map(t => {
                const side = t.gap_s < 0 ? 'BEHIND' : 'AHEAD';
                // Names can be online gamertags, so escape them before they go into innerHTML.
                const who = t.driver_name ? `${escapeHtml(t.driver_name.toUpperCase())} ` : '';
                const cls = t.threat_level === 'HIGH' ? 'threat-high'
                          : t.threat_level === 'MEDIUM' ? 'threat-medium' : 'threat-low';
                const note = t.factors && t.factors.length ? ` — ${t.factors[0]}` : '';
                return `<div class="gap-item ${cls}">
                    <span class="gap-label">${who}${side} ${Math.abs(t.gap_s).toFixed(1)}s</span>
                    <span class="gap-val">${t.threat_level}${note}</span>
                </div>`;
            }).join('');
        }

        // Race strategy (stop count, compounds, undercut on the car ahead, last call)
        const strategyCard = document.getElementById('strategyCard');
        const strategyList = document.getElementById('strategyList');
        if (strategyCard && strategyList && dirty('race_strategy')) {
            const strat = data.race_strategy;
            strategyCard.style.display = strat ? '' : 'none';
            if (strat) {
                const best = strat.best;
                const sets = best.compounds.map((c, i) => `${c[0]}${best.stint_laps[i]}`).join(' → ');
                const box = best.pit_laps.length ? `BOX L${best.pit_laps.join(', L')}` : 'NO STOP';
                const rows = [];
                // Live re-planning: a fresh Plan B call on top, then the plan we are on
                const rp = strat.replan;
                if (rp && rp.call_active && rp.last_call) {
                    rows.push(`<div class="gap-item threat-high plan-b-call">
                        <span class="gap-label">PLAN B · L${rp.last_call.lap}</span>
                        <span class="gap-val">${rp.last_call.message}</span>
                    </div>`);
                }
                // Track wetness: how wet, which way it is going, and when to change tyre type
                const cond = strat.conditions;
                if (cond) {
                    const words = { slick: 'SLICKS', inter: 'INTERS', wet: 'WETS' };
                    const sw = cond.switch_to
                        ? (cond.switch_in_laps === 0 ? `${words[cond.switch_to]} NOW` : `${words[cond.switch_to]} ~${cond.switch_in_laps} LAPS`)
                        : `${words[cond.best_tyre]} BEST`;
                    rows.push(`<div class="gap-item ${cond.switch_to && cond.switch_in_laps === 0 ? 'threat-medium' : 'threat-low'}">
                        <span class="gap-label">TRACK ${Math.round(cond.wetness * 100)}% WET · ${cond.trend.toUpperCase()}</span>
                        <span class="gap-val">${sw}</span>
                    </div>`);
                }
                if (rp && rp.plan_a) {
                    const a = rp.plan_a;
                    const aBox = a.pit_laps.length ? `BOX L${a.pit_laps.join(', L')}` : 'NO STOP';
                    const ch = rp.challenger;
                    const watch = ch && ch.clears_margin && rp.status !== 'switched'
                        ? ` · B? ${ch.label.toUpperCase()} +${ch.time_gain_s.toFixed(1)}s` : '';
                    rows.push(`<div class="gap-item">
                        <span class="gap-label">PLAN A ${a.label.toUpperCase()}${watch}</span>
                        <span class="gap-val">${aBox}</span>
                    </div>`);
                }
                rows.push(`<div class="gap-item${rp && rp.plan_a ? ' threat-low' : ''}">
                    <span class="gap-label">${rp && rp.plan_a ? 'MODEL ' : ''}${best.stops}-STOP ${sets}</span>
                    <span class="gap-val">${box}</span>
                </div>`);
                (strat.alternatives || []).forEach(o => rows.push(`<div class="gap-item threat-low">
                    <span class="gap-label">${o.stops}-STOP</span>
                    <span class="gap-val">+${o.slower_by_s.toFixed(1)}s</span>
                </div>`));
                // Monte Carlo race simulation: how often each plan comes out best
                const sim = strat.simulation;
                if (sim && sim.options && sim.options.length) {
                    sim.options.forEach((o, i) => {
                        const r = o.position_range;
                        rows.push(`<div class="gap-item ${i === 0 ? 'threat-high' : 'threat-low'}">
                            <span class="gap-label">SIM ${o.is_reference ? 'A ' : ''}${o.label.toUpperCase()} P${r[1]} (P${r[0]}–P${r[2]})</span>
                            <span class="gap-val">${Math.round(o.p_best * 100)}%</span>
                        </div>`);
                    });
                    rows.push(`<div class="gap-item threat-low">
                        <span class="gap-label">SIM ${sim.runs} RACES · SC ${Math.round(sim.p_safety_car * 100)}% · VSC ${Math.round(sim.p_vsc * 100)}%</span>
                        <span class="gap-val">PIT ${sim.pit_loss_s[1].toFixed(1)}s</span>
                    </div>`);
                }
                const uc = strat.undercut_on_car_ahead;
                if (uc) {
                    rows.push(`<div class="gap-item ${uc.works ? 'threat-high' : 'threat-low'}">
                        <span class="gap-label">UNDERCUT ${uc.gain_s.toFixed(1)}s vs GAP ${uc.gap_s.toFixed(1)}s</span>
                        <span class="gap-val">${uc.works ? 'ON' : 'OFF'}</span>
                    </div>`);
                }
                // Rival stop model: who is likely to box when, and onto what
                const rivals = (strat.rivals && strat.rivals.nearby) || [];
                rivals.forEach(r => {
                    const who = r.name ? `${r.name} ` : '';
                    const [soon, , late] = r.stop_lap_range || [];
                    const when = !r.will_stop ? 'NO STOP'
                        : `L${r.stop_lap}${late == null ? '+' : (late - soon >= 3 ? ` (${soon}-${late})` : '')}`;
                    const onto = r.will_stop && r.next_compound ? ` → ${r.next_compound[0]}` : '';
                    rows.push(`<div class="gap-item threat-low">
                        <span class="gap-label">P${r.position} ${who}${r.compound[0]}${r.tyre_age}</span>
                        <span class="gap-val">${when}${onto}</span>
                    </div>`);
                });
                if (strat.last_call) {
                    rows.push(`<div class="gap-item threat-medium">
                        <span class="gap-label">L${strat.last_call.lap ?? '-'} ${strat.last_call.kind.toUpperCase()}</span>
                        <span class="gap-val">${strat.last_call.message}</span>
                    </div>`);
                }
                strategyList.innerHTML = rows.join('');
            }
        }

        // Fitted tyre / fuel models: [low, expected, high] ranges
        const modelCard = document.getElementById('modelCard');
        const modelList = document.getElementById('modelList');
        if (modelCard && modelList && dirty('model_ranges')) {
            const m = data.model_ranges;
            modelCard.style.display = m ? '' : 'none';
            if (m) {
                const num = (v, d) => (v === null || v === undefined) ? '?' : Number(v).toFixed(d);
                const span = (r, d, unit) => r ? `${num(r[1], d)}${unit} (${num(r[0], d)}–${num(r[2], d)})` : '—';
                const laps = r => r ? `L${r[1] ?? '—'} (${r[0] ?? '—'}–${r[2] ?? 'flag'})` : '—';
                const basisNames = { track: 'fitted here', pooled: 'other tracks', prior: 'estimate' };
                const basis = document.getElementById('modelBasis');
                if (basis) {
                    basis.textContent = `${basisNames[m.basis] || m.basis}, ${m.fit_laps + m.stint_laps} laps`
                        + (m.simulated_fit ? ', sim' : '');
                }
                const rows = [
                    ['WEAR / LAP', span(m.wear_per_lap_pct, 2, '%')],
                    ['WEAR IN 5 LAPS', span(m.wear_in_5_laps_pct, 1, '%')],
                    ['FALL-OFF', span(m.deg_s_per_lap, 3, 's')],
                ];
                if (m.pit_ideal_lap) rows.push(['BOX', laps(m.pit_ideal_lap)]);
                if (m.pit_latest_lap) rows.push(['LATEST', laps(m.pit_latest_lap)]);
                if (m.finish_without_stop_prob !== null && m.finish_without_stop_prob !== undefined) {
                    rows.push(['REACH FLAG', `${Math.round(m.finish_without_stop_prob * 100)}%`]);
                }
                if (m.fuel_per_lap_kg) rows.push(['FUEL / LAP', span(m.fuel_per_lap_kg, 2, 'kg')]);
                if (m.fuel_margin_laps) rows.push(['FUEL MARGIN', span(m.fuel_margin_laps, 1, ' laps')]);
                modelList.innerHTML = rows.map(([label, value]) => `<div class="gap-item">
                    <span class="gap-label">${label}</span>
                    <span class="gap-val">${value}</span>
                </div>`).join('');
            }
        }

        // Race Control feed (newest first)
        const raceControlCard = document.getElementById('raceControlCard');
        const raceControlList = document.getElementById('raceControlList');
        if (raceControlCard && raceControlList && dirty('race_control')) {
            const events = data.race_control || [];
            raceControlCard.style.display = events.length ? '' : 'none';
            const escapeHtml = text => String(text).replace(/[&<>"']/g,
                ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
            raceControlList.innerHTML = events.slice(0, 6).map(e => {
                const cls = e.kind === 'penalty' || e.kind === 'flag' ? 'threat-high'
                          : e.kind === 'overtake' || e.kind === 'pit' ? 'threat-medium' : 'threat-low';
                const lap = e.lap ? `LAP ${e.lap}` : '';
                return `<div class="gap-item ${cls}">
                    <span class="gap-label">${lap}</span>
                    <span class="gap-val">${escapeHtml(e.text)}</span>
                </div>`;
            }).join('');
        }

        // Sector Splits (Purple/Green/Yellow). A sector not yet driven this lap shows the
        // last lap's time (dimmed), so S3 stays up after the line: the game never sends S3
        // live, it is worked out from the lap time when the lap ends.
        for (let s = 1; s <= 3; s++) {
            const box = document.getElementById(`sectorBox${s}`);
            const timeEl = document.getElementById(`secTime${s}`);
            const deltaEl = document.getElementById(`secDelta${s}`);
            if (!box || !timeEl) continue;
            const split = sectorSplit(data, s);

            box.className = 'sector-box';
            if (split) {
                if (split.previous) box.classList.add('sec-prev');
                timeEl.textContent = (split.timeMs / 1000).toFixed(3);
                if (split.bestMs) {
                    const diff = split.timeMs - split.bestMs;
                    if (deltaEl) deltaEl.textContent = (diff > 0 ? '+' : '') + (diff / 1000).toFixed(2);
                    if (diff <= -50) {
                        box.classList.add('sec-purple'); // Overall fastest session split
                    } else if (diff <= 50) {
                        box.classList.add('sec-green');  // Personal best
                    } else {
                        box.classList.add('sec-yellow'); // Slower
                    }
                } else {
                    box.classList.add('sec-green');
                    if (deltaEl) deltaEl.textContent = 'PB';
                }
            } else {
                timeEl.textContent = '--.---';
                if (deltaEl) deltaEl.textContent = '--';
            }
        }

        // 5. Tyre Thermal Heatmap & Degradation
        // Order: [RL, RR, FL, FR]
        const surfaceTemps = data.tyre_surface_temps || [100, 100, 100, 100];
        const innerTemps = data.tyre_inner_temps || [95, 95, 95, 95];
        const pressures = data.tyre_pressures || [23.5, 23.5, 22.0, 22.0];
        const wear = data.tyre_wear || [0, 0, 0, 0];

        const tyresConfig = [
            { id: 'RL', temp: surfaceTemps[0], inner: innerTemps[0], psi: pressures[0], wear: wear[0] },
            { id: 'RR', temp: surfaceTemps[1], inner: innerTemps[1], psi: pressures[1], wear: wear[1] },
            { id: 'FL', temp: surfaceTemps[2], inner: innerTemps[2], psi: pressures[2], wear: wear[2] },
            { id: 'FR', temp: surfaceTemps[3], inner: innerTemps[3], psi: pressures[3], wear: wear[3] }
        ];

        tyresConfig.forEach(t => {
            const card = document.getElementById(`tyreCard${t.id}`);
            const tempEl = document.getElementById(`temp${t.id}`);
            const innerEl = document.getElementById(`innerTemp${t.id}`);
            const wearEl = document.getElementById(`wear${t.id}`);
            const wearBar = document.getElementById(`wearBar${t.id}`);
            const psiEl = document.getElementById(`psi${t.id}`);

            if (tempEl) tempEl.textContent = `${t.temp}°C`;
            if (innerEl) innerEl.textContent = `Carcass: ${t.inner}°C`;
            if (wearEl) wearEl.textContent = `Wear: ${(t.wear || 0).toFixed(1)}%`;
            if (wearBar) wearBar.style.width = `${Math.min(100, t.wear || 0)}%`;
            if (psiEl) psiEl.textContent = `${(t.psi || 0).toFixed(1)} PSI`;

            if (card) {
                card.className = `tyre-card tyre-${t.id.toLowerCase()}`;
                if (t.temp < 85) card.classList.add('cold');
                else if (t.temp <= 104) card.classList.add('optimal');
                else if (t.temp <= 112) card.classList.add('warm');
                else card.classList.add('hot');
            }
        });

        // Tyre Compound Badge
        const compTag = document.getElementById('compoundTag');
        const towerComp = document.getElementById('towerCompoundBadge');
        const compoundName = data.tyre_compound || 'Medium';
        if (compTag) compTag.textContent = compoundName.toUpperCase();
        if (towerComp) {
            towerComp.textContent = compoundName[0].toUpperCase();
            towerComp.className = 'tyre-badge-mini';
            const cLower = compoundName.toLowerCase();
            if (cLower.includes('soft')) towerComp.classList.add('compound-soft');
            else if (cLower.includes('hard')) towerComp.classList.add('compound-hard');
            else if (cLower.includes('inter')) towerComp.classList.add('compound-inter');
            else if (cLower.includes('wet')) towerComp.classList.add('compound-wet');
            else towerComp.classList.add('compound-medium');
        }

        // Steer angle indicator in cockpit
        const steerGraphic = document.getElementById('steerAngleGraphic');
        if (steerGraphic && data.steer !== undefined) {
            steerGraphic.style.transform = `rotate(${data.steer * 45}deg)`;
        }

        // DRS Indicator
        const drsEl = document.getElementById('drsStatusDisplay');
        if (drsEl) {
            drsEl.textContent = data.drs ? 'DRS ACTIVE' : 'DRS OFF';
            drsEl.className = data.drs ? 'v-val drs-on' : 'v-val drs-off';
        }

        // 6. Broadcast Telemetry Bottom Ribbon
        const speedEl = document.getElementById('speedDisplay');
        if (speedEl) speedEl.textContent = data.speed || 0;

        const gearEl = document.getElementById('gearDisplay');
        if (gearEl) {
            const g = data.gear;
            gearEl.textContent = g === 0 ? 'N' : g < 0 ? 'R' : g;
        }

        const rpmEl = document.getElementById('rpmDisplay');
        if (rpmEl) rpmEl.textContent = `${(data.rpm || 0).toLocaleString()} RPM`;
        updateRevLeds(data.rpm || 0);

        const thrBar = document.getElementById('throttleBar');
        const thrTxt = document.getElementById('throttleText');
        const brkBar = document.getElementById('brakeBar');
        const brkTxt = document.getElementById('brakeText');

        const throttlePct = Math.round((data.throttle || 0) * 100);
        const brakePct = Math.round((data.brake || 0) * 100);

        if (thrBar) thrBar.style.width = `${throttlePct}%`;
        if (thrTxt) thrTxt.textContent = `${throttlePct}%`;
        if (brkBar) brkBar.style.width = `${brakePct}%`;
        if (brkTxt) brkTxt.textContent = `${brakePct}%`;

        // ERS
        const ersBar = document.getElementById('ersBarFill');
        const ersPct = document.getElementById('ersPctText');
        if (ersBar) ersBar.style.width = `${data.ers_percent || 100}%`;
        if (ersPct) ersPct.textContent = `${data.ers_percent || 100}%`;

        // G-Forces and Position Coordinates
        const coordsEl = document.getElementById('posCoordsDisplay');
        if (coordsEl && data.car_x !== undefined) {
            coordsEl.textContent = `${data.car_x.toFixed(1)}, ${data.car_z.toFixed(1)}`;
        }
        const latG = document.getElementById('latGDisplay');
        if (latG) latG.textContent = `${(data.g_force_lat || 0).toFixed(2)} G`;
        const lonG = document.getElementById('lonGDisplay');
        if (lonG) lonG.textContent = `${(data.g_force_lon || 0).toFixed(2)} G`;

        // 7. Live Track Map: stored circuit + pit lane, car moving on it
        updateMapState(data);
        redrawTrackMap();

        // 8. Radio Transcript Log
        const voiceList = document.getElementById('voiceList');
        if (voiceList && dirty('voice_log')) {
            const logs = data.voice_log || [];
            const countEl = document.getElementById('logCount');
            if (countEl) countEl.textContent = `${logs.length} messages`;

            voiceList.innerHTML = logs.map(item => `
                <div class="voice-item">
                    <span class="voice-time">${item.time}</span>
                    <span class="voice-text">${item.message}</span>
                </div>
            `).join('');
        }

        // 9. Corner Diagnostics Breakdown
        const cornerList = document.getElementById('cornerList');
        if (cornerList && dirty('corner_analysis')) {
            const corners = data.corner_analysis || [];
            cornerList.innerHTML = corners.map(c => `
                <div class="corner-card">
                    <div class="corner-title">
                        <span>${c.name}</span>
                        <span style="color: ${c.delta === null ? 'var(--text-dim)' : c.delta > 0.05 ? 'var(--f1-red)' : 'var(--sector-green)'};">
                            ${c.delta === null ? 'Guide only' : (c.delta > 0 ? '+' : '') + c.delta.toFixed(3) + 's'}
                        </span>
                    </div>
                    <div class="corner-stats">
                        <span>Apex: ${c.current_apex} km/h (Ref: ${c.best_apex} km/h, ${c.reference_source})</span>
                        <span>Braking: ${c.brake_point === null ? 'Not observed' : c.brake_point + 'm'}</span>
                    </div>
                    <div class="corner-advice ${c.delta !== null && c.delta <= 0.05 ? 'good' : ''}">
                        ${c.advice}
                    </div>
                </div>
            `).join('');
        }

        // 10. Speed Trace Chart
        if (speedChart && dirty('speed_trace_best', 'speed_trace_current', 'track_length')) {
            const points = trace => (trace || []).map(([x, y]) => ({ x, y }));
            speedChart.data.datasets[0].data = points(data.speed_trace_best);
            speedChart.data.datasets[1].data = points(data.speed_trace_current);
            speedChart.options.scales.x.max = data.track_length || undefined;
            speedChart.update('none');
        }

        // 11. Race Report Table (Feature: lap/sector/speed-trap report)
        if (dirty('race_report')) renderRaceReport(data.race_report);

        // 12. Pit Wall: timing tower, race chart, tyre stints, strategy odds
        renderPitWall(data, dirty);
    } catch (err) {
        console.error('[HUD] render failed', err);
    }
}

// ──────────────────────────────────────────────
// 12. Race Report Table (lap/sector/speed-trap telemetry report)
// ──────────────────────────────────────────────
function fmtSector(ms) {
    if (!ms || ms <= 0) return '--';
    return (ms / 1000).toFixed(3) + 's';
}

function renderRaceReport(laps) {
    const body = document.getElementById('reportTableBody');
    if (!body) return;
    if (!laps || laps.length === 0) {
        body.innerHTML = '<tr class="report-empty-row"><td colspan="7">No completed laps yet this session.</td></tr>';
        return;
    }
    body.innerHTML = laps.map(lap => `
        <tr class="${lap.eligible ? '' : 'report-row-invalid'}" title="${lap.eligible ? 'Valid lap' : 'Invalid / in-pit lap, excluded from reference comparisons'}">
            <td>${lap.lap_num}</td>
            <td>${fmtTime(lap.lap_time_ms)}</td>
            <td>${fmtSector(lap.s1_ms)}</td>
            <td>${fmtSector(lap.s2_ms)}</td>
            <td>${fmtSector(lap.s3_ms)}</td>
            <td>${lap.speed_trap_kph ? Math.round(lap.speed_trap_kph) + ' km/h' : '--'}</td>
            <td>${lap.compound || '--'}</td>
        </tr>
    `).join('');
}

// ──────────────────────────────────────────────
// 13. Pit Wall: timing tower, gap-to-leader chart, tyre stints, strategy odds
// ──────────────────────────────────────────────
let gapChart = null;
let oddsChart = null;
const COMPOUND_CLASS = { S: 'compound-soft', M: 'compound-medium', H: 'compound-hard',
                         I: 'compound-inter', W: 'compound-wet' };

function teamColour(teamId) {
    return TEAM_COLOURS[teamId] || '#94a3b8';
}

// "Verstappen" -> "VER"; the player without a name -> "YOU".
function driverCode(name, isPlayer, index) {
    const letters = String(name || '').replace(/[^A-Za-zÀ-ÿ]/g, '');
    if (letters.length >= 3) return letters.slice(0, 3).toUpperCase();
    return isPlayer ? 'YOU' : `CAR${(index ?? 0) + 1}`;
}

function compoundLetter(name) {
    const n = String(name || '').toLowerCase();
    if (!n) return null;
    if (n.startsWith('s') || n.includes('soft')) return 'S';
    if (n.startsWith('m')) return 'M';
    if (n.startsWith('h')) return 'H';
    if (n.startsWith('i')) return 'I';
    if (n.startsWith('w')) return 'W';
    return null;
}

function tyreBadge(letter) {
    if (!letter) return '';
    return `<span class="pw-tyre ${COMPOUND_CLASS[letter] || ''}">${letter}</span>`;
}

function pct(p) {
    return p == null ? '--' : `${Math.round(p * 100)}%`;
}

function initPitWallCharts() {
    if (typeof Chart === 'undefined') return;
    const gapCanvas = document.getElementById('gapChart');
    if (gapCanvas) {
        gapChart = new Chart(gapCanvas, {
            type: 'line',
            data: { datasets: [] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
                interaction: { mode: 'nearest', intersect: false },
                scales: {
                    x: { type: 'linear', min: 1, title: { display: true, text: 'Lap', color: '#64748b' },
                         grid: { color: 'rgba(39, 49, 66, 0.4)' }, ticks: { color: '#64748b', precision: 0 } },
                    y: { reverse: true, min: 0, title: { display: true, text: 'Gap to leader (s)', color: '#64748b' },
                         grid: { color: 'rgba(39, 49, 66, 0.4)' }, ticks: { color: '#64748b' } },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: c => `${c.dataset.label}: +${c.parsed.y.toFixed(1)}s (lap ${c.parsed.x})` } },
                },
            },
        });
    }
    const oddsCanvas = document.getElementById('oddsChart');
    if (oddsCanvas) {
        const line = (label, colour) => ({ label, data: [], borderColor: colour, backgroundColor: colour,
                                           borderWidth: 2, pointRadius: 2, tension: 0.25 });
        oddsChart = new Chart(oddsCanvas, {
            type: 'line',
            data: { datasets: [line('Win', '#b138dd'), line('Podium', '#00d2be'), line('Points', '#94a3b8')] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Lap simulated', color: '#64748b' },
                         grid: { color: 'rgba(39, 49, 66, 0.4)' }, ticks: { color: '#64748b', precision: 0 } },
                    y: { min: 0, max: 100, grid: { color: 'rgba(39, 49, 66, 0.4)' },
                         ticks: { color: '#64748b', callback: v => `${v}%` } },
                },
                plugins: { legend: { labels: { color: '#f8fafc', boxWidth: 12 } } },
            },
        });
    }
}

function renderPitWall(data, dirty) {
    const race = (data.session_mode || 'race') === 'race';
    if (dirty('lap_num', 'total_laps')) {
        const badge = document.getElementById('pwLapBadge');
        if (badge) badge.textContent = `LAP ${data.lap_num || '--'} / ${data.total_laps || '--'}`;
    }
    if (dirty('timing_tower', 'session_mode')) renderTimingTower(data.timing_tower || [], race);
    if (dirty('gap_history')) renderGapChart(data.gap_history || []);
    if (dirty('tyre_stints', 'total_laps', 'lap_num', 'race_strategy')) renderStints(data);
    if (dirty('strategy_odds', 'position')) renderStrategyOdds(data.strategy_odds, data.position);
}

function towerGap(row, race) {
    if (row.status !== 'RUN' && row.status !== 'FIN') return row.status;
    if (row.laps_down > 0) return `<span class="pw-lapped">+${row.laps_down} LAP${row.laps_down > 1 ? 'S' : ''}</span>`;
    if (row.position === 1 && row.gap_s === 0) return race ? 'LEADER' : '—';
    return row.gap_s == null ? '--' : `+${row.gap_s.toFixed(3)}`;
}

function renderTimingTower(rows, race) {
    const body = document.getElementById('pwTowerBody');
    if (!body) return;
    const gapHead = document.getElementById('pwGapHead');
    if (gapHead) gapHead.textContent = race ? 'GAP' : 'GAP (BEST)';
    if (!rows.length) {
        body.innerHTML = '<tr class="pw-empty"><td colspan="8">Waiting for the field&hellip;</td></tr>';
        return;
    }
    body.innerHTML = rows.map(r => {
        const out = r.status !== 'RUN' && r.status !== 'FIN';
        const cls = [r.is_player ? 'pw-player' : '', out ? 'pw-out' : ''].join(' ');
        const name = r.name ? escapeHtml(r.name) : (r.is_player ? 'You' : '');
        const interval = r.interval_s == null ? (r.position === 1 ? '' : '--')
            : `<span class="${race && r.interval_s < 1.0 ? 'pw-int-close' : ''}">+${r.interval_s.toFixed(3)}</span>`;
        const best = r.best_lap_ms ? `<span class="${r.fastest_lap ? 'pw-fastest' : ''}">${fmtTime(r.best_lap_ms)}</span>` : '--';
        const pit = r.pit ? '<span class="pw-inpit">PIT</span>' : `<span class="pw-stops">${r.stops || 0}</span>`;
        return `<tr class="${cls}">
            <td class="c-pos"><span class="pw-pos">${r.position}</span></td>
            <td class="c-drv"><div class="pw-drv"><span class="pw-team" style="background:${teamColour(r.team_id)}"></span>
                <span class="pw-code">${escapeHtml(driverCode(r.name, r.is_player, r.index))}</span>
                <span class="pw-name">${name}</span></div></td>
            <td class="c-num">${towerGap(r, race)}</td>
            <td class="c-num">${interval}</td>
            <td class="c-num c-last">${r.last_lap_ms ? fmtTime(r.last_lap_ms) : '--'}</td>
            <td class="c-num c-best">${best}</td>
            <td class="c-tyre">${tyreBadge(r.compound)}${r.tyre_age != null ? `<span class="pw-age">${r.tyre_age}</span>` : ''}</td>
            <td class="c-pit">${pit}</td>
        </tr>`;
    }).join('');
}

function renderGapChart(cars) {
    const empty = document.getElementById('gapChartEmpty');
    if (empty) empty.style.display = cars.some(c => c.points.length) ? 'none' : '';
    if (!gapChart) return;
    const seenTeams = {};
    // The player is drawn last so their line sits on top.
    const ordered = cars.filter(c => !c.is_player).concat(cars.filter(c => c.is_player));
    gapChart.data.datasets = ordered.map(c => {
        const colour = c.is_player ? '#ffffff' : teamColour(c.team_id);
        const second = c.team_id != null && seenTeams[c.team_id];
        if (c.team_id != null) seenTeams[c.team_id] = true;
        return {
            label: driverCode(c.name, c.is_player, c.index),
            data: c.points.map(([x, y]) => ({ x, y })),
            borderColor: colour, backgroundColor: colour,
            borderWidth: c.is_player ? 3 : 1.5,
            borderDash: second && !c.is_player ? [5, 3] : [],
            pointRadius: c.is_player ? 2 : 0, tension: 0.15,
        };
    });
    gapChart.update('none');
}

// The rest of our race as the strategy model plans it: [{compound, start_lap, end_lap}].
function plannedStints(strategy, lapNum, totalLaps) {
    const best = strategy && strategy.best;
    if (!best || !best.compounds || !lapNum || !totalLaps) return [];
    const plan = [];
    best.compounds.forEach((compound, i) => {
        const start = i === 0 ? lapNum + 1 : best.pit_laps[i - 1] + 1;
        const end = i < best.pit_laps.length ? best.pit_laps[i] : totalLaps;
        const letter = compoundLetter(compound);
        if (letter && end >= start) plan.push({ compound: letter, start_lap: start, end_lap: end });
    });
    return plan;
}

function renderStints(data) {
    const box = document.getElementById('pwStints');
    if (!box) return;
    const cars = data.tyre_stints || [];
    if (!cars.length) {
        box.innerHTML = '<div class="pw-empty-block">No tyre data yet.</div>';
        return;
    }
    const lapNum = data.lap_num || 0;
    const maxEnd = Math.max(1, ...cars.flatMap(c => c.stints.map(s => s.end_lap)));
    const total = Math.max(data.total_laps || 0, maxEnd, lapNum);
    const plan = (data.session_mode || 'race') === 'race' ? plannedStints(data.race_strategy, lapNum, total) : [];
    const segment = (s, extra) => {
        const left = (s.start_lap - 1) / total * 100;
        const width = (s.end_lap - s.start_lap + 1) / total * 100;
        const laps = s.end_lap - s.start_lap + 1;
        const label = width > 4 ? `${s.compound}${laps}` : '';
        return `<div class="pw-seg ${COMPOUND_CLASS[s.compound] || ''} ${extra}" style="left:${left}%;width:${width}%"
                     title="${s.compound} laps ${s.start_lap}–${s.end_lap} (${laps})">${label}</div>`;
    };
    const now = lapNum ? `<div class="pw-now" style="left:${Math.min(100, lapNum / total * 100)}%"></div>` : '';
    const rows = cars.map(c => {
        const segs = c.stints.map(s => segment(s, s.open ? 'pw-open' : '')).join('')
            + (c.is_player ? plan.map(s => segment(s, 'pw-plan')).join('') : '');
        return `<div class="pw-stint-row ${c.is_player ? 'pw-player' : ''}">
            <div class="pw-stint-label"><span class="pw-p">${c.position || ''}</span>${escapeHtml(driverCode(c.name, c.is_player, c.index))}</div>
            <div class="pw-stint-track">${segs}${now}</div>
        </div>`;
    });
    const step = total > 40 ? 10 : 5;
    const ticks = [];
    for (let lap = step; lap <= total; lap += step) {
        ticks.push(`<span style="left:${(lap - 0.5) / total * 100}%">${lap}</span>`);
    }
    box.innerHTML = rows.join('') + `<div class="pw-axis">${ticks.join('')}</div>`;
}

function trendText(history, key) {
    if (!history || history.length < 2) return '';
    const diff = Math.round((history[history.length - 1][key] - history[history.length - 2][key]) * 100);
    if (diff === 0) return 'no change';
    return `<span class="${diff > 0 ? 'pw-up' : 'pw-down'}">${diff > 0 ? '▲' : '▼'} ${Math.abs(diff)} pts</span> since L${history[history.length - 2].lap}`;
}

function renderStrategyOdds(odds, position) {
    const empty = document.getElementById('pwOddsEmpty');
    const body = document.getElementById('pwOddsBody');
    if (!body) return;
    if (!odds) {
        if (empty) empty.style.display = '';
        body.style.display = 'none';
        return;
    }
    if (empty) empty.style.display = 'none';
    body.style.display = '';
    const meta = document.getElementById('pwOddsMeta');
    if (meta) meta.textContent = odds.runs ? `${odds.runs} simulated races from lap ${odds.lap} · ${odds.laps_to_go} to go` : 'Race finished · final result';
    const set = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
    set('pwWin', pct(odds.p_win));
    set('pwPodium', pct(odds.p_podium));
    set('pwPoints', pct(odds.p_points));
    set('pwWinTrend', trendText(odds.history, 'p_win'));
    set('pwPodiumTrend', trendText(odds.history, 'p_podium'));
    set('pwPointsTrend', trendText(odds.history, 'p_points'));
    const range = odds.position_range || [];
    set('pwExpected', range.length ? `P${range[1]}` : '--');
    set('pwExpectedRange', range.length ? `P${range[0]}–P${range[2]} (80%)` : '');

    set('pwPlans', (odds.plans || []).map((p, i) => {
        const tyres = (p.compounds || []).map(c => tyreBadge(compoundLetter(c))).join('');
        const box = p.pit_laps && p.pit_laps.length ? `box L${p.pit_laps.join(', L')}` : 'no stop';
        const r = p.position_range || [];
        const slower = p.time_vs_best_s && i > 0 ? ` · +${p.time_vs_best_s[1].toFixed(1)}s` : '';
        return `<div class="pw-plan ${i === 0 ? 'pw-best' : ''}">
            <div class="pw-plan-name">${tyres}<span>${p.stops}-STOP</span></div>
            <div class="pw-plan-pct">${pct(p.p_best)}</div>
            <div class="pw-plan-bar"><div style="width:${Math.round((p.p_best || 0) * 100)}%"></div></div>
            <div class="pw-plan-meta">${box} · P${r[1] ?? '-'} (P${r[0] ?? '-'}–P${r[2] ?? '-'}) · win ${pct(p.p_win)} · podium ${pct(p.p_podium)}${slower}</div>
        </div>`;
    }).join(''));

    const probs = odds.position_probs || [];
    const peak = Math.max(0.01, ...probs);
    set('pwDist', probs.map((p, i) => {
        const pos = i + 1;
        const cls = [pos <= 3 ? 'pw-podium' : pos <= 10 ? 'pw-points' : '', pos === position ? 'pw-current' : ''].join(' ');
        return `<div class="pw-dist-col ${cls}" title="P${pos}: ${(p * 100).toFixed(1)}%">
            <div class="pw-dist-bar" style="height:${Math.round(p / peak * 100)}%"></div>
            <span class="pw-dist-lbl">${pos % 5 === 0 || pos === 1 ? pos : ''}</span>
        </div>`;
    }).join(''));
    const loss = odds.pit_loss_s || [];
    set('pwFactors', [
        `SC ${pct(odds.p_safety_car)}`, `VSC ${pct(odds.p_vsc)}`, `RAIN ${pct(odds.p_rain)}`,
        loss.length ? `PIT LOSS ${loss[1].toFixed(1)}s` : null,
    ].filter(Boolean).map(t => `<span class="pw-factor">${t}</span>`).join(''));

    if (oddsChart) {
        const history = odds.history || [];
        ['p_win', 'p_podium', 'p_points'].forEach((key, i) => {
            oddsChart.data.datasets[i].data = history.map(h => ({ x: h.lap, y: Math.round(h[key] * 1000) / 10 }));
        });
        oddsChart.update('none');
    }
}
