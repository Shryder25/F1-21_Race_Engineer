/**
 * F1 2021 AI Race Engineer — Client Controller & Telemetry Engine
 * ==============================================================
 * Manages:
 *   1. Real-time UDP telemetry polling & UI rendering (Timing Tower, Tyres, Pedals)
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

    // Start polling telemetry
    setInterval(fetchTelemetry, 150);
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
                drawTrackMap(lastKnownCoords, lastCarX, lastCarZ, lastBrakingZones, lastErsZones);
            } else if (btn.dataset.tab === 'tab-corners' && speedChart) {
                speedChart.resize();
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
let lastCarX = 0, lastCarZ = 0;
let lastBrakingZones = [];
let lastErsZones = [];

function initCanvas() {
    trackCanvas = document.getElementById('trackCanvas');
    if (!trackCanvas) return;
    trackCtx = trackCanvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    document.getElementById('btnFitMap')?.addEventListener('click', () => {
        drawTrackMap(lastKnownCoords, lastCarX, lastCarZ, lastBrakingZones, lastErsZones, true);
    });

    document.getElementById('btnClearTrace')?.addEventListener('click', () => {
        // Clear the local trail immediately so the button feels instant...
        lastKnownCoords = [];
        lastBrakingZones = [];
        lastErsZones = [];
        drawTrackMap([], 0, 0, [], []);
        // ...then tell the server to drop its recorded points too, otherwise the very
        // next telemetry poll repopulates the map from the old (un-cleared) trail.
        fetch('/api/reset_map', { method: 'POST' }).catch(() => {});
    });
}

function resizeCanvas() {
    if (!trackCanvas) return;
    const rect = trackCanvas.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
        trackCanvas.width = rect.width;
        trackCanvas.height = rect.height;
    }
}

function drawTrackMap(points, carX, carZ, brakingZones = [], ersZones = [], forceFit = false) {
    if (!trackCtx || !trackCanvas) return;
    const width = trackCanvas.width;
    const height = trackCanvas.height;

    // Clear background
    trackCtx.fillStyle = '#07090e';
    trackCtx.fillRect(0, 0, width, height);

    // Draw subtle grid lines
    trackCtx.strokeStyle = 'rgba(39, 49, 66, 0.3)';
    trackCtx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < width; x += gridSize) {
        trackCtx.beginPath();
        trackCtx.moveTo(x, 0);
        trackCtx.lineTo(x, height);
        trackCtx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
        trackCtx.beginPath();
        trackCtx.moveTo(0, y);
        trackCtx.lineTo(width, y);
        trackCtx.stroke();
    }

    if (!points || points.length < 2) {
        // Draw waiting prompt
        trackCtx.fillStyle = '#64748b';
        trackCtx.font = "bold 13px 'Barlow Condensed', sans-serif";
        trackCtx.textAlign = 'center';
        trackCtx.fillText('RECORDING TRACK COORDINATES... DRIVE A LAP TO MAP CIRCUIT', width / 2, height / 2);
        return;
    }

    // Determine bounding box
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of points) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minZ) minZ = p[1];
        if (p[1] > maxZ) maxZ = p[1];
    }

    const rangeX = (maxX - minX) || 1;
    const rangeZ = (maxZ - minZ) || 1;
    const padding = 50;

    const scaleX = (width - padding * 2) / rangeX;
    const scaleZ = (height - padding * 2) / rangeZ;
    const scale = Math.min(scaleX, scaleZ);

    const offsetX = (width - rangeX * scale) / 2;
    const offsetZ = (height - rangeZ * scale) / 2;

    const toCanvas = (x, z) => [
        offsetX + (x - minX) * scale,
        height - (offsetZ + (z - minZ) * scale) // Flip Z for correct orientation
    ];

    // 1. Draw glowing outer track glow
    trackCtx.strokeStyle = 'rgba(0, 210, 190, 0.15)';
    trackCtx.lineWidth = 10;
    trackCtx.lineCap = 'round';
    trackCtx.lineJoin = 'round';
    trackCtx.beginPath();
    const [startX, startZ] = toCanvas(points[0][0], points[0][1]);
    trackCtx.moveTo(startX, startZ);
    for (let i = 1; i < points.length; i++) {
        const [cx, cz] = toCanvas(points[i][0], points[i][1]);
        trackCtx.lineTo(cx, cz);
    }
    trackCtx.stroke();

    // 2. Draw sharp track racing line
    trackCtx.strokeStyle = '#273142';
    trackCtx.lineWidth = 4;
    trackCtx.beginPath();
    trackCtx.moveTo(startX, startZ);
    for (let i = 1; i < points.length; i++) {
        const [cx, cz] = toCanvas(points[i][0], points[i][1]);
        trackCtx.lineTo(cx, cz);
    }
    trackCtx.stroke();

    trackCtx.strokeStyle = '#94a3b8';
    trackCtx.lineWidth = 2;
    trackCtx.stroke();

    // 3. Draw ERS Deployment Zones (Cyan / Teal)
    if (ersZones && ersZones.length > 0) {
        trackCtx.fillStyle = '#00d2be';
        trackCtx.shadowColor = '#00d2be';
        trackCtx.shadowBlur = 6;
        for (const pt of ersZones) {
            const [ex, ez] = toCanvas(pt[0], pt[1]);
            trackCtx.beginPath();
            trackCtx.arc(ex, ez, 3, 0, Math.PI * 2);
            trackCtx.fill();
        }
        trackCtx.shadowBlur = 0;
    }

    // 4. Draw Optimal Braking Zones (Red / Coral)
    if (brakingZones && brakingZones.length > 0) {
        trackCtx.fillStyle = '#e10600';
        trackCtx.shadowColor = '#e10600';
        trackCtx.shadowBlur = 8;
        for (const pt of brakingZones) {
            const [bx, bz] = toCanvas(pt[0], pt[1]);
            const radius = pt[2] ? Math.max(3, Math.min(6, pt[2] * 6)) : 4;
            trackCtx.beginPath();
            trackCtx.arc(bx, bz, radius, 0, Math.PI * 2);
            trackCtx.fill();
        }
        trackCtx.shadowBlur = 0;
    }

    // 5. Draw Start / Finish Line
    const [sfx, sfz] = toCanvas(points[0][0], points[0][1]);
    trackCtx.fillStyle = '#fff';
    trackCtx.strokeStyle = '#000';
    trackCtx.lineWidth = 2;
    trackCtx.beginPath();
    trackCtx.arc(sfx, sfz, 6, 0, Math.PI * 2);
    trackCtx.fill();
    trackCtx.stroke();

    // 6. Draw Live Car Position Beacon
    if (carX !== undefined && carZ !== undefined && (carX !== 0 || carZ !== 0)) {
        const [px, pz] = toCanvas(carX, carZ);

        // Radar ping
        trackCtx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        trackCtx.lineWidth = 2;
        trackCtx.beginPath();
        trackCtx.arc(px, pz, 14, 0, Math.PI * 2);
        trackCtx.stroke();

        // Pulsing Car Center Dot
        trackCtx.fillStyle = '#ffffff';
        trackCtx.shadowColor = '#ffffff';
        trackCtx.shadowBlur = 10;
        trackCtx.beginPath();
        trackCtx.arc(px, pz, 5, 0, Math.PI * 2);
        trackCtx.fill();
        trackCtx.shadowBlur = 0;
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

    trackSelect.addEventListener('change', () => loadTrackSetup(trackSelect.value, weatherSelect.value));
    weatherSelect.addEventListener('change', () => loadTrackSetup(trackSelect.value, weatherSelect.value));

    btnConsult?.addEventListener('click', async () => {
        btnConsult.disabled = true;
        btnConsult.innerHTML = '<span>⏳</span> CONSULTING DEEPSEEK...';
        showLatencyCue(true);
        try {
            const res = await fetch('/api/setup/consult', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    track_id: parseInt(trackSelect.value, 10),
                    weather: weatherSelect.value,
                    handling_issue: balanceSelect.value
                })
            });
            if (res.ok) {
                const data = await res.json();
                renderSetupData(data);
            }
        } catch (err) {
            console.error('Setup consult error:', err);
        } finally {
            btnConsult.disabled = false;
            btnConsult.innerHTML = '<span class="btn-icon">🧠</span> CONSULT DEEPSEEK SETUP ADVISOR';
            showLatencyCue(false);
        }
    });

    // Initial load
    loadTrackSetup(trackSelect.value, weatherSelect ? weatherSelect.value : 'dry');
}

async function loadTrackSetup(trackId, weather) {
    try {
        const res = await fetch(`/api/setup?track_id=${trackId}&weather=${weather}`);
        if (res.ok) {
            const data = await res.json();
            renderSetupData(data);
        }
    } catch (e) {
        console.error('Error fetching track setup:', e);
    }
}

function renderSetupData(data) {
    const s = data.setup;
    if (!s) return;

    // Header banner
    const title = document.getElementById('setupTrackTitle');
    if (title) title.textContent = `${data.track_name} — ${data.country}`;
    const desc = document.getElementById('setupCharacteristics');
    if (desc) desc.textContent = data.characteristics || '';
    const df = document.getElementById('setupDownforce');
    if (df) df.textContent = `Downforce: ${data.downforce_level}`;

    // Aerodynamics
    document.getElementById('valFrontWing').textContent = s.aerodynamics?.front_wing ?? '7';
    document.getElementById('valRearWing').textContent = s.aerodynamics?.rear_wing ?? '8';

    // Transmission
    document.getElementById('valDiffOn').textContent = (s.transmission?.diff_on_throttle ?? 65) + '%';
    document.getElementById('valDiffOff').textContent = (s.transmission?.diff_off_throttle ?? 54) + '%';

    // Suspension Geometry
    document.getElementById('valFrontCamber').textContent = (s.suspension_geometry?.front_camber ?? -3.00) + '°';
    document.getElementById('valRearCamber').textContent = (s.suspension_geometry?.rear_camber ?? -1.50) + '°';
    document.getElementById('valFrontToe').textContent = (s.suspension_geometry?.front_toe ?? 0.08) + '°';
    document.getElementById('valRearToe').textContent = (s.suspension_geometry?.rear_toe ?? 0.22) + '°';

    // Suspension
    document.getElementById('valSuspension').textContent = `${s.suspension?.front_suspension ?? 5} / ${s.suspension?.rear_suspension ?? 4}`;
    document.getElementById('valArb').textContent = `${s.suspension?.front_anti_roll ?? 7} / ${s.suspension?.rear_anti_roll ?? 5}`;
    document.getElementById('valRideHeight').textContent = `${s.suspension?.front_ride_height ?? 3} / ${s.suspension?.rear_ride_height ?? 5}`;

    // Brakes
    document.getElementById('valBrakePressure').textContent = (s.brakes?.brake_pressure ?? 98) + '%';
    document.getElementById('valBrakeBias').textContent = (s.brakes?.brake_bias ?? 56) + '%';

    // Tyres
    document.getElementById('valFrontRightPsi').textContent = (s.tyres?.front_right_pressure ?? 23.5) + ' PSI';
    document.getElementById('valFrontLeftPsi').textContent = (s.tyres?.front_left_pressure ?? 23.5) + ' PSI';
    document.getElementById('valRearPsi').textContent = (s.tyres?.rear_left_pressure ?? 22.0) + ' PSI';

    // Briefing text
    const brief = document.getElementById('setupBriefingText');
    if (brief) {
        brief.textContent = data.ai_consultation || data.engineer_advice || s.notes || 'Setup loaded.';
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

async function fetchTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

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

        // 3. Safety Car Pill
        const scPill = document.getElementById('safetyCarPill');
        const scText = document.getElementById('safetyCarText');
        if (scPill && scText) {
            scPill.className = 'safety-car-pill';
            if (data.safety_car === 'SAFETY CAR') {
                scPill.classList.add('sc-full');
                scText.textContent = 'SAFETY CAR DEPLOYED';
            } else if (data.safety_car === 'VSC') {
                scPill.classList.add('sc-vsc');
                scText.textContent = 'VIRTUAL SAFETY CAR';
            } else {
                scPill.classList.add('sc-none');
                scText.textContent = 'TRACK CLEAR';
            }
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
                ? `INTERVAL +${data.gap_ahead_s.toFixed(1)}s` : 'INTERVAL --';
        }
        const gapBehind = document.getElementById('gapBehindDisplay');
        if (gapBehind) {
            gapBehind.textContent = data.gap_behind_s !== null && data.gap_behind_s !== undefined
                ? `INTERVAL -${data.gap_behind_s.toFixed(1)}s` : 'INTERVAL --';
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
        if (threatsCard && threatsList) {
            const threats = data.opponent_threats || [];
            threatsCard.style.display = threats.length ? '' : 'none';
            threatsList.innerHTML = threats.map(t => {
                const side = t.gap_s < 0 ? 'BEHIND' : 'AHEAD';
                const cls = t.threat_level === 'HIGH' ? 'threat-high'
                          : t.threat_level === 'MEDIUM' ? 'threat-medium' : 'threat-low';
                const note = t.factors && t.factors.length ? ` — ${t.factors[0]}` : '';
                return `<div class="gap-item ${cls}">
                    <span class="gap-label">${side} ${Math.abs(t.gap_s).toFixed(1)}s</span>
                    <span class="gap-val">${t.threat_level}${note}</span>
                </div>`;
            }).join('');
        }

        // Race Control feed (newest first)
        const raceControlCard = document.getElementById('raceControlCard');
        const raceControlList = document.getElementById('raceControlList');
        if (raceControlCard && raceControlList) {
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

        // Sector Splits (Purple/Green/Yellow)
        const currentSectors = data.current_sectors_ms || {};
        const personalSectors = data.personal_sectors_ms || {};

        for (let s = 1; s <= 3; s++) {
            const box = document.getElementById(`sectorBox${s}`);
            const timeEl = document.getElementById(`secTime${s}`);
            const deltaEl = document.getElementById(`secDelta${s}`);
            if (!box || !timeEl) continue;

            const timeMs = currentSectors[s];
            const bestMs = personalSectors[s];

            box.className = 'sector-box';
            if (timeMs && timeMs > 0) {
                timeEl.textContent = (timeMs / 1000).toFixed(3);
                if (bestMs) {
                    const diff = timeMs - bestMs;
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

        // 7. Live Track Map Drawing (Feature 3)
        if (data.track_map && data.track_map.length > 0) {
            lastKnownCoords = data.track_map;
        }
        if (data.car_x !== undefined) lastCarX = data.car_x;
        if (data.car_z !== undefined) lastCarZ = data.car_z;
        if (data.braking_zones) lastBrakingZones = data.braking_zones;
        if (data.ers_zones) lastErsZones = data.ers_zones;

        drawTrackMap(lastKnownCoords, lastCarX, lastCarZ, lastBrakingZones, lastErsZones);

        // 8. Radio Transcript Log
        const voiceList = document.getElementById('voiceList');
        if (voiceList) {
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
        if (cornerList) {
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
        if (speedChart) {
            const points = trace => (trace || []).map(([x, y]) => ({ x, y }));
            speedChart.data.datasets[0].data = points(data.speed_trace_best);
            speedChart.data.datasets[1].data = points(data.speed_trace_current);
            speedChart.options.scales.x.max = data.track_length || undefined;
            speedChart.update('none');
        }

        // 11. Race Report Table (Feature: lap/sector/speed-trap report)
        renderRaceReport(data.race_report);

    } catch (err) {
        // Disconnected state
        const beacon = document.getElementById('statusBeacon');
        const statusText = document.getElementById('statusText');
        if (beacon) beacon.classList.remove('connected');
        if (statusText) statusText.innerText = 'DISCONNECTED';
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
