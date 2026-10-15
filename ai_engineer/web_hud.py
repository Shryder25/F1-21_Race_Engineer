"""
F1 2021 AI Race Engineer — Web Visualizer & Live Telemetry HUD
==============================================================
Runs a lightweight HTTP server on port 5000.
Provides:
  1. Live telemetry gauges (Speed, RPM, Gear, Throttle/Brake, Tyres)
  2. Visual Speed Trace (Current Lap vs Best Lap)
  3. Corner-by-Corner "Why You Are Slower" Delta Breakdown
  4. Audio Callout Log & Setup Recommendations
"""

import http.server
import socketserver
import threading
import json
import time
import copy
from typing import Any

TELEMETRY_TIMEOUT = 2.0

# Shared data store for web client
dashboard_state: dict[str, Any] = {
    "connected": False,
    "track_name": "Silverstone",
    "track_id": 7,
    "track_length": 5891,
    "speed": 0,
    "gear": 0,
    "rpm": 0,
    "drs": False,
    "throttle": 0,
    "brake": 0,
    "tyre_surface_temps": [100, 100, 100, 100],  # RL, RR, FL, FR
    "tyre_inner_temps": [95, 95, 95, 95],
    "tyre_pressures": [23.5, 23.5, 22.5, 22.5],
    "engine_temp": 90,
    "lap_num": 1,
    "position": 1,
    "lap_time_ms": 0,
    "last_lap_time_ms": 0,
    "best_lap_time_ms": 0,
    "delta_ms": 0,
    "lap_distance": 0.0,
    "sector": 0,
    "voice_log": [],
    "corner_analysis": [],
    "speed_trace_current": [],
    "speed_trace_best": []
}

state_lock = threading.Lock()

def update_web_state(data: dict[str, Any]):
    with state_lock:
        dashboard_state.update(copy.deepcopy(data))


def get_web_state(now=None):
    now = time.monotonic() if now is None else now
    with state_lock:
        snapshot = copy.deepcopy(dashboard_state)
        last_update = snapshot.pop('last_update_monotonic', None)
        age = max(0.0, now - last_update) if last_update is not None else None
        snapshot['connected'] = bool(snapshot['connected'] and age is not None and age <= TELEMETRY_TIMEOUT)
        snapshot['data_age_seconds'] = age
        return snapshot

def add_voice_log(message: str, status='spoken'):
    with state_lock:
        log = dashboard_state.get("voice_log", [])
        log.insert(0, {"time": time.strftime("%H:%M:%S"), "message": message, 'status': status})
        dashboard_state["voice_log"] = log[:20]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 2021 AI Race Engineer - Telemetry HUD</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg: #0b0e14;
            --panel: #151922;
            --border: #232936;
            --f1-red: #e10600;
            --accent: #00d2be;
            --text: #f0f3f8;
            --text-dim: #8b949e;
            --green: #2ecc71;
            --yellow: #f1c40f;
            --orange: #e67e22;
            --blue: #3498db;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 16px; min-height: 100vh; }
        
        /* Header */
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 2px solid var(--border); margin-bottom: 16px; }
        .logo { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 20px; letter-spacing: 1px; }
        .f1-badge { background: var(--f1-red); color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: 900; }
        .status-badge { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; padding: 6px 12px; border-radius: 20px; background: rgba(0,210,190,0.1); color: var(--accent); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); }
        .dot.pulse { animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }

        /* Grid Layout */
        .grid { display: grid; grid-template-columns: 320px minmax(0, 1fr) 340px; gap: 16px; }
        .grid > div { min-width: 0; }
        @media (max-width: 1200px) { .grid { grid-template-columns: minmax(0, 1fr); } }
        @media (max-width: 600px) { header { flex-wrap: wrap; gap: 12px; } .logo { font-size: 16px; } }

        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 16px; }
        .card-title { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }

        /* Main Gauge */
        .hud-main { text-align: center; padding: 10px 0; }
        .speed-val { font-size: 64px; font-weight: 900; line-height: 1; color: #fff; letter-spacing: -2px; }
        .speed-unit { font-size: 14px; color: var(--text-dim); text-transform: uppercase; font-weight: 700; margin-top: 4px; }
        .gear-badge { display: inline-block; width: 60px; height: 60px; line-height: 60px; font-size: 38px; font-weight: 900; background: var(--f1-red); border-radius: 8px; margin: 10px 0; }
        .drs-badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 800; background: #222; color: #666; margin-left: 10px; }
        .drs-badge.active { background: var(--green); color: #000; box-shadow: 0 0 10px var(--green); }

        /* RPM Bar */
        .rpm-bar-container { width: 100%; height: 12px; background: #1c212c; border-radius: 6px; overflow: hidden; margin-top: 12px; display: flex; gap: 2px; }
        .rpm-led { flex: 1; background: #2a3140; border-radius: 2px; }
        .rpm-led.green { background: #2ecc71; }
        .rpm-led.red { background: #e74c3c; }
        .rpm-led.purple { background: #9b59b6; box-shadow: 0 0 8px #9b59b6; }

        /* Pedals */
        .pedal-group { display: flex; gap: 16px; margin-top: 16px; }
        .pedal { flex: 1; text-align: center; }
        .pedal-bar-outer { height: 120px; width: 32px; background: #1c212c; border-radius: 6px; margin: 8px auto; position: relative; overflow: hidden; display: flex; align-items: flex-end; }
        .pedal-fill { width: 100%; transition: height 0.05s ease; border-radius: 0 0 6px 6px; }
        .pedal-fill.throttle { background: var(--green); }
        .pedal-fill.brake { background: var(--f1-red); }

        /* Tyres Grid */
        .tyres-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
        .tyre-card { background: #1c212c; border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; position: relative; }
        .tyre-pos { font-size: 11px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; }
        .tyre-temp { font-size: 26px; font-weight: 800; margin: 4px 0; }
        .tyre-psi { font-size: 12px; color: var(--text-dim); }

        /* Lap Timing & Delta */
        .timing-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
        .timing-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; font-weight: 700; }
        .timing-val { font-size: 22px; font-weight: 800; font-family: monospace; }
        .delta-box { text-align: center; padding: 16px; border-radius: 8px; background: #1c212c; margin-top: 10px; }
        .delta-val { font-size: 42px; font-weight: 900; font-family: monospace; }
        .delta-val.faster { color: var(--green); text-shadow: 0 0 15px rgba(46,204,113,0.3); }
        .delta-val.slower { color: var(--f1-red); text-shadow: 0 0 15px rgba(231,76,60,0.3); }

        /* Corner Analysis */
        .corner-list { display: flex; flex-direction: column; gap: 10px; max-height: 480px; overflow-y: auto; padding-right: 4px; }
        .corner-item { background: #1c212c; border-radius: 8px; padding: 12px; border-left: 4px solid var(--border); transition: transform 0.1s; }
        .corner-item.losing { border-left-color: var(--f1-red); }
        .corner-item.gaining { border-left-color: var(--green); }
        .corner-header { display: flex; justify-content: space-between; font-weight: 700; font-size: 14px; margin-bottom: 6px; }
        .corner-stats { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-dim); margin-bottom: 6px; }
        .corner-advice { font-size: 12px; background: rgba(225,6,0,0.1); color: #ff9999; padding: 6px 8px; border-radius: 4px; border: 1px solid rgba(225,6,0,0.2); }
        .corner-advice.good { background: rgba(46,204,113,0.1); color: #a3e9b9; border-color: rgba(46,204,113,0.2); }

        /* Voice log */
        .voice-list { display: flex; flex-direction: column; gap: 8px; max-height: 250px; overflow-y: auto; }
        .voice-item { font-size: 12px; padding: 8px 10px; background: #1c212c; border-radius: 6px; border-left: 3px solid var(--accent); }
        .voice-time { color: var(--text-dim); font-size: 10px; margin-bottom: 2px; }

        /* Chart */
        .chart-container { height: 260px; position: relative; margin-top: 10px; }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span class="f1-badge">F1 2021</span>
            <span>AI RACE ENGINEER</span>
        </div>
        <div class="status-badge" id="statusBadge">
            <span class="dot pulse" id="statusDot"></span>
            <span id="statusText">CONNECTING...</span>
        </div>
    </header>

    <div class="grid">
        <!-- Left: Telemetry Gauges & Tyres -->
        <div>
            <div class="card">
                <div class="card-title">
                    <span>Live Cockpit</span>
                    <span id="trackBadge" style="color: var(--accent); font-weight:700;">SILVERSTONE</span>
                </div>
                <div class="hud-main">
                    <div class="speed-val" id="speed">0</div>
                    <div class="speed-unit">KM / H</div>
                    <div>
                        <span class="gear-badge" id="gear">N</span>
                        <span class="drs-badge" id="drs">DRS</span>
                    </div>
                </div>

                <!-- RPM LEDs -->
                <div class="rpm-bar-container" id="rpmBar"></div>

                <!-- Pedals -->
                <div class="pedal-group">
                    <div class="pedal">
                        <div class="timing-label">Throttle</div>
                        <div class="pedal-bar-outer">
                            <div class="pedal-fill throttle" id="throttleFill" style="height: 0%;"></div>
                        </div>
                        <div id="throttleVal" style="font-weight:700; font-size:13px;">0%</div>
                    </div>
                    <div class="pedal">
                        <div class="timing-label">Brake</div>
                        <div class="pedal-bar-outer">
                            <div class="pedal-fill brake" id="brakeFill" style="height: 0%;"></div>
                        </div>
                        <div id="brakeVal" style="font-weight:700; font-size:13px;">0%</div>
                    </div>
                </div>
            </div>

            <!-- Tyres Thermal Grid -->
            <div class="card">
                <div class="card-title">Tyre Thermal Heatmap (°C)</div>
                <div class="tyres-grid">
                    <div class="tyre-card" id="tyreFL">
                        <div class="tyre-pos">Front Left</div>
                        <div class="tyre-temp" id="flTemp">--°C</div>
                        <div class="tyre-psi" id="flPsi">-- PSI</div>
                    </div>
                    <div class="tyre-card" id="tyreFR">
                        <div class="tyre-pos">Front Right</div>
                        <div class="tyre-temp" id="frTemp">--°C</div>
                        <div class="tyre-psi" id="frPsi">-- PSI</div>
                    </div>
                    <div class="tyre-card" id="tyreRL">
                        <div class="tyre-pos">Rear Left</div>
                        <div class="tyre-temp" id="rlTemp">--°C</div>
                        <div class="tyre-psi" id="rlPsi">-- PSI</div>
                    </div>
                    <div class="tyre-card" id="tyreRR">
                        <div class="tyre-pos">Rear Right</div>
                        <div class="tyre-temp" id="rrTemp">--°C</div>
                        <div class="tyre-psi" id="rrPsi">-- PSI</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Center: Visual Delta & Speed Trace -->
        <div>
            <!-- Live Delta Box -->
            <div class="card">
                <div class="timing-row">
                    <div>
                        <div class="timing-label">Lap</div>
                        <div class="timing-val" id="lapNum">P1 | Lap 1</div>
                    </div>
                    <div>
                        <div class="timing-label">Current Lap</div>
                        <div class="timing-val" id="currentLapTime">0:00.000</div>
                    </div>
                    <div>
                        <div class="timing-label">Track Reference</div>
                        <div class="timing-val" id="bestLapTime" style="color: var(--accent);">--:--.---</div>
                    </div>
                </div>

                <div style="font-size:12px; color:var(--text-dim); line-height:1.8;" id="raceInfo">Race information awaiting telemetry.</div>

                <div class="delta-box">
                    <div class="timing-label">Performance Delta (vs Best Clean Lap)</div>
                    <div class="delta-val" id="deltaVal">---</div>
                    <div style="font-size:12px; color:var(--text-dim); margin-top:4px;" id="deltaStatus">Awaiting completed lap...</div>
                </div>
            </div>

            <!-- Visual Speed Trace Comparison Chart -->
            <div class="card">
                <div class="card-title">
                    <span>Visual Lap Analysis: Speed Trace Comparison</span>
                    <span style="font-size:11px; text-transform:none; color:var(--text-dim);">Green = Best Lap | Yellow = Current Lap</span>
                </div>
                <div class="chart-container">
                    <canvas id="speedChart"></canvas>
                </div>
            </div>

            <!-- AI Radio Voice Callouts Log -->
            <div class="card">
                <div class="card-title">Pit Wall Radio Transcripts</div>
                <div class="voice-list" id="voiceList">
                    <div class="voice-item">
                        <div class="voice-time">Ready</div>
                        <div>Engineer online. Listening for telemetry...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right: Corner Analysis & Why You Are Slower -->
        <div>
            <div class="card" style="height: calc(100% - 16px);">
                <div class="card-title">
                    <span>Corner Diagnostic: Why You're Slower</span>
                </div>
                <div style="font-size:12px; color:var(--text-dim); margin-bottom:12px;">
                    Measured corner deltas use your recorded reference lap. Track guides provide approximate speed and braking advice only.
                </div>
                <div class="corner-list" id="cornerList">
                    <!-- Populated dynamically -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize RPM LEDs
        const rpmContainer = document.getElementById('rpmBar');
        for (let i = 0; i < 20; i++) {
            const led = document.createElement('div');
            led.className = 'rpm-led';
            rpmContainer.appendChild(led);
        }

        // Initialize Chart.js Speed Trace
        const ctx = document.getElementById('speedChart').getContext('2d');
        const speedChart = typeof Chart === 'undefined' ? null : new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Reference Best Lap',
                        data: [],
                        borderColor: '#2ecc71',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    },
                    {
                        label: 'Current Lap',
                        data: [],
                        borderColor: '#f1c40f',
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
                        min: 0,
                        title: { display: true, text: 'Track Distance (m)', color: '#8b949e' },
                        grid: { color: '#1c212c' },
                        ticks: { color: '#8b949e', maxTicksLimit: 8 }
                    },
                    y: {
                        title: { display: true, text: 'Speed (km/h)', color: '#8b949e' },
                        grid: { color: '#1c212c' },
                        ticks: { color: '#8b949e' },
                        min: 50,
                        max: 350
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f0f3f8' } }
                }
            }
        });

        function getTempColor(t) {
            if (t < 85) return '#3498db'; // cold
            if (t <= 102) return '#2ecc71'; // optimal
            if (t <= 110) return '#e67e22'; // overheating
            return '#e74c3c'; // critical
        }

        function formatMS(ms) {
            if (!ms || ms <= 0) return '0:00.000';
            const totalSec = ms / 1000;
            const min = Math.floor(totalSec / 60);
            const sec = (totalSec % 60).toFixed(3);
            return `${min}:${sec.padStart(6, '0')}`;
        }

        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                
                // Status
                const badge = document.getElementById('statusBadge');
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                if (data.connected) {
                    text.innerText = 'TELEMETRY LIVE';
                    badge.style.color = 'var(--accent)';
                    dot.style.background = 'var(--accent)';
                } else {
                    text.innerText = data.paused ? 'GAME PAUSED' : data.spectating ? 'SPECTATING' :
                        data.data_age_seconds == null ? 'WAITING FOR CAR...' : 'TELEMETRY STALE';
                    badge.style.color = '#e67e22';
                    dot.style.background = '#e67e22';
                }

                // Cockpit
                document.getElementById('trackBadge').innerText = (data.track_name || 'SILVERSTONE').toUpperCase();
                document.getElementById('speed').innerText = data.speed;
                document.getElementById('gear').innerText = data.gear === 0 ? 'N' : data.gear === -1 ? 'R' : data.gear;
                const drsEl = document.getElementById('drs');
                if (data.drs) { drsEl.className = 'drs-badge active'; } else { drsEl.className = 'drs-badge'; }

                // RPM LEDs
                const leds = document.querySelectorAll('.rpm-led');
                const pct = Math.min(100, Math.max(0, (data.rpm - 4000) / (12500 - 4000) * 100));
                const activeCount = Math.floor(pct / 5);
                leds.forEach((led, idx) => {
                    led.className = 'rpm-led';
                    if (idx < activeCount) {
                        if (idx < 10) led.classList.add('green');
                        else if (idx < 16) led.classList.add('red');
                        else led.classList.add('purple');
                    }
                });

                // Pedals
                const thPct = Math.round(data.throttle * 100);
                const brPct = Math.round(data.brake * 100);
                document.getElementById('throttleFill').style.height = thPct + '%';
                document.getElementById('throttleVal').innerText = thPct + '%';
                document.getElementById('brakeFill').style.height = brPct + '%';
                document.getElementById('brakeVal').innerText = brPct + '%';

                // Tyres: RL, RR, FL, FR
                const st = data.tyre_surface_temps || [100, 100, 100, 100];
                const ps = data.tyre_pressures || [22.0, 22.0, 22.0, 22.0];
                const tyres = [
                    { id: 'tyreFL', temp: st[2], psi: ps[2], tempEl: 'flTemp', psiEl: 'flPsi' },
                    { id: 'tyreFR', temp: st[3], psi: ps[3], tempEl: 'frTemp', psiEl: 'frPsi' },
                    { id: 'tyreRL', temp: st[0], psi: ps[0], tempEl: 'rlTemp', psiEl: 'rlPsi' },
                    { id: 'tyreRR', temp: st[1], psi: ps[1], tempEl: 'rrTemp', psiEl: 'rrPsi' }
                ];
                tyres.forEach(t => {
                    const el = document.getElementById(t.id);
                    const color = getTempColor(t.temp);
                    el.style.borderColor = color;
                    document.getElementById(t.tempEl).innerText = t.temp + '°C';
                    document.getElementById(t.tempEl).style.color = color;
                    document.getElementById(t.psiEl).innerText = (t.psi ? t.psi.toFixed(1) : '--') + ' PSI';
                });

                // Timing & Delta
                document.getElementById('lapNum').innerText = `P${data.position || 1} | Lap ${data.lap_num || 1}`;
                document.getElementById('currentLapTime').innerText = formatMS(data.lap_time_ms);
                document.getElementById('bestLapTime').innerText = formatMS(data.best_lap_time_ms);
                const gap = value => value == null ? '--' : value.toFixed(1) + 's';
                const pit = data.pit_window ? 'Laps ' + data.pit_window.join('–') : '--';
                document.getElementById('raceInfo').innerText =
                    `Ahead: ${gap(data.gap_ahead_s)} | Behind: ${gap(data.gap_behind_s)} | Pit window: ${pit} | ` +
                    `Pace (3 clean laps): ${data.race_pace_ms ? formatMS(data.race_pace_ms) : '--'}`;
                
                const deltaEl = document.getElementById('deltaVal');
                const deltaStatus = document.getElementById('deltaStatus');
                if (data.delta_ms != null && data.delta_ms !== 0 && data.best_lap_time_ms > 0) {
                    const sec = Math.abs(data.delta_ms) / 1000;
                    if (data.delta_ms < 0) {
                        deltaEl.className = 'delta-val faster';
                        deltaEl.innerText = `-${sec.toFixed(3)}s`;
                        deltaStatus.innerText = 'FASTER THAN BEST PACE';
                    } else {
                        deltaEl.className = 'delta-val slower';
                        deltaEl.innerText = `+${sec.toFixed(3)}s`;
                        deltaStatus.innerText = 'LOSING TIME TO BEST PACE';
                    }
                } else if (data.delta_ms === 0 && data.best_lap_time_ms > 0) {
                    deltaEl.className = 'delta-val';
                    deltaEl.innerText = '0.000s';
                    deltaStatus.innerText = 'MATCHING BEST PACE';
                } else {
                    deltaEl.className = 'delta-val';
                    deltaEl.innerText = '---';
                    deltaStatus.innerText = data.best_lap_time_ms > 0 ? 'No reference at this distance' : 'Complete lap to establish delta baseline';
                }

                // Voice Log
                const vList = document.getElementById('voiceList');
                vList.innerHTML = (data.voice_log || []).map(v => `
                        <div class="voice-item">
                            <div class="voice-time">${v.time}${v.status === 'muted' ? ' · Voice off' : ''}</div>
                            <div>${v.message}</div>
                        </div>
                    `).join('');

                // Corner Diagnostics
                const cList = document.getElementById('cornerList');
                cList.innerHTML = (data.corner_analysis || []).map(c => `
                        <div class="corner-item ${c.delta == null ? '' : c.delta > 0.05 ? 'losing' : 'gaining'}">
                            <div class="corner-header">
                                <span>${c.name}</span>
                                <span style="color: ${c.delta == null ? 'var(--text-dim)' : c.delta > 0.05 ? 'var(--f1-red)' : 'var(--green)'};">
                                    ${c.delta == null ? 'Guide only' : (c.delta > 0 ? '+' : '') + c.delta.toFixed(3) + 's'}
                                </span>
                            </div>
                            <div class="corner-stats">
                                <span>Apex: ${c.current_apex} km/h (Ref: ${c.best_apex} km/h, ${c.reference_source})</span>
                                <span>Braking: ${c.brake_point == null ? 'Not observed' : c.brake_point + 'm'}</span>
                            </div>
                            <div class="corner-advice ${c.delta != null && c.delta <= 0.05 ? 'good' : ''}">
                                ${c.advice}
                            </div>
                        </div>
                    `).join('');

                // Speed Trace Chart update
                if (speedChart) {
                    const points = trace => (trace || []).map(([x, y]) => ({ x, y }));
                    speedChart.data.datasets[0].data = points(data.speed_trace_best);
                    speedChart.data.datasets[1].data = points(data.speed_trace_current);
                    speedChart.options.scales.x.max = data.track_length || undefined;
                    speedChart.update('none');
                }
            } catch (err) {
                console.error(err);
                document.getElementById('statusText').innerText = 'HUD DISCONNECTED';
                document.getElementById('statusDot').style.background = '#e67e22';
            }
        }

        setInterval(fetchTelemetry, 150); // 7 FPS HUD refresh
    </script>
</body>
</html>
"""

class TelemetryHUDHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data_copy = json.dumps(get_web_state(), allow_nan=False)
            self.wfile.write(data_copy.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in terminal
        return

def start_web_hud_server(port: int = 5000):
    """Start the HUD HTTP server on a background daemon thread."""
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = ReusableTCPServer(("0.0.0.0", port), TelemetryHUDHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="WebHUDServer")
        thread.start()
        print(f"[WebHUD] Visual Telemetry Dashboard live at http://localhost:{port}")
        return httpd
    except Exception as e:
        print(f"[WebHUD] Warning: Could not bind to port {port}: {e}")
        return None
