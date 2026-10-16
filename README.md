# F1-21 Race Engineer (AI-Powered)

An intelligent, real-time virtual race engineer for the F1 2021 game. This project ingests live UDP telemetry data from the game to act as a highly realistic pit-wall strategist. Moving beyond simple static scripts, it leverages machine learning and Large Language Models (LLMs) to analyze lap data, predict pit windows, and communicate through an authentic Formula 1 radio protocol.

---

## 🚀 Core Features

* **DeepSeek AI Integration:** Uses local or cloud-based DeepSeek LLMs to process complex telemetry statistics and translate them into natural, strategic pit-wall advice.
* **Predictive Machine Learning:** Logs session data into a local SQLite database (`history.py`) and utilizes `scikit-learn` to calculate predictive tire degradation curves and optimal pit windows[cite: 1].
* **Voice-Activated Radio Command:** Trigger the engineer using voice commands (e.g., "Gap to car behind", "Weather report"). Includes a real-time "We are checking" latency audio cue while the AI processes the response.
* **Realistic F1 Radio Protocol:** Enforces strict communication rules. The engineer will not interrupt during heavy braking or high-G cornering, matching real-world F1 etiquette.
* **Live Track Mapping & Telemetry Visualizer:** Translates X and Z axis telemetry into a live track map on the HUD. Shows the car's current position alongside optimal braking and ERS deployment markers.
* **Comprehensive Setup Advisor:** Includes a track database (`track_data.py`) covering all 24 circuits in F1 2021[cite: 1]. Suggests aerodynamic, transmission, and suspension setups customized to specific tracks.
* **F1-TV Style Web HUD:** A professional, dark-mode web dashboard (`web_hud.py`) displaying live timing towers, color-coded sector deltas, tire wear percentages, and individual tire temperatures[cite: 1].

---

## 📂 Repository Architecture

The project is structured into distinct modules to handle stream ingestion, data analysis, and user interfaces[cite: 1].

* `main.py` — The core execution loop initializing the telemetry listener and AI services[cite: 1].
* `run_engineer.bat` & `test_simulation.bat` — Windows batch scripts for quick launching and testing the application[cite: 1].
* **`ai_engineer/`**
  * `analysis.py` — Contains data analytics and machine learning logic for predicting race strategies[cite: 1].
  * `history.py` — Manages historical race logs and database interactions[cite: 1].
  * `packets.py` & `telemetry_stream.py` — Handle decoding and real-time processing of F1 2021 UDP binary packets[cite: 1].
  * `race_radio.py` — Manages speech-to-text input, F1 radio timing logic, and text-to-speech output[cite: 1].
  * `simulate_telemetry.py` — Enables offline testing by simulating live telemetry without needing the game running[cite: 1].
  * `track_data.py` — Stores track-specific metadata, DRS zones, and AI setup recommendations[cite: 1].
  * `voice_engine.py` — Core voice interface engine[cite: 1].
  * `web_hud.py` — The backend service serving the local F1-TV style web dashboard[cite: 1].
* **`tests/`**
  * Contains unit and integration tests (`test_analysis.py`, `test_hud.py`, `test_race_radio.py`, `test_stream.py`, etc.) to ensure system stability[cite: 1].
  * Includes frontend UI tests like `hud_client_test.js`[cite: 1].

---

## ⚙️ Installation Prerequisites

1. **Python 3.10+** installed on your system.
2. **UDP Telemetry Configuration**: In the F1 2021 game settings, enable UDP Telemetry, set the port to `20777`, set the send rate to 20Hz, and choose the 2021 format.
3. **Local LLM Setup (Optional but Recommended)**: Install Ollama and pull a DeepSeek model for zero-latency, offline AI processing:
   ```bash
   ollama run deepseek-coder-v2-lite

   🛠️ Setup Instructions
Clone the repository and navigate to the project directory:

Bash
git clone [https://github.com/Shryder25/F1-21_Race_Engineer.git](https://github.com/Shryder25/F1-21_Race_Engineer.git)
cd F1-21_Race_Engineer
Install the required dependencies:

Bash
pip install -r ai_engineer/requirements.txt
pip install scikit-learn pandas SpeechRecognition pyttsx3 sqlite3
Run the application using the provided batch file:

Bash
run_engineer.bat
Open the local Web HUD in your browser to monitor telemetry visually.

🎧 Usage Guide
During a Session: The application listens silently to your telemetry, speaking only when a critical event occurs (e.g., sudden weather change, critical tire wear) or when prompted.

Voice Commands: Press your mapped radio button or use the wake word followed by a command:

"Engineer, what is my gap to the car behind?"

"Give me a pace update."

"What is the pit window for these softs?"

HUD Interaction: Access the HUD on a secondary monitor or mobile device to view the Live Track Map, sector timings, and access the Track Setup Advisor before hitting the track.
