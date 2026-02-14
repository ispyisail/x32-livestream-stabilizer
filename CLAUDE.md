# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python web application for controlling a Behringer X32 audio mixer. Primary feature is a **Livestream Output Level Stabilizer** that uses a PID controller to automatically maintain consistent audio output levels. Designed for headless deployment on a Raspberry Pi 3.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web UI (Flask server on port 5000)
python app.py

# Run mixer controller standalone (no web UI)
python mixer_controller.py
```

No test framework, linter, or build system is configured. Code follows PEP 8 conventions.

## Architecture

Three-file application:

- **mixer_controller.py** — `MixerStateManager` class. Core logic: connects to X32 via OSC (`behringer-mixer` library), runs a PID control loop (`simple-pid`) in a separate asyncio thread to adjust fader levels, handles settings backup to JSON, and routing analysis.
- **app.py** — Flask web server. Exposes REST endpoints for status polling (`GET /status`), monitoring control (`POST /start_monitor`, `/stop_monitor`), PID tuning (`POST /set_pid`), target level (`POST /set_target_level`), backup (`POST /backup_settings`), routing analysis (`POST /analyze_routing`), and bus selection (`POST /set_livestream_bus`).
- **templates/index.html** — Dashboard with real-time status (2-second polling), level meter visualization, PID tuning controls, and utility functions.

### Key Design Patterns

- **Async-in-thread**: `MixerStateManager` runs its own asyncio event loop in a daemon thread. Flask communicates with it using `asyncio.run_coroutine_threadsafe()`.
- **Event-driven levels**: Audio levels come via OSC subscription callbacks, not polling.
- **PID control loop**: Runs every 0.5 seconds (`ADJUSTMENT_INTERVAL_SECONDS`), with fader safety limits (`MIN_FADER_DB`/`MAX_FADER_DB`).

### Configuration Constants (in mixer_controller.py)

`MIXER_IP`, `LIVESTREAM_BUS_NUMBER`, `TARGET_LEVEL_DB`, `KP`/`KI`/`KD` (PID gains), `ADJUSTMENT_INTERVAL_SECONDS`, `MIN_FADER_DB`/`MAX_FADER_DB`. Most are adjustable at runtime via the web UI.
