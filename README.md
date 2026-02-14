# X32 Livestream Output Level Stabilizer

A Python web application that automatically maintains consistent audio output levels on a Behringer X32 digital mixer. Designed for livestream scenarios where input levels vary but the output to a streaming encoder needs to stay steady.

Uses a feedforward control loop with configurable slew rate and meter smoothing to adjust fader levels in real time via OSC, with a browser-based dashboard for monitoring and control. Built for headless deployment on a Raspberry Pi.

## Features

- **Automatic level stabilization** — continuously adjusts a selected output fader so the signal stays at your target level
- **Web dashboard** — real-time level meters, fader position, and status at a glance
- **Configurable output** — monitor any mix bus (1-16), matrix (1-6), or the main stereo output
- **Runtime tuning** — adjust target level, slew rate, meter smoothing, and silence threshold without restarting
- **Silence detection** — freezes the fader when signal drops below a threshold to avoid boosting noise
- **Mixer IP configuration** — set the X32's IP address from the web UI
- **Settings backup** — save a snapshot of mixer settings (channels, buses, matrices) to JSON

## Prerequisites

- Python 3.7+
- A Behringer X32 mixer on the same network
- A web browser to access the dashboard

## Installation

```bash
git clone https://github.com/ispyisail/x32-livestream-stabilizer.git
cd x32-livestream-stabilizer
pip install -r requirements.txt
```

## Usage

Start the web server:

```bash
python app.py
```

Open `http://<your-ip>:5000` in a browser. From the dashboard:

1. **Set the mixer IP** — enter your X32's IP address and click "Set IP"
2. **Start monitoring** — this connects to the mixer and begins the control loop
3. **Select the output** — the dropdown populates with all buses/matrices and their names from the mixer
4. **Set the target level** — the desired output level in dB (e.g., -30 dB)
5. **Tune the stabilizer** — adjust slew rate, smoothing, and silence threshold as needed

To run the control loop without the web UI:

```bash
python mixer_controller.py
```

## Architecture

Three-file application:

| File | Role |
|------|------|
| `mixer_controller.py` | Core logic: connects to X32 via OSC, runs the control loop in a background asyncio thread, handles meter parsing, fader adjustments, and settings backup |
| `app.py` | Flask web server exposing REST endpoints for the dashboard |
| `templates/index.html` | Single-page dashboard with real-time meters (5 Hz polling) and controls |

### How it works

1. **OSC connection** — uses the `behringer-mixer` library to connect and subscribe to mixer state updates
2. **Meter subscription** — opens a raw UDP socket to receive `/meters/2` data at ~50 Hz from the X32
3. **Feedforward control** — every 0.5s, computes `desired_fader = target_output - smoothed_input_level`, applies slew rate limiting, and sends the fader position via OSC
4. **Silence gating** — if the raw meter level drops below the silence threshold, the fader is frozen until signal returns
5. **Async-in-thread** — the mixer event loop runs in a daemon thread; Flask communicates with it via `asyncio.run_coroutine_threadsafe()`

## Configuration Reference

All parameters are adjustable at runtime via the web UI. Defaults are set in `mixer_controller.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIXER_IP` | `192.168.150.10` | X32 mixer IP address |
| `LIVESTREAM_BUS_NUMBER` | `mtx1` | Output to monitor (int 1-16 for buses, `main_st`, `mtx1`-`mtx6`) |
| `TARGET_LEVEL_DB` | `-30.0` | Desired output level in dB |
| `MAX_FADER_SLEW_DB` | `2.0` | Max fader movement per 0.5s cycle (dB). Lower = smoother |
| `METER_SMOOTHING` | `0.15` | Exponential moving average factor (0.05-0.5). Lower = smoother |
| `SIGNAL_THRESHOLD_DB` | `-60.0` | Below this level, fader freezes (silence detection) |
| `MIN_FADER_DB` | `-80.0` | Fader safety floor |
| `MAX_FADER_DB` | `10.0` | Fader safety ceiling |
| `ADJUSTMENT_INTERVAL_SECONDS` | `0.5` | Control loop interval |

## Raspberry Pi Deployment

Full walkthrough for installing on a Raspberry Pi (tested on Pi 3 / Pi 4 with Raspberry Pi OS).

### 1. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

### 2. Clone the project and create a virtual environment

```bash
cd /home/pi
git clone https://github.com/ispyisail/x32-livestream-stabilizer.git x32
cd x32
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Test that it runs

```bash
python app.py
```

Open `http://<raspberry-pi-ip>:5000` from another device on the same network. Press `Ctrl+C` to stop once you've confirmed it works.

### 5. Install as a systemd service

Create the service file:

```bash
sudo nano /etc/systemd/system/x32-stabilizer.service
```

Paste the following:

```ini
[Unit]
Description=X32 Livestream Stabilizer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/x32
ExecStart=/home/pi/x32/venv/bin/gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 app:app
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable x32-stabilizer
sudo systemctl start x32-stabilizer
```

### 6. Managing the service

```bash
# Check status
sudo systemctl status x32-stabilizer

# View live logs
journalctl -u x32-stabilizer -f

# Restart after code changes
sudo systemctl restart x32-stabilizer

# Stop the service
sudo systemctl stop x32-stabilizer
```

### Network notes

- The Raspberry Pi and the X32 must be on the same network (or have a direct Ethernet connection).
- The X32 uses UDP port **10023** for OSC. Make sure no firewall is blocking it.
- If you're using a dedicated network between the Pi and the X32, assign a static IP to the Pi in the same subnet as the mixer (e.g., `192.168.150.x`).

## API Reference

All endpoints return JSON.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML page |
| `GET` | `/status` | Full status (connection, levels, tuning, bus selection) |
| `GET` | `/meter` | Lightweight meter data (level_db, fader_db) for fast polling |
| `GET` | `/output_list` | All mixer outputs with names (requires connection) |
| `POST` | `/start_monitor` | Connect to mixer and start the control loop |
| `POST` | `/stop_monitor` | Stop the control loop and disconnect |
| `POST` | `/set_target_level` | Set target level. Body: `{"target_level_db": -30.0}` |
| `POST` | `/set_tuning` | Set stabilizer params. Body: `{"slew_rate": 2.0, "smoothing": 0.15, "silence_threshold": -60.0}` |
| `POST` | `/set_mixer_ip` | Set mixer IP. Body: `{"ip": "192.168.1.10"}` |
| `POST` | `/set_livestream_bus` | Set monitored output. Body: `{"bus_id": "mtx1"}` |
| `POST` | `/set_fader` | Manually set fader (pauses auto-control for 5s). Body: `{"fader_db": -10.0}` |
| `POST` | `/backup_settings` | Save mixer settings snapshot to JSON file |
| `POST` | `/analyze_routing` | Analyze mixer routing and suggest livestream outputs |

## License

[MIT](LICENSE)
