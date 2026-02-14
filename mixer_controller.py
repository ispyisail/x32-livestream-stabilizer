import asyncio
import logging
import math
import struct
import threading
import json
import os
import time

from behringer_mixer import mixer_api


# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _build_fader_db_key(bus_id):
    """
    Build the correct mixer state key for a fader_db value
    from a LIVESTREAM_BUS_NUMBER value.
      int 1-16       -> "/bus/1/mix_fader_db"
      "main_st"      -> "/main/st/mix_fader_db"
      "mtx1" .. "mtx6" -> "/mtx/1/mix_fader_db"
    """
    if isinstance(bus_id, int):
        return f"/bus/{bus_id}/mix_fader_db"
    s = str(bus_id)
    if s == "main_st":
        return "/main/st/mix_fader_db"
    if s.startswith("mtx"):
        n = s[3:]
        return f"/mtx/{n}/mix_fader_db"
    # fallback: treat as bus number string
    return f"/bus/{s}/mix_fader_db"


def _build_meter_index(bus_id):
    """
    Map a LIVESTREAM_BUS_NUMBER value to its index(es) in the /meters/2 blob.
    Returns a list of indices (usually one, two for stereo main).

    /meters/2 layout (70 values):
      0-15  = mix bus 1-16
      16-21 = matrix 1-6
      22-23 = main L/R stereo
      24    = mono/center
    Note: Exact layout may differ — check METER DUMP log on first packet.
    """
    if isinstance(bus_id, int):
        return [bus_id - 1]  # bus 1-16 → index 0-15
    s = str(bus_id)
    if s == "main_st":
        return [22, 23]  # average L and R
    if s.startswith("mtx"):
        n = int(s[3:])
        return [15 + n]  # mtx1→16, mtx2→17 ... mtx6→21
    # fallback: try as int
    return [int(s) - 1]


def x32_float_to_db(f):
    """
    Convert an X32 fader float (0.0 – 1.0) to dB using the X32's
    piecewise-linear mapping. Used for fader positions only.
    Returns a float in the range -90 .. +10.
    """
    if f <= 0.0:
        return -90.0
    if f < 0.0625:
        return f / 0.0625 * 30.0 - 90.0   # -90 .. -60
    if f < 0.25:
        return (f - 0.0625) / 0.1875 * 30.0 - 60.0  # -60 .. -30
    if f < 0.5:
        return (f - 0.25) / 0.25 * 20.0 - 30.0  # -30 .. -10
    if f < 0.75:
        return (f - 0.5) / 0.25 * 10.0 - 10.0  # -10 .. 0
    if f <= 1.0:
        return (f - 0.75) / 0.25 * 10.0  # 0 .. +10
    return 10.0


def meter_float_to_db(f):
    """
    Convert an X32 meter float to dBFS using standard linear-to-dB.
    Meter values are linear amplitude: 1.0 = 0 dBFS, up to 8.0 = +18 dBFS.
    """
    if f <= 0.0:
        return -90.0
    return 20.0 * math.log10(f)


def db_to_x32_float(db):
    """
    Inverse of x32_float_to_db: convert dB (-90..+10) to X32 fader float (0.0..1.0).
    """
    if db <= -90.0:
        return 0.0
    if db < -60.0:
        return (db + 90.0) / 30.0 * 0.0625         # 0.0 .. 0.0625
    if db < -30.0:
        return (db + 60.0) / 30.0 * 0.1875 + 0.0625  # 0.0625 .. 0.25
    if db < -10.0:
        return (db + 30.0) / 20.0 * 0.25 + 0.25     # 0.25 .. 0.5
    if db < 0.0:
        return (db + 10.0) / 10.0 * 0.25 + 0.5      # 0.5 .. 0.75
    if db <= 10.0:
        return db / 10.0 * 0.25 + 0.75              # 0.75 .. 1.0
    return 1.0


def _build_fader_osc_address(bus_id):
    """
    Build the raw X32 OSC address for a fader (float 0-1 scale).
      int 1-16       -> "/bus/01/mix/fader"
      "main_st"      -> "/main/st/mix/fader"
      "mtx1".."mtx6" -> "/mtx/01/mix/fader"
    """
    if isinstance(bus_id, int):
        return f"/bus/{bus_id:02d}/mix/fader"
    s = str(bus_id)
    if s == "main_st":
        return "/main/st/mix/fader"
    if s.startswith("mtx"):
        n = int(s[3:])
        return f"/mtx/{n:02d}/mix/fader"
    return f"/bus/{int(s):02d}/mix/fader"


def _build_osc_message(address, *args):
    """
    Build a raw OSC message as bytes.
    Supports string (s), int32 (i), and float32 (f) arguments.
    """
    # Pad address to 4-byte boundary
    addr_bytes = address.encode('ascii') + b'\x00'
    while len(addr_bytes) % 4 != 0:
        addr_bytes += b'\x00'

    # Build type tag string
    type_tag = ','
    for arg in args:
        if isinstance(arg, str):
            type_tag += 's'
        elif isinstance(arg, int):
            type_tag += 'i'
        elif isinstance(arg, float):
            type_tag += 'f'
    type_bytes = type_tag.encode('ascii') + b'\x00'
    while len(type_bytes) % 4 != 0:
        type_bytes += b'\x00'

    # Encode arguments
    data_bytes = b''
    for arg in args:
        if isinstance(arg, str):
            s = arg.encode('ascii') + b'\x00'
            while len(s) % 4 != 0:
                s += b'\x00'
            data_bytes += s
        elif isinstance(arg, int):
            data_bytes += struct.pack('>i', arg)
        elif isinstance(arg, float):
            data_bytes += struct.pack('>f', arg)

    return addr_bytes + type_bytes + data_bytes


class MeterProtocol(asyncio.DatagramProtocol):
    """
    UDP protocol that receives /meters/2 blobs from the X32,
    parses the target meter indices, and updates the manager's level.
    """
    def __init__(self, manager, meter_indices):
        self._manager = manager
        self._meter_indices = meter_indices
        self.transport = None
        self._dump_count = 0  # log first few meter packets for diagnostics

    def connection_made(self, transport):
        self.transport = transport

    def _parse_blob(self, data):
        """Parse an OSC meter response, returning (count, floats_data) or None."""
        try:
            i = data.index(b'\x00')
            i += (4 - i % 4) if i % 4 != 0 else 4
            j = data.index(b'\x00', i)
            j += (4 - j % 4) if j % 4 != 0 else 4
            blob_size = struct.unpack('>i', data[j:j+4])[0]
            blob_data = data[j+4:j+4+blob_size]
            if len(blob_data) < 4:
                return None
            count = struct.unpack('<i', blob_data[0:4])[0]
            floats_data = blob_data[4:]
            return count, floats_data
        except Exception:
            return None

    def datagram_received(self, data, addr):
        try:
            result = self._parse_blob(data)
            if not result:
                return
            count, floats_data = result

            # Log first packet for diagnostics
            if self._dump_count < 1:
                self._dump_count += 1
                logging.info(f"METER: receiving /meters/2 with {count} values, reading indices {self._meter_indices}")

            values = []
            for idx in self._meter_indices:
                if idx < count:
                    offset = idx * 4
                    if offset + 4 <= len(floats_data):
                        val = struct.unpack('<f', floats_data[offset:offset+4])[0]
                        values.append(val)

            if values:
                avg_float = sum(values) / len(values)
                self._manager.current_level_db = meter_float_to_db(avg_float)
        except Exception as e:
            logging.debug(f"Meter parse error: {e}")

    def error_received(self, exc):
        logging.warning(f"Meter UDP error: {exc}")

    def connection_lost(self, exc):
        logging.debug("Meter UDP connection closed")


class MixerStateManager:
    """
    Manages the state and control loop for the X32 mixer, running its asyncio operations
    in a separate thread.
    """
    def __init__(self):
        self._mixer = None
        self.mixer_connected = False
        self._mixer_task = None
        self._loop = None
        self._thread = None
        self._stop_event = threading.Event()
        self._config_file = "config.json"

        # Configuration Parameters for Livestream Stabilizer
        self.MIXER_IP = "192.168.150.10"
        self.LIVESTREAM_BUS_NUMBER = "mtx1"

        self.TARGET_LEVEL_DB = -30.0     # Desired output level in dB (output = input + fader)

        # PID Controller Parameters (These will need tuning!)
        self.KP = 0.2  # Proportional gain (lower = less aggressive)
        self.KI = 0.02 # Integral gain (slowly corrects steady-state error)
        self.KD = 0.01 # Derivative gain (dampens oscillation)

        # How often to check and adjust the level (also the PID sample time)
        self.ADJUSTMENT_INTERVAL_SECONDS = 0.5

        # Minimum and Maximum fader values in dB for safety
        self.MIN_FADER_DB = -80.0 # Effectively muted
        self.MAX_FADER_DB = 10.0  # Max output

        # Signal threshold: don't adjust fader if meter level is below this
        # (prevents PID from reacting to noise floor / silence)
        self.SIGNAL_THRESHOLD_DB = -60.0

        # Slew rate: max dB the fader can move per adjustment cycle
        # Prevents sudden jumps; lower = smoother. At 0.5s interval, 2.0 means max 4 dB/sec.
        self.MAX_FADER_SLEW_DB = 2.0

        # Meter smoothing: exponential moving average factor (0-1)
        # Lower = smoother but slower to react. 0.15 = 85% old + 15% new each sample.
        self.METER_SMOOTHING = 0.15
        self._smoothed_level_db = None

        # Connection retry parameters
        self.MAX_RETRY_ATTEMPTS = 5
        self.RETRY_DELAY_SECONDS = 5

        # Current status (for web UI)
        self.current_level_db = None  # None until real meter data arrives
        self.current_fader_db = 0.0
        self.status_message = "Not running"
        self.suggested_livestream_output = []

        # The state key we are currently monitoring (set when monitoring starts)
        self._target_fader_key = None
        self._subscribe_task = None
        self._meter_transport = None
        self._meter_keepalive_task = None
        self._fader_osc_address = None
        self._initial_fader_db = None  # saved on monitor start for restore
        self._manual_override_until = 0  # timestamp: PID paused until this time
        self._in_silence = False  # True when signal is below threshold; fader frozen

        # Load saved settings from config file (overrides defaults above)
        self._load_config()

    def _load_config(self):
        """Load persisted settings from config.json, if it exists."""
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r') as f:
                    cfg = json.load(f)
                self.MIXER_IP = cfg.get("mixer_ip", self.MIXER_IP)
                self.LIVESTREAM_BUS_NUMBER = cfg.get("livestream_bus", self.LIVESTREAM_BUS_NUMBER)
                self.TARGET_LEVEL_DB = cfg.get("target_level_db", self.TARGET_LEVEL_DB)
                self.MAX_FADER_SLEW_DB = cfg.get("slew_rate", self.MAX_FADER_SLEW_DB)
                self.METER_SMOOTHING = cfg.get("smoothing", self.METER_SMOOTHING)
                self.SIGNAL_THRESHOLD_DB = cfg.get("silence_threshold", self.SIGNAL_THRESHOLD_DB)
                logging.info(f"Loaded settings from {self._config_file}")
        except Exception as e:
            logging.warning(f"Could not load config: {e}, using defaults")

    def _save_config(self):
        """Persist current settings to config.json."""
        cfg = {
            "mixer_ip": self.MIXER_IP,
            "livestream_bus": self.LIVESTREAM_BUS_NUMBER,
            "target_level_db": self.TARGET_LEVEL_DB,
            "slew_rate": self.MAX_FADER_SLEW_DB,
            "smoothing": self.METER_SMOOTHING,
            "silence_threshold": self.SIGNAL_THRESHOLD_DB,
        }
        try:
            with open(self._config_file, 'w') as f:
                json.dump(cfg, f, indent=2)
            logging.debug(f"Settings saved to {self._config_file}")
        except Exception as e:
            logging.warning(f"Could not save config: {e}")

    def _send_fader_osc(self, fader_db):
        """
        Send a fader SET command directly via raw UDP, bypassing the
        behringer-mixer library (which has a type bug in set_value).
        """
        if not self._meter_transport or not self._fader_osc_address:
            return
        fader_float = db_to_x32_float(fader_db)
        msg = _build_osc_message(self._fader_osc_address, fader_float)
        try:
            self._meter_transport.sendto(msg)
        except Exception as e:
            logging.debug(f"Fader OSC send error: {e}")

    def _fader_update_callback(self, data):
        """
        Callback function for subscribed updates from the mixer.
        Filters for the target bus fader_db key and updates current_fader_db.
        """
        if self._target_fader_key and data.get("property") == self._target_fader_key:
            self.current_fader_db = data["value"]
            logging.debug(f"Fader update callback: {self.current_fader_db:.2f} dB")

    async def _connect_to_mixer_async(self):
        """
        Attempts to connect to the mixer with retries (async version).
        """
        for attempt in range(1, self.MAX_RETRY_ATTEMPTS + 1):
            logging.info(f"Attempting to connect to X32 mixer at {self.MIXER_IP} (Attempt {attempt}/{self.MAX_RETRY_ATTEMPTS})...")
            self.status_message = f"Connecting (Attempt {attempt}/{self.MAX_RETRY_ATTEMPTS})"
            try:
                self._mixer = mixer_api.create("X32", ip=self.MIXER_IP, logLevel=logging.DEBUG)
                connected = await self._mixer.start()
                if connected:
                    logging.info("Successfully connected to the mixer.")
                    self.mixer_connected = True
                    self.status_message = "Connected and running"
                    return True
                else:
                    logging.warning("Mixer connection validation failed.")
            except Exception as e:
                logging.error(f"Connection attempt {attempt} failed: {e}")

            if attempt < self.MAX_RETRY_ATTEMPTS:
                logging.info(f"Retrying connection in {self.RETRY_DELAY_SECONDS} seconds...")
                await asyncio.sleep(self.RETRY_DELAY_SECONDS)

        logging.critical(f"Failed to connect to mixer after {self.MAX_RETRY_ATTEMPTS} attempts.")
        self.status_message = "Failed to connect"
        return False

    async def _monitor_livestream_level_async(self):
        """
        Monitors the livestream output level and adjusts the fader to maintain a target level
        using a PID controller (async version).
        """
        if not self.mixer_connected or not self._mixer:
            logging.error("Mixer not connected. Cannot start monitoring.")
            return

        # Load all initial state from the mixer so mixer.state() has data
        logging.info("Loading initial mixer state...")
        await self._mixer.reload()
        await asyncio.sleep(1)  # Allow time for all responses to arrive
        logging.info(f"Loaded {len(self._mixer.state())} state keys from mixer.")

        self._target_fader_key = _build_fader_db_key(self.LIVESTREAM_BUS_NUMBER)
        self._fader_osc_address = _build_fader_osc_address(self.LIVESTREAM_BUS_NUMBER)
        logging.info(f"Starting livestream level monitor for key '{self._target_fader_key}' (OSC: {self._fader_osc_address})")
        logging.info(f"Target Output: {self.TARGET_LEVEL_DB} dB, Signal Threshold: {self.SIGNAL_THRESHOLD_DB} dB, Slew: {self.MAX_FADER_SLEW_DB} dB/cycle")

        try:
            initial_fader_db = self._mixer.state(self._target_fader_key)
            if initial_fader_db is not None:
                self.current_fader_db = initial_fader_db
                self._initial_fader_db = initial_fader_db
                logging.info(f"Initial fader position: {initial_fader_db:.2f} dB (saved for restore)")
            else:
                logging.warning(f"Could not read initial fader value for '{self._target_fader_key}'")
            self.status_message = "Monitoring active"

            # Subscribe runs an infinite renewal loop, so launch it as a
            # background task. The callback keeps fader position updated in
            # real-time; our PID loop reads from mixer.state() below.
            self._subscribe_task = asyncio.create_task(
                self._mixer.subscribe(self._fader_update_callback)
            )
            logging.info(f"Subscribed to mixer updates, filtering for '{self._target_fader_key}'.")

            # Start real-time meter subscription via raw UDP
            meter_indices = _build_meter_index(self.LIVESTREAM_BUS_NUMBER)
            logging.info(f"Starting meter subscription for indices {meter_indices}")
            loop = asyncio.get_event_loop()
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: MeterProtocol(self, meter_indices),
                remote_addr=(self.MIXER_IP, 10023)
            )
            self._meter_transport = transport

            async def _meter_keepalive():
                """Re-send /meters subscription every 8s (expires after ~10s)."""
                msg = _build_osc_message('/meters', '/meters/2', 0)
                logging.info(f"Meter keepalive: subscribing to /meters/2, msg={msg.hex()}")
                while not self._stop_event.is_set():
                    try:
                        self._meter_transport.sendto(msg)
                        logging.debug("Meter keepalive sent /meters/2 subscription")
                    except Exception as e:
                        logging.warning(f"Meter keepalive send error: {e}")
                    await asyncio.sleep(8)

            self._meter_keepalive_task = asyncio.create_task(_meter_keepalive())
            logging.info("Meter UDP subscription started.")

        except Exception as e:
            logging.error(f"Error during PID setup or subscription: {e}")
            self.status_message = "Error during monitoring setup"
            return

        while not self._stop_event.is_set():
            try:
                # Read the current fader position from state cache
                fader_val = self._mixer.state(self._target_fader_key)
                if fader_val is not None:
                    self.current_fader_db = fader_val

                # Skip PID until real meter data has arrived
                if self.current_level_db is None:
                    logging.info("Waiting for meter data...")
                    await asyncio.sleep(self.ADJUSTMENT_INTERVAL_SECONDS)
                    continue

                # Pause during manual fader override
                if time.time() < self._manual_override_until:
                    await asyncio.sleep(self.ADJUSTMENT_INTERVAL_SECONDS)
                    continue

                # Check for silence using RAW level (not smoothed) for fast detection
                if self.current_level_db < self.SIGNAL_THRESHOLD_DB:
                    if not self._in_silence:
                        logging.info(f"Signal dropped below threshold ({self.current_level_db:.1f} dB), freezing fader at {self.current_fader_db:.1f} dB")
                        self._in_silence = True
                    await asyncio.sleep(self.ADJUSTMENT_INTERVAL_SECONDS)
                    continue

                # Coming out of silence: re-seed the smoothed level before moving fader
                if self._in_silence:
                    logging.info(f"Signal returned ({self.current_level_db:.1f} dB), resuming fader control")
                    self._in_silence = False
                    self._smoothed_level_db = self.current_level_db

                # Smooth the meter reading to reduce noise-driven jitter
                if self._smoothed_level_db is None:
                    self._smoothed_level_db = self.current_level_db
                else:
                    alpha = self.METER_SMOOTHING
                    self._smoothed_level_db = alpha * self.current_level_db + (1 - alpha) * self._smoothed_level_db

                # Feedforward: fader = target_output - input_level
                desired_fader_db = self.TARGET_LEVEL_DB - self._smoothed_level_db
                desired_fader_db = max(self.MIN_FADER_DB, min(self.MAX_FADER_DB, desired_fader_db))

                # Apply slew rate limit: cap how much fader moves per cycle
                fader_delta = desired_fader_db - self.current_fader_db
                if abs(fader_delta) > self.MAX_FADER_SLEW_DB:
                    desired_fader_db = self.current_fader_db + self.MAX_FADER_SLEW_DB * (1 if fader_delta > 0 else -1)

                est_output = self._smoothed_level_db + desired_fader_db
                logging.info(f"Monitor - Input: {self._smoothed_level_db:.1f} dB, Target: {self.TARGET_LEVEL_DB:.1f} dB, Fader: {self.current_fader_db:.1f} -> {desired_fader_db:.1f} dB, Est.Output: {est_output:.1f} dB")

                # Only move fader if change is significant (deadband)
                if abs(desired_fader_db - self.current_fader_db) > 0.5:
                    self._send_fader_osc(desired_fader_db)
                    self.current_fader_db = desired_fader_db

            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                self.status_message = f"Error: {e}"

            await asyncio.sleep(self.ADJUSTMENT_INTERVAL_SECONDS)

        logging.info("Livestream monitoring loop exited.")

    async def _cleanup_mixer(self):
        """Clean up mixer connection and reset state."""
        try:
            if self._meter_keepalive_task and not self._meter_keepalive_task.done():
                self._meter_keepalive_task.cancel()
            if self._meter_transport:
                self._meter_transport.close()
                self._meter_transport = None
        except Exception as e:
            logging.debug(f"Meter cleanup error: {e}")
        try:
            if self._mixer:
                await self._mixer.unsubscribe()
                if self._subscribe_task and not self._subscribe_task.done():
                    self._subscribe_task.cancel()
                await self._mixer.stop()
        except Exception as e:
            logging.debug(f"Mixer cleanup error: {e}")
        self._mixer = None
        self.mixer_connected = False
        self.current_level_db = None
        self._smoothed_level_db = None

    async def _run_mixer_loop_async(self):
        """
        Main asynchronous function to connect and run the monitor.
        Automatically reconnects if the mixer goes offline, unless the
        user has manually stopped monitoring via stop_monitoring().
        """
        RECONNECT_DELAY = 10  # seconds between reconnect attempts

        while not self._stop_event.is_set():
            if await self._connect_to_mixer_async():
                try:
                    await self._monitor_livestream_level_async()
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                finally:
                    await self._cleanup_mixer()

            # If user manually stopped, don't reconnect
            if self._stop_event.is_set():
                self.status_message = "Stopped"
                break

            # Connection lost or failed — wait and retry
            self.status_message = f"Reconnecting in {RECONNECT_DELAY}s..."
            logging.info(f"Will attempt to reconnect in {RECONNECT_DELAY} seconds...")
            for _ in range(RECONNECT_DELAY):
                if self._stop_event.is_set():
                    self.status_message = "Stopped"
                    return
                await asyncio.sleep(1)

        self.status_message = "Stopped"

    def _start_mixer_thread(self):
        """
        Starts the asyncio event loop in a separate thread.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_mixer_loop_async())

    def start_monitoring(self):
        """
        Public method to start the mixer monitoring in a new thread.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("Mixer monitoring already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._start_mixer_thread, daemon=True)
        self._thread.start()
        self.status_message = "Attempting to start monitoring"
        logging.info("Mixer monitoring thread started.")

    def stop_monitoring(self):
        """
        Public method to stop the mixer monitoring thread.
        """
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logging.warning("Mixer monitoring thread did not stop gracefully.")
            else:
                logging.info("Mixer monitoring thread stopped.")
        else:
            logging.info("Mixer monitoring not running.")
        self.status_message = "Stopped"

    def get_all_settings(self):
        """
        Downloads a subset of common X32 mixer settings using mixer.state().
        Returns a dictionary of settings. This is synchronous since mixer.state()
        reads from the local state cache.
        """
        if not self.mixer_connected or not self._mixer:
            logging.error("Mixer not connected. Cannot download settings.")
            return {"error": "Mixer not connected"}

        settings = {}
        try:
            # Main ST (stereo)
            settings['main_st'] = {
                'fader': self._mixer.state("/main/st/mix_fader_db"),
                'mute': self._mixer.state("/main/st/mix_on"),
                'name': "Main ST",
            }

            # Channels 1-32
            settings['channels'] = {}
            for i in range(1, 33):
                settings['channels'][f'ch{i}'] = {
                    'fader': self._mixer.state(f"/ch/{i}/mix_fader_db"),
                    'mute': self._mixer.state(f"/ch/{i}/mix_on"),
                    'name': self._mixer.state(f"/ch/{i}/config_name"),
                }

            # Mix Buses 1-16
            settings['mix_buses'] = {}
            for i in range(1, 17):
                settings['mix_buses'][f'bus{i}'] = {
                    'fader': self._mixer.state(f"/bus/{i}/mix_fader_db"),
                    'mute': self._mixer.state(f"/bus/{i}/mix_on"),
                    'name': self._mixer.state(f"/bus/{i}/config_name"),
                }

            # Matrices 1-6
            settings['matrices'] = {}
            for i in range(1, 7):
                settings['matrices'][f'mtx{i}'] = {
                    'fader': self._mixer.state(f"/mtx/{i}/mix_fader_db"),
                    'mute': self._mixer.state(f"/mtx/{i}/mix_on"),
                    'name': self._mixer.state(f"/mtx/{i}/config_name"),
                }

            logging.info("Successfully retrieved mixer settings.")
            return settings

        except Exception as e:
            logging.error(f"Error retrieving mixer settings: {e}")
            return {"error": f"Error retrieving settings: {e}"}

    def analyze_livestream_routing(self):
        """
        Analyzes mixer settings to suggest potential livestream output sources.
        """
        settings = self.get_all_settings()
        if "error" in settings:
            return {"error": settings["error"]}

        suggestions = []

        # Check Main ST
        main_fader = settings['main_st']['fader']
        if main_fader is not None and main_fader > self.MIN_FADER_DB:
            suggestions.append({"type": "Main ST", "id": "main_st", "name": "Main ST", "fader_db": main_fader})

        # Check Mix Buses
        for bus_id in range(1, 17):
            bus_key = f'bus{bus_id}'
            if bus_key in settings['mix_buses']:
                bus_data = settings['mix_buses'][bus_key]
                fader = bus_data['fader']
                if fader is not None and fader > self.MIN_FADER_DB:
                    name = bus_data['name']
                    has_custom_name = name and name != f'Bus {bus_id}'
                    if has_custom_name and "stream" in name.lower():
                        suggestions.insert(0, {"type": "Mix Bus", "id": bus_id, "name": name, "fader_db": fader})
                    elif has_custom_name:
                        suggestions.append({"type": "Mix Bus", "id": bus_id, "name": name, "fader_db": fader})
                    else:
                        suggestions.append({"type": "Mix Bus", "id": bus_id, "fader_db": fader})

        # Check Matrices
        for mtx_id in range(1, 7):
            mtx_key = f'mtx{mtx_id}'
            if mtx_key in settings['matrices']:
                mtx_data = settings['matrices'][mtx_key]
                fader = mtx_data['fader']
                if fader is not None and fader > self.MIN_FADER_DB:
                    name = mtx_data['name']
                    has_custom_name = name and name != f'Matrix {mtx_id}'
                    if has_custom_name and "stream" in name.lower():
                        suggestions.insert(0, {"type": "Matrix", "id": f"mtx{mtx_id}", "name": name, "fader_db": fader})
                    elif has_custom_name:
                        suggestions.append({"type": "Matrix", "id": f"mtx{mtx_id}", "name": name, "fader_db": fader})
                    else:
                        suggestions.append({"type": "Matrix", "id": f"mtx{mtx_id}", "fader_db": fader})

        logging.info(f"Livestream routing analysis suggestions: {suggestions}")
        self.suggested_livestream_output = suggestions
        return suggestions


    def get_status(self):
        """
        Returns a dictionary of the current mixer status.
        """
        return {
            "connected": self.mixer_connected,
            "status_message": self.status_message,
            "current_level_db": f"{self.current_level_db:.2f} dB" if self.current_level_db is not None else "Waiting...",
            "current_fader_db": f"{self.current_fader_db:.2f} dB",
            "target_level_db": f"{self.TARGET_LEVEL_DB:.2f} dB",
            "slew_rate": self.MAX_FADER_SLEW_DB,
            "smoothing": self.METER_SMOOTHING,
            "silence_threshold": self.SIGNAL_THRESHOLD_DB,
            "livestream_bus": self.LIVESTREAM_BUS_NUMBER,
            "mixer_ip": self.MIXER_IP,
            "suggested_livestream_output": self.suggested_livestream_output
        }

    def set_tuning(self, slew_rate, smoothing, silence_threshold):
        self.MAX_FADER_SLEW_DB = slew_rate
        self.METER_SMOOTHING = smoothing
        self.SIGNAL_THRESHOLD_DB = silence_threshold
        self._save_config()
        logging.info(f"Stabilizer tuning updated: slew={slew_rate}, smoothing={smoothing}, threshold={silence_threshold}")

    def set_target_level(self, target_level_db):
        self.TARGET_LEVEL_DB = target_level_db
        self._save_config()
        logging.info(f"Target level updated to {target_level_db} dB. Restart monitoring for changes to take effect.")

    def set_fader_level(self, fader_db):
        """
        Manually set the monitored fader to a specific dB value.
        Pauses PID control for 5 seconds so it doesn't immediately override.
        """
        fader_db = max(self.MIN_FADER_DB, min(self.MAX_FADER_DB, fader_db))
        if not self.mixer_connected or not self._meter_transport:
            logging.warning("Cannot set fader: mixer not connected or meter transport not available.")
            return False
        try:
            self._send_fader_osc(fader_db)
            self.current_fader_db = fader_db
            self._manual_override_until = time.time() + 5
            logging.info(f"Manually set fader to {fader_db:.2f} dB (PID paused 5s)")
            return True
        except Exception as e:
            logging.error(f"Error setting fader: {e}")
            return False

    def set_livestream_bus_number(self, bus_id):
        """
        Sets the LIVESTREAM_BUS_NUMBER dynamically.
        Accepts: int (bus 1-16), "main_st", "mtx1" through "mtx6"
        """
        self.LIVESTREAM_BUS_NUMBER = bus_id
        self._save_config()
        logging.info(f"Livestream monitoring bus/matrix set to: {bus_id} (key: {_build_fader_db_key(bus_id)}). Restart monitoring for changes to take effect.")

    def set_mixer_ip(self, ip):
        """
        Sets the mixer IP address at runtime.
        Only works when monitoring is stopped.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("Cannot change IP while monitoring is running.")
            return False
        self.MIXER_IP = ip
        self._save_config()
        logging.info(f"Mixer IP set to: {ip}")
        return True

    def get_output_list(self):
        """
        Returns a list of all possible monitored outputs with names from mixer state.
        Includes Main ST, Bus 1-16, Matrix 1-6 (always all 23).
        """
        if not self.mixer_connected or not self._mixer:
            return {"error": "Mixer not connected"}

        outputs = []
        try:
            # Main ST
            name = "Main ST"
            fader = self._mixer.state("/main/st/mix_fader_db")
            outputs.append({
                "id": "main_st",
                "type": "Main",
                "name": name,
                "fader_db": fader if fader is not None else 0.0
            })

            # Bus 1-16
            for i in range(1, 17):
                custom_name = self._mixer.state(f"/bus/{i}/config_name")
                display_name = custom_name if custom_name else f"Bus {i}"
                fader = self._mixer.state(f"/bus/{i}/mix_fader_db")
                outputs.append({
                    "id": i,
                    "type": "Bus",
                    "name": display_name,
                    "fader_db": fader if fader is not None else 0.0
                })

            # Matrix 1-6
            for i in range(1, 7):
                custom_name = self._mixer.state(f"/mtx/{i}/config_name")
                display_name = custom_name if custom_name else f"Matrix {i}"
                fader = self._mixer.state(f"/mtx/{i}/mix_fader_db")
                outputs.append({
                    "id": f"mtx{i}",
                    "type": "Matrix",
                    "name": display_name,
                    "fader_db": fader if fader is not None else 0.0
                })

            return outputs

        except Exception as e:
            logging.error(f"Error getting output list: {e}")
            return {"error": f"Error getting output list: {e}"}


# When mixer_controller.py is run directly, it should still start the monitoring
if __name__ == "__main__":
    manager = MixerStateManager()
    try:
        manager.start_monitoring()
        # Keep the main thread alive for a long time to allow the daemon thread to run
        while manager._thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Main program interrupted. Stopping mixer monitoring.")
    finally:
        manager.stop_monitoring()
        logging.info("Application shut down.")
