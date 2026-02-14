from flask import Flask, render_template, jsonify, request
import logging
import threading
import asyncio
import time
from mixer_controller import MixerStateManager # Import our mixer manager

app = Flask(__name__)

# Initialize the MixerStateManager
mixer_manager = MixerStateManager()

# This event will be used to signal the monitoring thread to stop
stop_monitor_event = threading.Event()

@app.before_request
def check_connection_status():
    """
    Ensure the mixer manager's thread is alive if monitoring was started.
    This is a simple check; more robust health checks might be needed.
    """
    if mixer_manager._thread and not mixer_manager._thread.is_alive() and mixer_manager.mixer_connected:
        # If the thread died unexpectedly but mixer was connected,
        # reset connected status for a clean restart attempt.
        mixer_manager.mixer_connected = False
        mixer_manager.status_message = "Monitoring thread stopped unexpectedly."


@app.route('/')
def index():
    """
    Renders the main dashboard HTML page.
    """
    return render_template('index.html')

@app.route('/status')
def status():
    """
    Returns the current status of the mixer and stabilizer as JSON.
    """
    return jsonify(mixer_manager.get_status())

@app.route('/meter')
def meter():
    """
    Fast lightweight endpoint for real-time meter data.
    Returns only level and fader dB values for high-frequency polling.
    """
    return jsonify({
        "level_db": round(mixer_manager.current_level_db, 2) if mixer_manager.current_level_db is not None else None,
        "fader_db": round(mixer_manager.current_fader_db, 2)
    })

@app.route('/start_monitor', methods=['POST'])
def start_monitor():
    """
    Starts the mixer monitoring thread.
    """
    # If already running, do nothing or return a warning
    if mixer_manager._thread and mixer_manager._thread.is_alive():
        return jsonify({"message": "Mixer monitoring already running.", "status": mixer_manager.get_status()}), 200

    mixer_manager.start_monitoring()
    return jsonify({"message": "Mixer monitoring started (or attempted to start).", "status": mixer_manager.get_status()})

@app.route('/stop_monitor', methods=['POST'])
def stop_monitor():
    """
    Stops the mixer monitoring thread.
    """
    mixer_manager.stop_monitoring()
    return jsonify({"message": "Mixer monitoring stopped.", "status": mixer_manager.get_status()})

@app.route('/set_tuning', methods=['POST'])
def set_tuning():
    """
    Sets stabilizer tuning parameters.
    Expects JSON: {'slew_rate': float, 'smoothing': float, 'silence_threshold': float}
    """
    data = request.get_json()
    try:
        slew_rate = float(data['slew_rate'])
        smoothing = float(data['smoothing'])
        silence_threshold = float(data['silence_threshold'])
        # Validate ranges
        slew_rate = max(0.5, min(10.0, slew_rate))
        smoothing = max(0.05, min(0.5, smoothing))
        silence_threshold = max(-90.0, min(-20.0, silence_threshold))
        mixer_manager.set_tuning(slew_rate, smoothing, silence_threshold)
        return jsonify({"message": "Stabilizer tuning updated.", "status": mixer_manager.get_status()})
    except (TypeError, KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid tuning parameters: {e}"}), 400

@app.route('/set_target_level', methods=['POST'])
def set_target_level():
    """
    Sets a new target level.
    Expects JSON: {'target_level_db': float}
    """
    data = request.get_json()
    try:
        target_level_db = float(data['target_level_db'])
        mixer_manager.set_target_level(target_level_db)
        return jsonify({"message": "Target level updated. Restart monitoring for changes to take effect.", "status": mixer_manager.get_status()})
    except (TypeError, KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid target level: {e}. Expected target_level_db as float."}), 400

@app.route('/backup_settings', methods=['POST'])
def backup_settings():
    """
    Triggers the backup of mixer settings to a JSON file.
    Now synchronous since get_all_settings() reads from local state cache.
    """
    if not mixer_manager.mixer_connected or not mixer_manager._thread or not mixer_manager._thread.is_alive():
        return jsonify({"error": "Mixer not connected or monitoring not active to perform backup."}), 400

    try:
        settings_data = mixer_manager.get_all_settings()

        if "error" in settings_data:
            return jsonify({"error": f"Failed to retrieve settings: {settings_data['error']}"}), 500

        save_result = mixer_manager.save_settings_to_file(settings_data)
        if "error" in save_result:
            return jsonify({"error": f"Failed to save settings: {save_result['error']}"}), 500

        return jsonify({"message": f"Mixer settings backed up successfully. {save_result['message']}"}), 200

    except Exception as e:
        logging.error(f"Error during backup operation: {e}")
        return jsonify({"error": f"An unexpected error occurred during backup: {e}"}), 500

@app.route('/analyze_routing', methods=['POST'])
def analyze_routing():
    """
    Analyzes mixer settings to suggest potential livestream output sources.
    Now synchronous since analyze_livestream_routing() reads from local state cache.
    """
    if not mixer_manager.mixer_connected or not mixer_manager._thread or not mixer_manager._thread.is_alive():
        return jsonify({"error": "Mixer not connected or monitoring not active to perform analysis."}), 400

    try:
        suggestions = mixer_manager.analyze_livestream_routing()
        if "error" in suggestions:
            return jsonify({"error": f"Failed to analyze routing: {suggestions['error']}"}), 500

        return jsonify({"message": "Routing analysis complete.", "suggestions": suggestions}), 200

    except Exception as e:
        logging.error(f"Error during routing analysis: {e}")
        return jsonify({"error": f"An unexpected error occurred during routing analysis: {e}"}), 500

@app.route('/set_mixer_ip', methods=['POST'])
def set_mixer_ip():
    """
    Sets the mixer IP address.
    Expects JSON: {'ip': '192.168.x.x'}
    """
    data = request.get_json()
    try:
        ip = data['ip'].strip()
        # Basic IP format validation
        parts = ip.split('.')
        if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return jsonify({"error": "Invalid IP address format."}), 400
        if mixer_manager.set_mixer_ip(ip):
            return jsonify({"message": f"Mixer IP set to {ip}.", "status": mixer_manager.get_status()})
        else:
            return jsonify({"error": "Cannot change IP while monitoring is running. Stop monitoring first."}), 400
    except (TypeError, KeyError) as e:
        return jsonify({"error": f"Invalid request: {e}. Expected 'ip' as string."}), 400

@app.route('/output_list')
def output_list():
    """
    Returns the list of all mixer outputs with their names.
    Only works when connected.
    """
    if not mixer_manager.mixer_connected:
        return jsonify({"error": "Mixer not connected."}), 400
    result = mixer_manager.get_output_list()
    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 500
    return jsonify({"outputs": result})

@app.route('/set_livestream_bus', methods=['POST'])
def set_livestream_bus():
    """
    Sets the LIVESTREAM_BUS_NUMBER for monitoring.
    Expects JSON: {'bus_id': str or int}
    """
    data = request.get_json()
    try:
        bus_id = data['bus_id']
        # Try to convert to int if it's a number, otherwise keep as string
        try:
            bus_id = int(bus_id)
        except ValueError:
            pass # Keep as string if not a valid int

        mixer_manager.set_livestream_bus_number(bus_id)
        return jsonify({"message": f"Livestream bus set to {bus_id}. Restart monitoring for changes to take effect.", "status": mixer_manager.get_status()}), 200
    except (TypeError, KeyError) as e:
        return jsonify({"error": f"Invalid bus ID: {e}. Expected 'bus_id' as string or int."}), 400


@app.route('/set_fader', methods=['POST'])
def set_fader():
    """
    Manually set the monitored fader level.
    Expects JSON: {'fader_db': float}
    """
    data = request.get_json()
    try:
        fader_db = float(data['fader_db'])
        if mixer_manager.set_fader_level(fader_db):
            return jsonify({"message": f"Fader set to {fader_db:.1f} dB"})
        else:
            return jsonify({"error": "Mixer not connected or command failed."}), 400
    except (TypeError, KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid fader value: {e}"}), 400


if __name__ == '__main__':
    # It's generally not recommended to run Flask with debug=True in production
    # For Raspberry Pi deployment, use a production-ready WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)
