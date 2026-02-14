# X32 Mixer Control Project

## Project Overview
This project aims to provide a Python-based interface for viewing and configuring a Behringer X32 audio mixer. It utilizes the `behringer-mixer` library to communicate with the mixer via OSC (Open Sound Control) messages asynchronously.

The primary goal of this iteration is to implement a **Livestream Output Level Stabilizer** using a **PID (Proportional-Integral-Derivative) controller** to automatically maintain a consistent audio output level for a designated livestream mix.

This project is now adapted for deployment on a Raspberry Pi 3 as a headless service, with a web-based user interface (`app.py`) built using Flask for monitoring and control.

### New Feature: X32 Settings Backup & Routing Analysis
Functionality has been added to download a subset of X32 settings to a JSON file for backup and analysis. Furthermore, the system can now analyze these settings to suggest potential livestream output buses or matrices, allowing the user to dynamically select the correct source for stabilization.

## Building and Running

### Prerequisites
*   Python 3.x
*   A Behringer X32 audio mixer connected to the same network as the computer running this script.
*   **For Web UI:** A device with a web browser to access the Flask application.

### Setup
1.  **Clone the repository (if applicable) or navigate to the project directory.**
    ```bash
    cd /path/to/your/project
    ```
2.  **Install the required Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
Before running, you need to update the `MIXER_IP` variable in `mixer_controller.py` with the actual IP address of your Behringer X32 mixer. The `LIVESTREAM_BUS_NUMBER` can now be selected dynamically via the web UI after running an analysis.

The PID controller parameters (`KP`, `KI`, `KD`) are configurable in `mixer_controller.py` and can also be adjusted via the web UI.

### Running the Application

There are two main ways to run the application:

1.  **Running the Mixer Controller Directly (without Web UI):**
    To run only the automatic livestream stabilizer:
    ```bash
    python mixer_controller.py
    ```
    This is useful for initial testing of the PID logic without the web interface overhead.

2.  **Running the Web UI (which will manage the Mixer Controller):**
    To start the Flask web server:
    ```bash
    python app.py
    ```
    Once the Flask server is running, you can access the web interface from a browser by navigating to `http://<Your_Raspberry_Pi_IP_Address>:5000`.

## Development Conventions
*   **Language:** Python 3.x
*   **Mixer Communication:** The `behringer-mixer` library is used for asynchronous OSC communication with the X32 mixer.
*   **Asynchronous Programming:** The project uses `asyncio` for non-blocking communication and handling real-time updates.
*   **Control Logic:** A `simple-pid` controller is implemented for automated gain control to stabilize the livestream audio output.
*   **Web Framework:** Flask is used for building the web-based user interface.
*   **Code Style:** Adhere to standard Python best practices (e.g., PEP 8).

## PID Controller Tuning

The performance of the Livestream Output Level Stabilizer heavily depends on the correct tuning of the PID controller's gains (`KP`, `KI`, `KD`).

*   **`KP` (Proportional Gain):** Affects the speed of response to current errors. A higher `KP` means a stronger, faster reaction. Too high can cause overshoot and oscillation.
*   **`KI` (Integral Gain):** Addresses steady-state errors by summing past errors. Helps eliminate persistent deviations but too high can lead to sluggishness or integrator wind-up.
*   **`KD` (Derivative Gain):** Damps oscillations by considering the rate of change of the error. Can improve stability and reduce overshoot, but too high can make the system sensitive to noise.

**Tuning Strategy (Manual):**

1.  **Start with `KI` and `KD` at zero.**
2.  **Increase `KP`** until the output oscillates around the target. Then, reduce `KP` slightly (e.g., to 50-70% of that value) to reduce oscillations.
3.  **Increase `KI`** to eliminate any steady-state error. Start with small values and increase gradually. Be careful not to make the response too sluggish or cause oscillations.
4.  **Increase `KD`** to reduce overshoot and dampen oscillations. Again, start with small values. Too high a `KD` can make the system sensitive to noise.

This manual tuning process will require running the script (initially with simulated audio or when the mixer is not in live use) and observing its behavior.

## Further Development
*   **Automated PID Tuning:** While currently manual, research and implement auto-tuning methods (e.g., Ziegler-Nichols, relay method) for PID parameters, possibly requiring a dedicated calibration mode.
*   **Dynamic PID Parameters:** Explore varying PID parameters based on the audio content (e.g., different gains for music vs. speech).
*   **Enhanced User Interface:** Develop a more comprehensive GUI with real-time graphs and better controls.
*   **Advanced Mixer Controls:** Implement more extensive control over mixer parameters (sends, routing, effects, scenes/snippets) via the web UI.
*   **Two-Way Communication:** Ensure that changes made directly on the physical X32 mixer are reflected in the web UI.
*   **Input Validation:** Add comprehensive validation for user inputs (e.g., ensuring bus numbers are valid).
*   **Headless Service Deployment:** Provide detailed scripts and instructions for deploying the Flask app and mixer controller as `systemd` services on a Raspberry Pi.
*   **Security:** Implement more robust security for the web interface, especially if exposed to a wider network.
