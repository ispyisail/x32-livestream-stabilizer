"""
Simulator for the X32 Livestream Stabilizer fader algorithm.
Extended test suite covering realistic church/livestream scenarios.

Run:  python simulator.py
"""

import math
import random

# -- Algorithm parameters (mirrors mixer_controller.py defaults) ----------
TARGET_LEVEL_DB = -42.0
ADJUSTMENT_INTERVAL = 0.5      # seconds per cycle
FAST_EMA_SECONDS = 1.0
AVERAGING_WINDOW_SECONDS = 15.0 # slow EMA time constant
SCENE_CHANGE_DB = 4.0
STABLE_DEADBAND_DB = 2.0
SCENE_DEADBAND_DB = 0.5
MAX_FADER_SLEW_DB = 1.5        # stable-mode slew limit
EMA_PULL_RATE = 0.2            # how fast slow EMA catches up during scene change
SIGNAL_THRESHOLD_DB = -50.0
MIN_FADER_DB = -80.0
MAX_FADER_DB = 10.0


# -- Extended scenario definition -----------------------------------------
# Each segment: dict with keys:
#   duration  - seconds
#   base      - base level in dB
#   dynamics  - random fluctuation range (gaussian sigma)
#   label     - display name
#   drift     - optional: dB drift over the segment (speaker moving)
#   fade_in   - optional: seconds to fade in from silence
#   fade_out  - optional: seconds to fade out to silence
#   spikes    - optional: list of (time_offset, spike_db, duration) for coughs/plosives

SCENARIOS = [
    # --- Pre-service ---
    {"duration": 30, "base": -26.0, "dynamics": 1.5, "label": "Pre-service music"},
    {"duration": 3,  "base": -70.0, "dynamics": 1.0, "label": "Silence (music stops)"},

    # --- Welcome ---
    {"duration": 20, "base": -28.0, "dynamics": 2.5, "label": "Welcome speaker",
     "spikes": [(3.0, 15.0, 0.3), (8.0, 12.0, 0.2)]},  # mic pops

    # --- Worship set (loud, dynamic) ---
    {"duration": 5,  "base": -14.0, "dynamics": 3.0, "label": "Worship band - intro",
     "fade_in": 3.0},
    {"duration": 40, "base": -14.0, "dynamics": 4.0, "label": "Worship band - full"},
    {"duration": 20, "base": -22.0, "dynamics": 2.0, "label": "Soft worship song"},
    {"duration": 10, "base": -14.0, "dynamics": 4.0, "label": "Worship band - big finish"},
    {"duration": 4,  "base": -14.0, "dynamics": 3.0, "label": "Band fading out",
     "fade_out": 3.5},

    # --- Sermon ---
    {"duration": 2,  "base": -70.0, "dynamics": 1.0, "label": "Silence (transition)"},
    {"duration": 60, "base": -30.0, "dynamics": 2.5, "label": "Pastor speaking",
     "drift": -4.0,  # slowly gets quieter (moves from mic)
     "spikes": [(15.0, 10.0, 0.3), (35.0, 12.0, 0.2)]},  # occasional emphasis
    {"duration": 3,  "base": -70.0, "dynamics": 0.5, "label": "Dramatic pause"},
    {"duration": 30, "base": -25.0, "dynamics": 3.0, "label": "Pastor speaking (animated)"},

    # --- Video playback ---
    {"duration": 2,  "base": -70.0, "dynamics": 1.0, "label": "Silence (to video)"},
    {"duration": 25, "base": -20.0, "dynamics": 1.0, "label": "Video playback"},
    {"duration": 2,  "base": -70.0, "dynamics": 1.0, "label": "Silence (from video)"},

    # --- Different speaker ---
    {"duration": 30, "base": -35.0, "dynamics": 2.0, "label": "Quiet guest speaker",
     "drift": 3.0},  # slowly gets closer to mic
    {"duration": 1,  "base": -70.0, "dynamics": 0.5, "label": "Silence (handoff)"},
    {"duration": 20, "base": -22.0, "dynamics": 2.5, "label": "Announcements speaker"},

    # --- Offering / background music ---
    {"duration": 30, "base": -28.0, "dynamics": 1.5, "label": "Offering music"},

    # --- Closing worship ---
    {"duration": 5,  "base": -15.0, "dynamics": 3.0, "label": "Closing song - starts",
     "fade_in": 2.0},
    {"duration": 30, "base": -15.0, "dynamics": 4.5, "label": "Closing song - full"},
    {"duration": 8,  "base": -15.0, "dynamics": 3.0, "label": "Closing song - ends",
     "fade_out": 6.0},

    # --- Post-service ---
    {"duration": 3,  "base": -70.0, "dynamics": 1.0, "label": "Silence (end)"},
    {"duration": 20, "base": -32.0, "dynamics": 2.0, "label": "Post-service chat"},
]


def generate_meter_readings(scenarios=None):
    """Generate simulated meter readings from the scenario timeline."""
    if scenarios is None:
        scenarios = SCENARIOS
    readings = []
    t = 0.0
    for seg in scenarios:
        duration = seg["duration"]
        base = seg["base"]
        dynamics = seg["dynamics"]
        label = seg["label"]
        drift = seg.get("drift", 0.0)
        fade_in = seg.get("fade_in", 0.0)
        fade_out = seg.get("fade_out", 0.0)
        spikes = seg.get("spikes", [])

        segment_start = t
        while t - segment_start < duration:
            elapsed = t - segment_start
            progress = elapsed / duration if duration > 0 else 1.0

            # Base level with drift
            level = base + drift * progress

            # Fade in (from silence)
            if fade_in > 0 and elapsed < fade_in:
                fade_progress = elapsed / fade_in
                level = -70.0 + (level - (-70.0)) * fade_progress

            # Fade out (to silence)
            remaining = duration - elapsed
            if fade_out > 0 and remaining < fade_out:
                fade_progress = remaining / fade_out
                level = -70.0 + (level - (-70.0)) * fade_progress

            # Normal dynamics (gaussian noise)
            noise = random.gauss(0, dynamics * 0.5)
            level += noise

            # Spikes (coughs, plosives, emphasis)
            for spike_time, spike_db, spike_dur in spikes:
                if abs(elapsed - spike_time) < spike_dur:
                    level += spike_db

            readings.append((t, level, label))
            t += ADJUSTMENT_INTERVAL
    return readings


def run_new_algorithm(readings):
    """Run the adaptive dual-EMA algorithm (v2 with fixes)."""
    fast_ema = None
    slow_ema = None
    fader_db = 0.0
    in_silence = False
    silence_count = 0
    SILENCE_DEBOUNCE = 6

    results = []
    dt = ADJUSTMENT_INTERVAL
    fast_alpha = 1.0 - math.exp(-dt / FAST_EMA_SECONDS)
    slow_alpha = 1.0 - math.exp(-dt / max(AVERAGING_WINDOW_SECONDS, 0.5))

    for t, level_db, label in readings:
        if level_db < SIGNAL_THRESHOLD_DB:
            silence_count += 1
            if silence_count >= SILENCE_DEBOUNCE:
                in_silence = True
            output_db = level_db + fader_db
            state = "silence" if in_silence else "below"
            results.append((t, level_db, fader_db, output_db, label, state, 0.0))
            continue

        if silence_count > 0 or in_silence:
            fast_ema = level_db
            slow_ema = level_db
            # Jump fader immediately on silence exit (like cold start)
            fader_db = TARGET_LEVEL_DB - level_db
            fader_db = max(MIN_FADER_DB, min(MAX_FADER_DB, fader_db))
            in_silence = False
        silence_count = 0

        if fast_ema is None:
            fast_ema = level_db
            slow_ema = level_db
            fader_db = TARGET_LEVEL_DB - level_db
            fader_db = max(MIN_FADER_DB, min(MAX_FADER_DB, fader_db))
            output_db = level_db + fader_db
            results.append((t, level_db, fader_db, output_db, label, "COLDSTART", 0.0))
            continue

        fast_ema += fast_alpha * (level_db - fast_ema)
        slow_ema += slow_alpha * (level_db - slow_ema)

        change_mag = abs(fast_ema - slow_ema)
        is_scene = change_mag > SCENE_CHANGE_DB

        if is_scene:
            control_level = fast_ema
            slow_ema += EMA_PULL_RATE * (fast_ema - slow_ema)
        else:
            control_level = slow_ema

        desired = TARGET_LEVEL_DB - control_level
        desired = max(MIN_FADER_DB, min(MAX_FADER_DB, desired))

        diff = desired - fader_db
        output_error = abs((level_db + fader_db) - TARGET_LEVEL_DB)

        if is_scene:
            deadband = SCENE_DEADBAND_DB
            max_slew = 20.0
        elif output_error > 5.0:
            deadband = STABLE_DEADBAND_DB
            max_slew = MAX_FADER_SLEW_DB * 5.0
        else:
            deadband = STABLE_DEADBAND_DB
            max_slew = MAX_FADER_SLEW_DB

        if abs(diff) > deadband:
            if abs(diff) > max_slew:
                diff = max_slew if diff > 0 else -max_slew
            fader_db += diff
            fader_db = max(MIN_FADER_DB, min(MAX_FADER_DB, fader_db))

        output_db = level_db + fader_db
        state = "SCENE" if is_scene else "stable"
        results.append((t, level_db, fader_db, output_db, label, state, change_mag))

    return results


def run_old_algorithm(readings):
    """Run the old batch-window algorithm for comparison."""
    fader_db = 0.0
    samples = []
    last_avg_time = 0.0
    in_silence = False
    silence_count = 0
    SILENCE_DEBOUNCE = 6

    results = []

    for t, level_db, label in readings:
        if level_db < SIGNAL_THRESHOLD_DB:
            silence_count += 1
            if silence_count >= SILENCE_DEBOUNCE:
                in_silence = True
            if in_silence:
                output_db = level_db + fader_db
                results.append((t, level_db, fader_db, output_db, label, "silence", 0.0))
                continue
        else:
            silence_count = 0
            if in_silence:
                in_silence = False

        if level_db >= SIGNAL_THRESHOLD_DB:
            samples.append((t, level_db))

        if t - last_avg_time < AVERAGING_WINDOW_SECONDS:
            output_db = level_db + fader_db
            results.append((t, level_db, fader_db, output_db, label, "waiting", 0.0))
            continue

        cutoff = t - AVERAGING_WINDOW_SECONDS
        samples = [(st, sl) for st, sl in samples if st >= cutoff]
        if not samples:
            output_db = level_db + fader_db
            results.append((t, level_db, fader_db, output_db, label, "no_samples", 0.0))
            continue

        linear = [10.0 ** (s[1] / 20.0) for s in samples]
        avg_linear = sum(linear) / len(linear)
        avg_db = 20.0 * math.log10(avg_linear) if avg_linear > 0 else -90.0
        last_avg_time = t

        desired = TARGET_LEVEL_DB - avg_db
        desired = max(MIN_FADER_DB, min(MAX_FADER_DB, desired))

        if abs(desired - fader_db) > 1.0:
            fader_db = desired

        output_db = level_db + fader_db
        results.append((t, level_db, fader_db, output_db, label, "adjusted", 0.0))

    return results


# -- Analysis helpers -----------------------------------------------------

def segment_stats(results, target):
    """Compute per-segment statistics, using the settled (last 50%) portion."""
    segments = {}
    for r in results:
        label = r[4]
        if label not in segments:
            segments[label] = []
        segments[label].append(r)

    stats = []
    for label, data in segments.items():
        active = [r for r in data if r[5] not in ("silence", "below")]
        if len(active) < 4:
            stats.append({"label": label, "n": 0})
            continue

        settled = active[len(active)//2:]
        outputs = [r[3] for r in settled]
        faders = [r[2] for r in settled]
        errors = [o - target for o in outputs]

        avg_err = sum(errors) / len(errors)
        max_err = max(abs(e) for e in errors)
        std_err = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        fader_moves = sum(abs(faders[i] - faders[i-1]) for i in range(1, len(faders)))
        within_3db = sum(1 for e in errors if abs(e) <= 3.0) / len(errors) * 100

        # Response time: first reading within 3dB of target
        first_t = active[0][0]
        settle_time = None
        for r in active:
            if abs(r[3] - target) <= 3.0:
                settle_time = r[0] - first_t
                break

        stats.append({
            "label": label,
            "n": len(settled),
            "avg_err": avg_err,
            "max_err": max_err,
            "std_err": std_err,
            "fader_wobble": fader_moves,
            "within_3db": within_3db,
            "settle_time": settle_time,
        })
    return stats


def worst_spikes(results, target, n=10):
    """Find the N worst output level spikes (biggest error from target)."""
    active = [(r[0], r[3], r[3]-target, r[4], r[5]) for r in results if r[5] not in ("silence", "below")]
    active.sort(key=lambda x: abs(x[2]), reverse=True)
    return active[:n]


def fader_movements(results, min_delta=1.0):
    """Find all fader movements larger than min_delta."""
    moves = []
    prev_fader = None
    for r in results:
        if prev_fader is not None:
            delta = r[2] - prev_fader
            if abs(delta) >= min_delta:
                moves.append((r[0], prev_fader, r[2], delta, r[4], r[5]))
        prev_fader = r[2]
    return moves


def print_report(old_results, new_results, target):
    """Print comprehensive comparison report."""
    total_time = new_results[-1][0] if new_results else 0
    print(f"\n{'='*100}")
    print(f"  FULL TEST REPORT  |  Duration: {total_time:.0f}s ({total_time/60:.1f} min)  |  Target: {target} dB")
    print(f"{'='*100}")

    # Overall stats
    for name, results in [("OLD", old_results), ("NEW", new_results)]:
        active = [r for r in results if r[5] not in ("silence", "below")]
        outputs = [r[3] for r in active]
        faders = [r[2] for r in active]

        if not outputs:
            continue

        avg_out = sum(outputs) / len(outputs)
        std_out = (sum((o - target)**2 for o in outputs) / len(outputs)) ** 0.5
        within_3 = sum(1 for o in outputs if abs(o - target) <= 3.0) / len(outputs) * 100
        within_6 = sum(1 for o in outputs if abs(o - target) <= 6.0) / len(outputs) * 100
        worst = max(abs(o - target) for o in outputs)
        total_move = sum(abs(faders[i] - faders[i-1]) for i in range(1, len(faders)))
        scene_count = sum(1 for r in results if r[5] == "SCENE")

        print(f"\n  --- {name} Algorithm ---")
        print(f"  Output avg error:    {avg_out - target:+.1f} dB")
        print(f"  Output std dev:      {std_out:.1f} dB")
        print(f"  Within +/-3dB:       {within_3:.1f}%")
        print(f"  Within +/-6dB:       {within_6:.1f}%")
        print(f"  Worst spike:         {worst:.1f} dB from target")
        print(f"  Total fader travel:  {total_move:.1f} dB")
        if name == "NEW":
            print(f"  Scene changes:       {scene_count} cycles")

    # Per-segment comparison
    print(f"\n  {'-'*96}")
    print(f"  PER-SEGMENT BREAKDOWN (settled portion)")
    print(f"  {'-'*96}")
    print(f"  {'Segment':40s} {'Algo':4s} {'Err':>6s} {'Max':>6s} {'Std':>5s} {'<3dB':>6s} {'Settle':>7s} {'Wobble':>7s}")
    print(f"  {'-'*96}")

    old_stats = segment_stats(old_results, target)
    new_stats = segment_stats(new_results, target)

    for os_, ns in zip(old_stats, new_stats):
        if os_["n"] == 0 and ns["n"] == 0:
            print(f"  {os_['label']:40s}  (silence / below threshold)")
            continue
        for algo, s in [("OLD", os_), ("NEW", ns)]:
            if s["n"] == 0:
                continue
            settle = f"{s['settle_time']:.1f}s" if s["settle_time"] is not None else "never"
            print(f"  {s['label'] if algo == 'OLD' else '':40s} {algo:4s} "
                  f"{s['avg_err']:+5.1f}  {s['max_err']:5.1f}  {s['std_err']:4.1f}  "
                  f"{s['within_3db']:5.1f}% {settle:>7s}  {s['fader_wobble']:5.1f}dB")

    # Worst spikes
    print(f"\n  {'-'*96}")
    print(f"  TOP 10 WORST OUTPUT SPIKES")
    print(f"  {'-'*96}")
    for name, results in [("OLD", old_results), ("NEW", new_results)]:
        print(f"\n  {name}:")
        spikes = worst_spikes(results, target)
        for t, output, err, label, state in spikes:
            print(f"    {t:6.1f}s  output={output:+6.1f}  error={err:+6.1f}  [{label}] ({state})")

    # Major fader movements
    print(f"\n  {'-'*96}")
    print(f"  MAJOR FADER MOVEMENTS (> 3dB)")
    print(f"  {'-'*96}")
    for name, results in [("OLD", old_results), ("NEW", new_results)]:
        moves = fader_movements(results, min_delta=3.0)
        print(f"\n  {name}: {len(moves)} large movements")
        for t, before, after, delta, label, state in moves[:20]:
            print(f"    {t:6.1f}s  {before:+6.1f} -> {after:+6.1f}  ({delta:+6.1f})  [{label}] ({state})")


def make_charts(old_results, new_results, target, filename="simulator_results.png"):
    """Generate matplotlib charts for visual comparison."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("\n  (Install matplotlib for graphical charts: pip install matplotlib)")
        return

    total_time = new_results[-1][0]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f"X32 Livestream Stabilizer  --  Full Service Test ({total_time/60:.0f} min)",
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    gs = GridSpec(4, 2, hspace=0.4, wspace=0.2,
                  height_ratios=[1, 1, 1.2, 0.6], top=0.95, bottom=0.05, left=0.06, right=0.97)

    for col, (name, results) in enumerate([
        ("OLD (batch window)", old_results),
        ("NEW (adaptive dual-EMA)", new_results)
    ]):
        times = [r[0] for r in results]
        levels = [r[1] for r in results]
        faders = [r[2] for r in results]
        outputs = [r[3] for r in results]
        states = [r[5] for r in results]

        # Segment boundaries for background shading
        segments = []
        prev_label = ""
        for r in results:
            if r[4] != prev_label:
                segments.append((r[0], r[4]))
                prev_label = r[4]

        def shade_segments(ax):
            colors = ['#1a1a2e', '#16213e']
            for i, (seg_t, seg_label) in enumerate(segments):
                end_t = segments[i+1][0] if i+1 < len(segments) else total_time
                ax.axvspan(seg_t, end_t, alpha=0.3, color=colors[i % 2], zorder=0)

        def style_ax(ax, title):
            ax.set_facecolor('#0d1117')
            ax.set_title(title, fontsize=10, color='white', pad=8)
            ax.tick_params(colors='#888')
            ax.set_ylabel("dB", color='#888')
            for spine in ax.spines.values():
                spine.set_color('#333')
            ax.grid(True, alpha=0.15, color='#444')
            ax.set_xlim(0, total_time)

        # Row 0: Input level
        ax1 = fig.add_subplot(gs[0, col])
        ax1.plot(times, levels, color='#4CAF50', linewidth=0.4, alpha=0.8)
        ax1.axhline(y=SIGNAL_THRESHOLD_DB, color='#666', linestyle=':', alpha=0.5)
        ax1.set_ylim(-75, 0)
        style_ax(ax1, f"{name} -- Input Level")
        shade_segments(ax1)

        # Segment labels on top
        for i, (seg_t, seg_label) in enumerate(segments):
            if "silence" not in seg_label.lower() and "pause" not in seg_label.lower():
                ax1.text(seg_t + 1, -3, seg_label, fontsize=5, rotation=60,
                         va='bottom', ha='left', color='#aaa', alpha=0.8)

        # Row 1: Fader position
        ax2 = fig.add_subplot(gs[1, col])
        ax2.plot(times, faders, color='#2196F3', linewidth=1)
        style_ax(ax2, "Fader Position")
        shade_segments(ax2)

        if col == 1:
            scene_t = [r[0] for r in results if r[5] == "SCENE"]
            scene_f = [r[2] for r in results if r[5] == "SCENE"]
            if scene_t:
                ax2.scatter(scene_t, scene_f, color='#ff4444', s=6, zorder=5,
                           alpha=0.7, label=f'Scene change ({len(scene_t)} cycles)')
                ax2.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333',
                          labelcolor='white')

        # Row 2: Output level (the key chart)
        ax3 = fig.add_subplot(gs[2, col])
        ax3.plot(times, outputs, color='#FF9800', linewidth=0.6, alpha=0.9)
        ax3.axhline(y=target, color='#FFD700', linestyle='--', alpha=0.7,
                    label=f'Target ({target} dB)')
        ax3.axhline(y=target + 3, color='#FFD700', linestyle=':', alpha=0.25)
        ax3.axhline(y=target - 3, color='#FFD700', linestyle=':', alpha=0.25)
        ax3.fill_between([0, total_time], target - 3, target + 3,
                        alpha=0.08, color='#FFD700')
        ax3.set_ylim(target - 25, target + 25)
        style_ax(ax3, "Output Level (input + fader) -- what the livestream hears")
        shade_segments(ax3)
        ax3.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

        # Row 3: Error from target (new visual)
        ax4 = fig.add_subplot(gs[3, col])
        errors = [o - target for o in outputs]
        err_colors = ['#4CAF50' if abs(e) <= 3 else '#FF9800' if abs(e) <= 6 else '#f44336' for e in errors]
        ax4.bar(times, errors, width=ADJUSTMENT_INTERVAL * 0.9, color=err_colors, alpha=0.7)
        ax4.axhline(y=0, color='#FFD700', linewidth=0.5)
        ax4.axhline(y=3, color='#FFD700', linestyle=':', alpha=0.3)
        ax4.axhline(y=-3, color='#FFD700', linestyle=':', alpha=0.3)
        ax4.set_ylim(-20, 20)
        style_ax(ax4, "Error from Target (green < 3dB, orange < 6dB, red > 6dB)")
        ax4.set_xlabel("Time (s)", color='#888')
        shade_segments(ax4)

    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n  Chart saved to {filename}")

    # Also save a zoomed view of key transitions
    fig2, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig2.patch.set_facecolor('#0d1117')
    fig2.suptitle("Key Transitions (Zoomed)", fontsize=13, fontweight='bold', color='white')

    # Find interesting transition points
    transitions = []
    prev_label = ""
    for r in new_results:
        if r[4] != prev_label and prev_label != "":
            transitions.append((r[0], prev_label, r[4]))
        prev_label = r[4]

    # Pick 6 interesting transitions
    interesting = [
        ("Band starts", "Welcome speaker", "Worship band"),
        ("Band -> Soft", "Worship band - full", "Soft worship song"),
        ("Band -> Pastor", "Band fading out", "Silence (transition)"),
        ("Video -> Guest", "Silence (from video)", "Quiet guest speaker"),
        ("Guest -> Announce", "Silence (handoff)", "Announcements speaker"),
        ("Closing band", "Offering music", "Closing song"),
    ]

    for idx, (title, from_seg, to_seg) in enumerate(interesting):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor('#0d1117')

        # Find the transition time
        trans_time = None
        for t_time, from_l, to_l in transitions:
            if to_seg.split(" -")[0] in to_l or to_seg == to_l:
                trans_time = t_time
                break
        if trans_time is None:
            # Try partial match
            for t_time, from_l, to_l in transitions:
                if any(w in to_l for w in to_seg.split()):
                    trans_time = t_time
                    break

        if trans_time is None:
            ax.text(0.5, 0.5, f"'{to_seg}' not found", transform=ax.transAxes,
                   ha='center', color='white')
            continue

        window = 15  # seconds before and after
        t_start = max(0, trans_time - window)
        t_end = min(total_time, trans_time + window)

        for name, results, color in [
            ("OLD", old_results, '#f44336'),
            ("NEW", new_results, '#4CAF50')
        ]:
            t_data = [(r[0], r[3]) for r in results if t_start <= r[0] <= t_end]
            if t_data:
                ax.plot([d[0] for d in t_data], [d[1] for d in t_data],
                       color=color, linewidth=1, alpha=0.8, label=name)

        ax.axhline(y=target, color='#FFD700', linestyle='--', alpha=0.5)
        ax.axhline(y=target+3, color='#FFD700', linestyle=':', alpha=0.2)
        ax.axhline(y=target-3, color='#FFD700', linestyle=':', alpha=0.2)
        ax.fill_between([t_start, t_end], target-3, target+3, alpha=0.05, color='#FFD700')
        ax.axvline(x=trans_time, color='white', linestyle='--', alpha=0.3)
        ax.set_title(title, fontsize=9, color='white')
        ax.set_ylim(target - 20, target + 20)
        ax.set_xlim(t_start, t_end)
        ax.tick_params(colors='#888', labelsize=7)
        ax.grid(True, alpha=0.15, color='#444')
        for spine in ax.spines.values():
            spine.set_color('#333')
        if idx == 0:
            ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    plt.savefig("simulator_transitions.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"  Transitions chart saved to simulator_transitions.png")


def main():
    total_dur = sum(s["duration"] for s in SCENARIOS)
    print("X32 Livestream Stabilizer - Extended Algorithm Simulator")
    print(f"Total duration: {total_dur}s ({total_dur/60:.1f} min)")
    print(f"Target output: {TARGET_LEVEL_DB} dB")
    print(f"Fast EMA: {FAST_EMA_SECONDS}s | Slow EMA: {AVERAGING_WINDOW_SECONDS}s | Scene threshold: {SCENE_CHANGE_DB} dB")
    print(f"Stable deadband: {STABLE_DEADBAND_DB} dB | Stable slew: {MAX_FADER_SLEW_DB} dB/cycle")

    print(f"\nScenario timeline ({len(SCENARIOS)} segments):")
    t = 0
    for seg in SCENARIOS:
        extras = []
        if seg.get("drift"):
            extras.append(f"drift:{seg['drift']:+.0f}")
        if seg.get("fade_in"):
            extras.append(f"fade_in:{seg['fade_in']:.0f}s")
        if seg.get("fade_out"):
            extras.append(f"fade_out:{seg['fade_out']:.0f}s")
        if seg.get("spikes"):
            extras.append(f"{len(seg['spikes'])} spikes")
        extra_str = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  {t:5.0f}s - {t+seg['duration']:5.0f}s : {seg['label']:35s} "
              f"(base: {seg['base']:+.0f} dB, dyn: +/-{seg['dynamics']:.0f}){extra_str}")
        t += seg["duration"]

    random.seed(42)
    readings = generate_meter_readings()

    old_results = run_old_algorithm(readings)
    new_results = run_new_algorithm(readings)

    print_report(old_results, new_results, TARGET_LEVEL_DB)
    make_charts(old_results, new_results, TARGET_LEVEL_DB)


if __name__ == "__main__":
    main()
