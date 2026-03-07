"""
Parameter sweep for the X32 stabilizer algorithm.
Tests thousands of parameter combinations to find the optimal config.
"""
import math
import random
import time

from simulator import (SCENARIOS, generate_meter_readings, run_old_algorithm,
                       TARGET_LEVEL_DB, ADJUSTMENT_INTERVAL, MIN_FADER_DB, MAX_FADER_DB,
                       SIGNAL_THRESHOLD_DB)


def run_algo(readings, fast_ema_s, slow_ema_s, scene_db, stable_dead, slew,
             fade_protect=False, ema_pull=0.3, error_slew_mult=3.0, error_slew_thresh=5.0):
    """Configurable algorithm with all tunable parameters."""
    fast_ema = None
    slow_ema = None
    fader_db = 0.0
    in_silence = False
    silence_count = 0
    SILENCE_DEBOUNCE = 6
    prev_level = None
    level_velocity = 0.0

    results = []
    dt = ADJUSTMENT_INTERVAL
    fast_alpha = 1.0 - math.exp(-dt / fast_ema_s)
    slow_alpha = 1.0 - math.exp(-dt / max(slow_ema_s, 0.5))

    for t, level_db, label in readings:
        if level_db < SIGNAL_THRESHOLD_DB:
            silence_count += 1
            if silence_count >= SILENCE_DEBOUNCE:
                in_silence = True
            output_db = level_db + fader_db
            results.append((t, level_db, fader_db, output_db, label,
                           "silence" if in_silence else "below", 0.0))
            continue

        if silence_count > 0 or in_silence:
            fast_ema = level_db
            slow_ema = level_db
            in_silence = False
            prev_level = None
            level_velocity = 0.0
        silence_count = 0

        # Track level velocity (rate of change in dB/cycle)
        if prev_level is not None:
            level_velocity = 0.7 * level_velocity + 0.3 * (level_db - prev_level)
        prev_level = level_db

        if fast_ema is None:
            fast_ema = level_db
            slow_ema = level_db
            fader_db = TARGET_LEVEL_DB - level_db
            fader_db = max(MIN_FADER_DB, min(MAX_FADER_DB, fader_db))
            output_db = level_db + fader_db
            results.append((t, level_db, fader_db, output_db, label, "COLD", 0.0))
            continue

        fast_ema += fast_alpha * (level_db - fast_ema)
        slow_ema += slow_alpha * (level_db - slow_ema)

        change_mag = abs(fast_ema - slow_ema)
        is_scene = change_mag > scene_db

        # Fade detection: level changing rapidly in one direction
        is_fading = fade_protect and abs(level_velocity) > 2.0

        if is_scene and not is_fading:
            control_level = fast_ema
            slow_ema += ema_pull * (fast_ema - slow_ema)
        elif is_fading:
            control_level = 0.5 * fast_ema + 0.5 * slow_ema
            slow_ema += 0.15 * (fast_ema - slow_ema)
        else:
            control_level = slow_ema

        desired = TARGET_LEVEL_DB - control_level
        desired = max(MIN_FADER_DB, min(MAX_FADER_DB, desired))

        diff = desired - fader_db
        output_error = abs((level_db + fader_db) - TARGET_LEVEL_DB)

        if is_scene and not is_fading:
            deadband = 0.5
            max_slew = 20.0
        elif output_error > error_slew_thresh:
            deadband = stable_dead
            max_slew = slew * error_slew_mult
        else:
            deadband = stable_dead
            max_slew = slew

        if abs(diff) > deadband:
            if abs(diff) > max_slew:
                diff = max_slew if diff > 0 else -max_slew
            fader_db += diff
            fader_db = max(MIN_FADER_DB, min(MAX_FADER_DB, fader_db))

        output_db = level_db + fader_db
        state = "SCENE" if is_scene else ("FADE" if is_fading else "stable")
        results.append((t, level_db, fader_db, output_db, label, state, change_mag))

    return results


def score_results(results, target):
    """Score algorithm results with a composite metric."""
    active = [r for r in results if r[5] not in ("silence", "below")]
    if not active:
        return None
    outputs = [r[3] for r in active]
    faders = [r[2] for r in active]
    errors = [abs(o - target) for o in outputs]

    within_3 = sum(1 for e in errors if e <= 3.0) / len(errors) * 100
    within_6 = sum(1 for e in errors if e <= 6.0) / len(errors) * 100
    std = (sum(e**2 for e in errors) / len(errors)) ** 0.5
    worst = max(errors)
    fader_travel = sum(abs(faders[i] - faders[i-1]) for i in range(1, len(faders)))
    scenes = sum(1 for r in results if r[5] == "SCENE")

    # Composite: reward accuracy, penalize spikes and excessive wobble
    composite = (within_3 * 1.0
                 + within_6 * 0.5
                 - worst * 0.5
                 - fader_travel * 0.02
                 - std * 2.0)

    return {
        "within_3": within_3, "within_6": within_6, "std": std,
        "worst": worst, "travel": fader_travel, "scenes": scenes,
        "composite": composite
    }


def per_segment_settle(results, target, settle_db=3.0):
    """Measure settle time per segment."""
    segments = {}
    for r in results:
        if r[4] not in segments:
            segments[r[4]] = []
        segments[r[4]].append(r)

    settle_times = {}
    for label, data in segments.items():
        active = [r for r in data if r[5] not in ("silence", "below")]
        if not active:
            continue
        first_t = active[0][0]
        settled = None
        for r in active:
            if abs(r[3] - target) <= settle_db:
                settled = r[0] - first_t
                break
        settle_times[label] = settled
    return settle_times


def main():
    t_start = time.time()

    # Generate readings with fixed seed
    random.seed(42)
    readings = generate_meter_readings()

    # Old algorithm baseline
    old_results = run_old_algorithm(readings)
    old_s = score_results(old_results, TARGET_LEVEL_DB)

    print("X32 Stabilizer - Parameter Sweep")
    print("=" * 120)
    print(f"OLD baseline: <3dB={old_s['within_3']:.1f}%  <6dB={old_s['within_6']:.1f}%  "
          f"std={old_s['std']:.1f}  worst={old_s['worst']:.1f}  travel={old_s['travel']:.0f}  "
          f"score={old_s['composite']:.1f}")
    print()

    # Current v2 defaults
    v2 = run_algo(readings, 2.0, 10.0, 6.0, 2.0, 0.5)
    v2_s = score_results(v2, TARGET_LEVEL_DB)
    print(f"Current v2:   <3dB={v2_s['within_3']:.1f}%  <6dB={v2_s['within_6']:.1f}%  "
          f"std={v2_s['std']:.1f}  worst={v2_s['worst']:.1f}  travel={v2_s['travel']:.0f}  "
          f"score={v2_s['composite']:.1f}")
    print()

    # === SWEEP ===
    all_results = []
    count = 0

    for fast_ema in [1.0, 1.5, 2.0, 3.0]:
        for slow_ema in [5.0, 8.0, 10.0, 15.0]:
            for scene_db in [4.0, 5.0, 6.0, 8.0]:
                for deadband in [1.0, 1.5, 2.0, 3.0]:
                    for slew in [0.3, 0.5, 0.8, 1.0, 1.5]:
                        for pull in [0.2, 0.3, 0.5]:
                            for fade_prot in [False, True]:
                                for err_mult in [2.0, 3.0, 5.0]:
                                    for err_thresh in [3.0, 5.0, 8.0]:
                                        r = run_algo(readings, fast_ema, slow_ema,
                                                     scene_db, deadband, slew,
                                                     fade_prot, pull, err_mult,
                                                     err_thresh)
                                        s = score_results(r, TARGET_LEVEL_DB)
                                        if s:
                                            params = {
                                                "fast": fast_ema, "slow": slow_ema,
                                                "scene": scene_db, "dead": deadband,
                                                "slew": slew, "pull": pull,
                                                "fade": fade_prot,
                                                "err_mult": err_mult,
                                                "err_thresh": err_thresh
                                            }
                                            all_results.append((s, params))
                                        count += 1
                                        if count % 5000 == 0:
                                            print(f"  ...tested {count} combinations")

    elapsed = time.time() - t_start
    all_results.sort(key=lambda x: x[0]["composite"], reverse=True)

    print(f"\nTested {count} combinations in {elapsed:.1f}s")
    print()

    # Top 20
    print("TOP 20 CONFIGURATIONS:")
    header = (f"{'Rank':>4} {'<3dB':>6} {'<6dB':>6} {'Std':>5} {'Worst':>6} "
              f"{'Travel':>7} {'Score':>7}  Parameters")
    print(header)
    print("-" * 120)

    for i, (s, p) in enumerate(all_results[:20]):
        fade_str = "+fade" if p["fade"] else "     "
        print(f"{i+1:4d} {s['within_3']:5.1f}% {s['within_6']:5.1f}% "
              f"{s['std']:5.1f} {s['worst']:6.1f} {s['travel']:6.0f}dB "
              f"{s['composite']:7.1f}  "
              f"fast={p['fast']} slow={p['slow']} scene={p['scene']} "
              f"dead={p['dead']} slew={p['slew']} pull={p['pull']} "
              f"err={p['err_mult']}x@{p['err_thresh']}dB {fade_str}")

    # Show the best config in detail
    best_s, best_p = all_results[0]
    print()
    print("=" * 120)
    print("BEST CONFIGURATION DETAIL:")
    print(f"  Parameters: {best_p}")
    print(f"  <3dB: {best_s['within_3']:.1f}%  <6dB: {best_s['within_6']:.1f}%  "
          f"std: {best_s['std']:.1f}  worst: {best_s['worst']:.1f}  "
          f"travel: {best_s['travel']:.0f}dB")
    print()

    # Per-segment settle times for best vs v2 vs old
    print("PER-SEGMENT SETTLE TIME (seconds to reach +/-3dB of target):")
    print(f"  {'Segment':40s} {'OLD':>7} {'V2':>7} {'BEST':>7}")
    print("  " + "-" * 70)

    old_settle = per_segment_settle(old_results, TARGET_LEVEL_DB)
    v2_settle = per_segment_settle(v2, TARGET_LEVEL_DB)
    best_results = run_algo(readings, best_p["fast"], best_p["slow"], best_p["scene"],
                            best_p["dead"], best_p["slew"], best_p["fade"],
                            best_p["pull"], best_p["err_mult"], best_p["err_thresh"])
    best_settle = per_segment_settle(best_results, TARGET_LEVEL_DB)

    for label in old_settle:
        ov = old_settle.get(label)
        vv = v2_settle.get(label)
        bv = best_settle.get(label)
        os = f"{ov:.1f}s" if ov is not None else "never"
        vs = f"{vv:.1f}s" if vv is not None else "never"
        bs = f"{bv:.1f}s" if bv is not None else "never"
        print(f"  {label:40s} {os:>7} {vs:>7} {bs:>7}")

    # Improvement over v2
    print()
    print("IMPROVEMENT: BEST vs V2 vs OLD:")
    for metric in ["within_3", "within_6", "std", "worst", "travel"]:
        o = old_s[metric]
        v = v2_s[metric]
        b = best_s[metric]
        print(f"  {metric:12s}  OLD={o:7.1f}  V2={v:7.1f}  BEST={b:7.1f}")

    # Test best config across multiple random seeds for robustness
    print()
    print("ROBUSTNESS TEST (best config across 10 random seeds):")
    seed_scores = []
    for seed in range(10):
        random.seed(seed)
        rd = generate_meter_readings()
        r = run_algo(rd, **best_p)
        s = score_results(r, TARGET_LEVEL_DB)
        seed_scores.append(s)
        print(f"  Seed {seed}: <3dB={s['within_3']:5.1f}%  <6dB={s['within_6']:5.1f}%  "
              f"std={s['std']:4.1f}  worst={s['worst']:5.1f}  score={s['composite']:6.1f}")

    avg_3 = sum(s["within_3"] for s in seed_scores) / len(seed_scores)
    avg_6 = sum(s["within_6"] for s in seed_scores) / len(seed_scores)
    avg_std = sum(s["std"] for s in seed_scores) / len(seed_scores)
    avg_worst = sum(s["worst"] for s in seed_scores) / len(seed_scores)
    print(f"  AVG:    <3dB={avg_3:5.1f}%  <6dB={avg_6:5.1f}%  "
          f"std={avg_std:4.1f}  worst={avg_worst:5.1f}")

    # Also test v2 across same seeds for fair comparison
    print()
    print("ROBUSTNESS TEST (v2 config across same 10 seeds):")
    v2_seed_scores = []
    for seed in range(10):
        random.seed(seed)
        rd = generate_meter_readings()
        r = run_algo(rd, 2.0, 10.0, 6.0, 2.0, 0.5)
        s = score_results(r, TARGET_LEVEL_DB)
        v2_seed_scores.append(s)
        print(f"  Seed {seed}: <3dB={s['within_3']:5.1f}%  <6dB={s['within_6']:5.1f}%  "
              f"std={s['std']:4.1f}  worst={s['worst']:5.1f}  score={s['composite']:6.1f}")

    avg_3 = sum(s["within_3"] for s in v2_seed_scores) / len(v2_seed_scores)
    avg_6 = sum(s["within_6"] for s in v2_seed_scores) / len(v2_seed_scores)
    avg_std = sum(s["std"] for s in v2_seed_scores) / len(v2_seed_scores)
    avg_worst = sum(s["worst"] for s in v2_seed_scores) / len(v2_seed_scores)
    print(f"  AVG:    <3dB={avg_3:5.1f}%  <6dB={avg_6:5.1f}%  "
          f"std={avg_std:4.1f}  worst={avg_worst:5.1f}")


if __name__ == "__main__":
    main()
