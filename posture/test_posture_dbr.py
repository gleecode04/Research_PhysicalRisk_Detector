"""Tests for the calibrated DBR posture engine (posture_dbr.py).

Run directly (python test_posture_dbr.py) or with pytest, matching
test_posture_feedback.py's style.
"""
import math

import pytest

from posture_dbr import (
    CAUSE_ASYMMETRIC,
    CAUSE_FORWARD_BENDING,
    CAUSE_NEUTRAL,
    CAUSE_OVERHEAD,
    CAUSE_STATIC,
    CAUSE_TWISTING,
    GOOD_MAX_DEG,
    GravitySample,
    HIGH,
    LOW,
    MEDIUM,
    STATIC_BAD_SECONDS,
    STATIC_CRITICAL_SECONDS,
    axis_delta,
    build_feedback_message,
    build_voice_message,
    classify_cause,
    classify_tilt_severity,
    compute_window_stats,
    escalate_for_static_duration,
    tilt_angle_deg,
)

NEUTRAL = GravitySample(0.0, 0.0, 1.0)  # device "up" axis aligned with gravity at calibration


def _window(samples):
    return list(samples)


# --- tilt_angle_deg -----------------------------------------------------

def test_tilt_is_zero_at_neutral_regardless_of_mount_orientation():
    # The whole point of calibrating against a vector-angle instead of a
    # fixed-axis formula: neutral reads 0 no matter which raw axis gravity
    # happens to sit on for this mount point.
    weird_mount_neutral = GravitySample(0.6, -0.7, 0.39)
    assert tilt_angle_deg(weird_mount_neutral, weird_mount_neutral) == pytest.approx(0.0, abs=1e-9)


def test_tilt_is_90_for_a_right_angle_bend():
    bent = GravitySample(1.0, 0.0, 0.0)
    assert tilt_angle_deg(bent, NEUTRAL) == pytest.approx(90.0, abs=1e-6)


def test_tilt_grows_monotonically_with_forward_lean():
    angles_deg = [0, 10, 30, 45, 60, 90]
    samples = [GravitySample(math.sin(math.radians(a)), 0.0, math.cos(math.radians(a))) for a in angles_deg]
    tilts = [tilt_angle_deg(s, NEUTRAL) for s in samples]
    assert tilts == sorted(tilts)
    for expected, actual in zip(angles_deg, tilts):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_axis_delta_is_zero_at_neutral():
    delta = axis_delta(NEUTRAL, NEUTRAL)
    assert (delta.ax, delta.ay, delta.az) == (0.0, 0.0, 0.0)


# --- classify_tilt_severity ----------------------------------------------

def test_severity_boundaries():
    assert classify_tilt_severity(0.0) == LOW
    assert classify_tilt_severity(20.0) == LOW
    assert classify_tilt_severity(20.1) == MEDIUM
    assert classify_tilt_severity(45.0) == MEDIUM
    assert classify_tilt_severity(45.1) == HIGH


# --- classify_cause -------------------------------------------------------

def _tilted(ax, ay, az):
    return GravitySample(ax, ay, az)


def test_neutral_window_has_neutral_cause():
    window = [NEUTRAL] * 10
    stats = compute_window_stats(window, NEUTRAL)
    assert classify_cause(stats) == CAUSE_NEUTRAL


def test_sustained_forward_lean_is_forward_bending():
    # A 50-degree-ish lean held steady (low within-window variation).
    lean = GravitySample(0.76, 0.0, 0.64)
    window = [lean] * 10
    stats = compute_window_stats(window, NEUTRAL)
    assert stats.max_tilt_deg > 45
    assert classify_cause(stats) == CAUSE_FORWARD_BENDING


def test_oscillating_ax_and_ay_is_twisting():
    window = []
    for i in range(10):
        sign = 1 if i % 2 == 0 else -1
        window.append(GravitySample(0.5 * sign, 0.5 * sign, 0.7))
    stats = compute_window_stats(window, NEUTRAL)
    assert classify_cause(stats) == CAUSE_TWISTING


def test_a_single_steady_window_is_not_static_cause():
    # Static Posture is a multi-minute DURATION concept (see
    # escalate_for_static_duration), not something one steady ~1s window
    # should claim -- otherwise every calm, deliberate bend would get
    # mislabeled "static" immediately. A single held window should still
    # resolve to a dominant-axis cause like forward_bending.
    held = GravitySample(0.4, 0.0, 0.9)
    window = [held] * 10
    stats = compute_window_stats(window, NEUTRAL)
    assert stats.max_tilt_deg > GOOD_MAX_DEG
    assert classify_cause(stats) != CAUSE_STATIC


def test_high_ay_variation_is_overhead_work():
    window = []
    for i in range(10):
        wobble = 0.3 if i % 2 == 0 else -0.3
        window.append(GravitySample(0.2, wobble, 0.85))
    stats = compute_window_stats(window, NEUTRAL)
    assert classify_cause(stats) == CAUSE_OVERHEAD


def test_diffuse_variation_with_no_dominant_axis_is_asymmetric():
    window = []
    for i in range(10):
        jitter = 0.05 if i % 2 == 0 else -0.05
        window.append(GravitySample(0.3 + jitter, 0.3 + jitter, 0.8 + jitter))
    stats = compute_window_stats(window, NEUTRAL)
    assert classify_cause(stats) == CAUSE_ASYMMETRIC


# --- escalate_for_static_duration -----------------------------------------

def test_no_escalation_below_duration_threshold():
    severity, cause = escalate_for_static_duration(LOW, CAUSE_NEUTRAL, STATIC_BAD_SECONDS - 1)
    assert (severity, cause) == (LOW, CAUSE_NEUTRAL)


def test_escalates_neutral_hold_to_medium_after_five_minutes():
    severity, cause = escalate_for_static_duration(LOW, CAUSE_NEUTRAL, STATIC_BAD_SECONDS)
    assert severity == MEDIUM
    assert cause == CAUSE_STATIC


def test_escalates_to_high_after_ten_minutes():
    severity, cause = escalate_for_static_duration(LOW, CAUSE_NEUTRAL, STATIC_CRITICAL_SECONDS)
    assert severity == HIGH
    assert cause == CAUSE_STATIC


def test_active_bad_angle_keeps_its_own_cause_even_if_also_sustained():
    severity, cause = escalate_for_static_duration(HIGH, CAUSE_FORWARD_BENDING, STATIC_CRITICAL_SECONDS)
    assert severity == HIGH
    assert cause == CAUSE_FORWARD_BENDING  # not overwritten to static_posture


def test_never_downgrades():
    severity, cause = escalate_for_static_duration(HIGH, CAUSE_TWISTING, 0)
    assert severity == HIGH
    assert cause == CAUSE_TWISTING


# --- feedback / voice messages --------------------------------------------

def test_low_severity_has_no_feedback():
    assert build_feedback_message(LOW, CAUSE_NEUTRAL) is None
    assert build_voice_message(LOW, CAUSE_NEUTRAL) is None


def test_medium_forward_bending_uses_cause_specific_text():
    assert build_feedback_message(MEDIUM, CAUSE_FORWARD_BENDING) == (
        "Straighten your back and lift your head slightly."
    )


def test_high_twisting_uses_cause_specific_text():
    assert build_feedback_message(HIGH, CAUSE_TWISTING) == "Stop and reposition before continuing."


def test_unknown_cause_falls_back_to_generic_text():
    assert build_feedback_message(HIGH, "some_future_cause") == (
        "Unsafe posture detected. Correct posture immediately if safe."
    )


def test_voice_message_names_the_cause():
    msg = build_voice_message(HIGH, CAUSE_FORWARD_BENDING)
    assert "forward bending" in msg
    assert msg.endswith("Unsafe posture detected. Stand upright and reposition.")


def test_voice_message_none_when_no_feedback():
    assert build_voice_message(LOW, CAUSE_NEUTRAL) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
