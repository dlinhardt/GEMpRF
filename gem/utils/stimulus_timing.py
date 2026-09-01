# -*- coding: utf-8 -*-

"""
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   Does the stimulus cover the same stretch of time as the scan it is fitted against?",
"""

import numpy as np

from gem.utils.gem_errors import TimepointMismatchError
from gem.utils.logger import Logger

# How far the two durations may drift apart before the analysis is refused, as a multiple of one
# stimulus frame. A high-rate aperture is written on its own frame clock, so the last frame can
# round either way against the scan; anything past a single frame is a different design, not
# rounding.
DURATION_TOLERANCE_IN_FRAMES = 1.0

# The downsample factor is TR / stimulus-frame-duration and so is a whole number for essentially
# every high-rate aperture (cons01: 3080 / 385 = 8). This is the residual allowed before the
# header-free fallback calls it suspicious.
INTEGER_FACTOR_TOLERANCE = 1e-3


def _stimulus_frame_duration(stimulus):
    """Seconds per stimulus frame from the aperture's own header, or None when it has no usable one."""
    try:
        frame_duration = float(stimulus.Header['pixdim'][4])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return frame_duration if np.isfinite(frame_duration) and frame_duration > 0 else None


def validate_stimulus_timing(stimulus, num_measured_timepoints, measured_tr, input_filepath):
    """Refuse an analysis whose stimulus does not span the same amount of time as its scan.

    The frame-count check comes first and is the historical one: the measured timecourse has to be
    as long as the model timecourse will be.

    The duration check is the one that catches a wrong aperture. With high_temporal_resolution the
    frame counts cannot disagree -- ``num_frames_downsampled`` is a config value set to the BOLD
    length, so comparing the data against it is comparing the data against itself. What is never
    otherwise examined is the stimulus' own extent:
    ``downsample_frame_indices()`` spreads whatever it is handed across ``num_frames_downsampled``
    samples with ``np.linspace``, so a 300 s design against a 385 s scan is silently played back 28%
    slow, every sweep drifts out of phase with the data, and the fit degrades into noise while still
    producing a full set of plausible-looking estimates.

    Both sides of that comparison come from headers, so when the scan cannot state a TR (GIFTI) the
    check falls back to warning about a downsample factor that is not a whole number -- the same
    mistake seen without trusting anyone's temporal spacing, and only a warning because a genuinely
    unusual setup should not be blocked on a heuristic.
    """
    # the model timecourse is as long as the stimulus, or as long as it will be downsampled to
    stimulus_num_frames = (stimulus.NumFrames, stimulus.NumFramesDownsampled)[stimulus.HighTemporalResolutionEnabled]
    if num_measured_timepoints != stimulus_num_frames:
        raise TimepointMismatchError(
            f"Number of timepoints in measured fMRI data ({num_measured_timepoints}) and stimulus "
            f"({stimulus_num_frames}) do not match for file: {input_filepath}")

    if not stimulus.HighTemporalResolutionEnabled:
        return

    stimulus_frame_duration = _stimulus_frame_duration(stimulus)
    stimulus_frames = stimulus.NumFrames

    if stimulus_frame_duration is not None and measured_tr is not None:
        stimulus_duration = stimulus_frames * stimulus_frame_duration
        scan_duration = num_measured_timepoints * measured_tr
        if abs(stimulus_duration - scan_duration) > DURATION_TOLERANCE_IN_FRAMES * stimulus_frame_duration:
            # an empty aperture has no speed to be replayed at, so say so rather than divide by zero
            speedup = f"{scan_duration / stimulus_duration:.3f}x" if stimulus_duration else "an undefined multiple of"
            raise TimepointMismatchError(
                f"Stimulus and scan cover different amounts of time for file: {input_filepath}\n"
                f"  stimulus: {stimulus_duration:g} s "
                f"({stimulus_frames} frames @ {stimulus_frame_duration:g} s)\n"
                f"  scan    : {scan_duration:g} s "
                f"({num_measured_timepoints} timepoints @ TR {measured_tr:g} s)\n"
                f"The stimulus is resampled onto the scan's timepoints regardless, so it would be "
                f"replayed at {speedup} its real speed and no pRF would line up with the data. "
                f"Check that this aperture belongs to this run.")
        return

    # No dependable TR on one side or the other: fall back to the downsample factor, which is
    # TR / frame-duration and so is a whole number whenever the two clocks agree.
    factor = stimulus_frames / stimulus.NumFramesDownsampled
    if abs(factor - round(factor)) > INTEGER_FACTOR_TOLERANCE:
        Logger.print_yellow_message(
            f"Warning: stimulus frames ({stimulus_frames}) is not a whole multiple of the "
            f"downsampled length ({stimulus.NumFramesDownsampled}) -- the factor is {factor:.4f}.\n"
            f"The stimulus will be stretched to fit, which is what a stimulus belonging to a "
            f"different run looks like. Could not verify against the scan duration because the "
            f"{'stimulus' if stimulus_frame_duration is None else 'measured data'} does not state a "
            f"usable temporal spacing. File: {input_filepath}", print_file_name=False)
