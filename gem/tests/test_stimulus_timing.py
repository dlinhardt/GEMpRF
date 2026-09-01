"""Does the stimulus cover the same stretch of time as the scan it is fitted against?

sub-roadmap01 was analysed twice, under two GEMpRF versions, and both times produced maps with no
retinotopic organisation: R2 never above 0.38 across 141k vertices and a median eccentricity of 14.2
against a 15 deg search radius, i.e. the pRFs pushed out to the edge because nothing fitted. The
aperture was a 300 s design (2400 frames @ 0.125 s) against a 385 s scan (385 volumes @ TR 1 s).

Nothing caught it. The check that existed compared the measured timepoint count against
`num_frames_downsampled`, which is a config value set BY HAND to the BOLD length -- so it compared
the data against itself and always passed. The stimulus' own extent was never looked at, and
`downsample_frame_indices()` spreads whatever it is given across that many samples with np.linspace,
so the 300 s design was replayed over 385 s: every bar sweep 28% slow, drifting out of phase with
the data, and a full set of plausible-looking estimates written to disk.

The comparison that catches it is duration against duration, from each file's own header. Where a
scan cannot state a TR (GIFTI) there is a header-free fallback: the downsample factor is
TR / frame-duration and so is a whole number whenever the two clocks agree (cons01: 3080/385 = 8
exactly; roadmap01: 6.234).

`gem.utils.stimulus_timing` imports no CuPy, so this runs without a GPU.
"""
import pytest

from gem.utils.gem_errors import TimepointMismatchError
from gem.utils.stimulus_timing import validate_stimulus_timing


class _FakeStimulus:
    """The four attributes validate_stimulus_timing() reads."""

    def __init__(self, num_frames, frame_duration=0.125, enabled=True, num_frames_downsampled=385):
        self.NumFrames = num_frames
        self.HighTemporalResolutionEnabled = enabled
        self.NumFramesDownsampled = num_frames_downsampled
        # a nibabel header indexes like a mapping; pixdim[4] is the temporal spacing
        self.Header = {"pixdim": [1.0, 1.0, 1.0, 1.0, frame_duration]}


def _cons01():
    """The run that is fine: 3080 frames @ 0.125 s = 385 s, against 385 timepoints @ TR 1 s."""
    return _FakeStimulus(num_frames=3080)


def _roadmap01():
    """The run that was broken: 2400 frames @ 0.125 s = 300 s, against the same 385 s scan."""
    return _FakeStimulus(num_frames=2400)


def test_matching_durations_pass():
    validate_stimulus_timing(_cons01(), 385, measured_tr=1.0, input_filepath="cons01_bold.nii.gz")


def test_short_stimulus_is_refused_with_both_durations_named():
    """The roadmap01 case: frame counts agree, durations do not."""
    with pytest.raises(TimepointMismatchError) as excinfo:
        validate_stimulus_timing(_roadmap01(), 385, measured_tr=1.0, input_filepath="roadmap01_bold.nii.gz")

    message = str(excinfo.value)
    # the numbers are the whole point -- without them the error says nothing actionable
    assert "300" in message and "385" in message
    assert "2400 frames" in message
    assert "roadmap01_bold.nii.gz" in message


def test_stimulus_that_is_a_whole_multiple_too_long_is_still_refused():
    """A 770 s aperture divides evenly into 385, so only the duration test can catch it."""
    too_long = _FakeStimulus(num_frames=6160)  # 6160 / 385 = 16.0 exactly, but 770 s of stimulus
    assert (6160 / 385).is_integer()

    with pytest.raises(TimepointMismatchError):
        validate_stimulus_timing(too_long, 385, measured_tr=1.0, input_filepath="bold.nii.gz")


def test_mismatched_frame_count_is_refused_before_any_duration_arithmetic():
    """The historical check, unchanged: the measured timecourse must be as long as the model one."""
    with pytest.raises(TimepointMismatchError, match="do not match"):
        validate_stimulus_timing(_cons01(), 300, measured_tr=1.0, input_filepath="bold.nii.gz")


def test_high_temporal_resolution_off_compares_raw_frames_and_stops_there():
    """Without downsampling the stimulus IS the model timecourse, so only the count is meaningful."""
    stimulus = _FakeStimulus(num_frames=385, enabled=False)
    validate_stimulus_timing(stimulus, 385, measured_tr=1.0, input_filepath="bold.nii.gz")

    with pytest.raises(TimepointMismatchError, match="do not match"):
        validate_stimulus_timing(stimulus, 300, measured_tr=1.0, input_filepath="bold.nii.gz")


@pytest.mark.parametrize("measured_tr, frame_duration", [(None, 0.125), (1.0, 0.0), (1.0, None)])
def test_without_a_usable_tr_a_bad_factor_warns_but_does_not_raise(capsys, measured_tr, frame_duration):
    """GIFTI, or a header that never had its temporal spacing written."""
    stimulus = _roadmap01()
    if frame_duration is None:
        stimulus.Header = {}
    else:
        stimulus.Header = {"pixdim": [1.0, 1.0, 1.0, 1.0, frame_duration]}

    validate_stimulus_timing(stimulus, 385, measured_tr=measured_tr, input_filepath="bold.gii")

    warning = capsys.readouterr().out
    assert "2400" in warning and "385" in warning
    assert "6.23" in warning  # the offending factor itself


def test_without_a_usable_tr_a_whole_factor_is_silent(capsys):
    validate_stimulus_timing(_cons01(), 385, measured_tr=None, input_filepath="bold.gii")
    assert capsys.readouterr().out == ""
