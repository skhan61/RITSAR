"""Tests for RITSAR on a current Python, numpy and scipy.

There were none before. Every bug the python3-compatibility branch fixes was a
crash on the first call, not a wrong number, so a single end-to-end run of each
public entry point would have caught all of them years ago. That is what this
file is.

TWO KINDS OF TEST, and the distinction is the point:

  * The image-formation tests place a point scatterer at a KNOWN position,
    focus it, and assert the peak lands there. They check the answer, not just
    that nothing raised. A focuser with a sign error still returns an image.

  * The reader tests load the sample data committed under examples/data and
    assert the shapes are self-consistent. There is no ground truth for a real
    collect, so they check that the reader runs and that what it returns agrees
    with the platform dictionary it built.

The geometry below is SIDE-LOOKING on purpose. An earlier version of these
tests put the platform directly overhead, where across-track displacement is
almost perpendicular to the line of sight: 6 m of it changed the range by under
2 mm, far below the 0.25 m resolution, so the across-track axis silently did
not resolve at all and every point appeared to focus to the same column.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

from ritsar import imgTools, phsRead, phsTools

SPEED_OF_LIGHT = 3.0e8
DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


@pytest.fixture(scope="module")
def platform() -> dict:
    """A side-looking X-band spotlight collect: 600 MHz, 620 m aperture, 10 km slant."""
    npulses, nsamples = 128, 256
    centre_frequency, bandwidth, pulse_duration = 9.6e9, 600e6, 10.0e-6
    chirprate = bandwidth / pulse_duration
    fast_time = numpy.linspace(-pulse_duration / 2, pulse_duration / 2, nsamples)
    frequency = centre_frequency + chirprate * fast_time
    along_track = numpy.linspace(-310.0, 310.0, npulses)
    position = numpy.stack(
        [
            numpy.full(npulses, -8000.0),  # across-track, so the look is not nadir
            along_track,
            numpy.full(npulses, 6000.0),  # altitude
        ],
        axis=1,
    )
    return {
        "f_0": centre_frequency,
        "freq": frequency,
        "nsamples": nsamples,
        "npulses": npulses,
        "pos": position,
        "delta_r": SPEED_OF_LIGHT / (2 * bandwidth),
        "R_c": position[npulses // 2],
        "t": fast_time,
        "k_r": 4 * numpy.pi * frequency / SPEED_OF_LIGHT,
        "chirprate": chirprate,
        "T_p": pulse_duration,
        "B_IF": bandwidth,
        "delta_t": fast_time[1] - fast_time[0],
        "k_y": None,
    }


@pytest.fixture(scope="module")
def image_plane(platform: dict) -> dict:
    """The reconstruction grid the focusers write into."""
    return imgTools.img_plane_dict(platform, res_factor=1.0, aspect=1, upsample=True)


@pytest.fixture(scope="module")
def focused_inputs(platform: dict) -> tuple:
    """One scatterer at a known offset, simulated and RVP-corrected.

    Module-scoped because simulating 128 pulses is the slow part and every
    image-formation test wants the same phase history.
    """
    point = [6.0, 9.0, 0.0]
    phase_history = phsTools.simulate_phs(platform, points=[point], amplitudes=[1.0])
    return point, phsTools.RVP_correct(phase_history, platform)


def peak_position(image, image_plane: dict) -> tuple[float, float]:
    """Where the brightest pixel sits, in metres on the image plane."""
    magnitude = numpy.abs(image)
    flat = int(numpy.argmax(magnitude))
    row, column = flat // magnitude.shape[1], flat % magnitude.shape[1]
    return float(image_plane["u"][column]), float(image_plane["v"][row])


def test_simulate_phs_returns_complex_phase_history(platform: dict) -> None:
    """Catches a forward model that silently drops the imaginary part.

    A real-valued phase history focuses to a symmetric pair of peaks rather than
    one, which reads as a sidelobe problem instead of as a broken simulator.

    THE SCATTERER IS OFF CENTRE, and that is not incidental. At the scene centre
    the differential range is exactly zero, so the phase is zero and every
    sample is 1+0j: a correct simulator produces a real-valued array there, and
    asserting otherwise fails against the physics rather than against a bug.
    """
    phase_history = phsTools.simulate_phs(platform, points=[[6.0, 9.0, 0.0]], amplitudes=[1.0])
    assert phase_history.shape == (platform["npulses"], platform["nsamples"])
    assert numpy.iscomplexobj(phase_history)
    assert numpy.any(phase_history.imag != 0.0)


@pytest.mark.parametrize(
    "algorithm",
    ["backprojection", "polar_format", "DSBP", "FFBP"],
    ids=["backprojection", "polar_format", "DSBP", "FFBP"],
)
def test_a_scatterer_focuses_where_it_was_placed(
    algorithm: str, platform: dict, image_plane: dict, focused_inputs: tuple
) -> None:
    """Catches every crash the python3 branch fixes, and sign errors besides.

    FFBP is what surfaced both the removed numpy aliases and the two bugs in
    signal.decimate: it was the only algorithm reaching them.

    The tolerance is one resolution cell. The peak is a discrete pixel, so
    demanding better than the grid spacing would be demanding the impossible.
    """
    point, corrected = focused_inputs
    if algorithm == "backprojection":
        image = imgTools.backprojection(
            corrected, platform, image_plane, taylor=17, upsample=2, prnt=False
        )
    elif algorithm == "polar_format":
        image = imgTools.polar_format(corrected, platform, image_plane, taylor=17)
    elif algorithm == "DSBP":
        image = imgTools.DSBP(corrected, platform, image_plane)
    else:
        image = imgTools.FFBP(corrected, platform, image_plane, prnt=False)

    across, along = peak_position(image, image_plane)
    resolution = platform["delta_r"]

    # The image plane's u axis runs opposite to the scene's x axis. That is the
    # library's convention, not an error, and asserting it here means a future
    # change to it fails loudly rather than flipping every image silently.
    assert abs(across - (-point[0])) <= resolution, f"{algorithm}: across-track off"
    assert abs(along - point[1]) <= resolution, f"{algorithm}: along-track off"


def test_decimate_accepts_an_odd_length_signal() -> None:
    """Catches the two bugs in `signal.decimate` directly rather than through FFBP.

    `padlen = n/2` was a float that scipy pushed into a slice, and the result was
    indexed with a list of slices, which numpy removed in 1.23. Both raised from
    inside scipy or numpy, so the traceback pointed away from the real cause.
    """
    from ritsar import signal

    decimated = signal.decimate(numpy.arange(401, dtype=float), q=2)
    assert decimated.ndim == 1
    assert decimated.size == pytest.approx(401 / 2, abs=1)


@pytest.mark.parametrize(
    ("name", "loader"),
    [
        ("AFRL", lambda: phsRead.AFRL(str(DATA / "AFRL" / "pass1"), "HH", start_az=1, n_az=3)),
        ("Sandia", lambda: phsRead.Sandia(str(DATA / "Sandia") + "/")),
    ],
)
def test_a_reader_returns_a_phase_history_matching_its_platform(name: str, loader) -> None:
    """Catches an integer-division fix that changed a shape instead of preserving it.

    There is no ground truth for a recorded collect, so the check is internal
    consistency: the array the reader returns has to have the dimensions the
    platform dictionary it built says it has. A wrong `//` would break that.
    """
    phase_history, platform = loader()
    assert numpy.iscomplexobj(phase_history), f"{name} returned real data"
    assert phase_history.shape == (platform["npulses"], platform["nsamples"])
    assert platform["pos"].shape == (platform["npulses"], 3)


def test_the_dirsig_reader_loads_its_sample_collect() -> None:
    """Separate from the others because it needs `spectral`, which is optional.

    DIRSIG also carried a division by THREE, `len(pos_dirs)/3`, which the
    sweep for `/2` missed entirely. Nothing else in the package divides by 3.
    """
    spectral = pytest.importorskip("spectral", reason="the DIRSIG reader needs `spectral`")
    assert spectral is not None
    phase_history, platform = phsRead.DIRSIG(str(DATA / "DIRSIG") + "/")
    assert numpy.iscomplexobj(phase_history)
    assert phase_history.shape[0] == platform["npulses"]
