"""Tests for the pitch encoder/decoder functions."""

import pytest

from melody_hpo.data.encoder.pitch import PitchCode

PITCH_CODE_PAIRS = [
    ("C4", 6035),   # Middle C
    ("F#4", 6638),  # Sharp in middle octave
    ("Gb4", 6639),  # Enharmonic to the previous pitch
    ("Bb3", 5834),  # Flat below middle C
    ("A0", 2112),   # Piano low A
    ("C0", 1207),   # Theoretical lower bound
]


@pytest.mark.parametrize(("pitch", "expected_code"), PITCH_CODE_PAIRS)
def test_pitch_encode(pitch: str, expected_code: int) -> None:
    assert PitchCode.encode(pitch) == expected_code


@pytest.mark.parametrize(("pitch", "expected_code"), PITCH_CODE_PAIRS)
def test_pitch_decode(pitch: str, expected_code: int) -> None:
    assert PitchCode.decode(expected_code) == pitch


@pytest.mark.parametrize(("pitch", "expected_code"), PITCH_CODE_PAIRS)
def test_pitch_roundtrip(pitch: str, expected_code: int) -> None:
    assert PitchCode.decode(PitchCode.encode(pitch)) == pitch


def test_pitch_encode_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid pitch string"):
        PitchCode.encode("X4")


def test_pitch_encode_below_range() -> None:
    with pytest.raises(ValueError, match="below lower bound"):
        PitchCode.encode("Cb0")


def test_pitch_encode_above_range() -> None:
    with pytest.raises(ValueError, match="above upper bound"):
        PitchCode.encode("B#9")
