"""Encoder/decoder functions for pitch strings, to be used by a tokenizer.

This module contains functions to encode and decode pitch strings into integer codes,
and vice versa. The integer codes are used to represent pitches in the model's input and
output layers.

Usage::

    >>> from melody_hpo.data.encoder.pitch import PitchCode
    >>> PitchCode.encode("C4")
    6035
    >>> PitchCode.decode(6035)
    'C4'
    >>> PitchCode.range
    {'min': 1207, 'max': 13176}

Pitch strings are symbolically encoded as "<letter_name><accidentals><octave_asa>", where

    <letter_name> := ([A-G])   # Standard letter names
    <accidentals> := (#|b)*    # Standard notation for sharps and flats
    <octave_asa>  := (\\d)     # ASA octave designation

Pitch notation follows the Acoustical Society of America (ASA) standard, designating
Middle C as C4. Note that while C4 is the fourth octave in scientific notation, it
corresponds to MIDI number 60, which represents the start of the fifth MIDI octave.
To maintain this mapping, the encoder function represents C4 as the binomial code 6035.
  Calculation: `midi_octave` 5; `chromatic` 60; `diatonic` 35 -> (100 * 60) + 35.

Pitch Encoding Range and Boundaries:

The integer encoding is designed to cover the full spectrum of Western equal-tempered
music while reserving lower values for non-pitch tokens:
- Theoretical Lower Bound of the present pitch string notation (C0): Maps to a code of 1207.
  Calculation: `midi_octave` 1; `chromatic` 12; `diatonic` 7 -> (100 * 12) + 7.
- Audible Lower Bound (~20 Hz): Generally corresponds to E0 (MIDI 16), which maps to 1609.
  Calculation: `midi_octave` 1; `chromatic` 16; `diatonic` 9 -> (100 * 16) + 9.
- Musical Range: Practical musical pitches typically reside above this threshold (e.g., Piano A0
  at MIDI 21, encoded as 2112.
  Calculation: `midi_octave` 1; `chromatic` 21; `diatonic` 12 -> (100 * 21) + 12.

By anchoring C0 at 1207, this encoding uniformly accommodates all standard MIDI pitches (0-127)
while reserving the 0-1206 range for auxiliary token codes or special control symbols.

Token Interoperability:

The reserved range (0-1206) provides ample headroom for integration with standard tokenization
schemes. Specifically, this encoding is compatible with architectures that utilize reserved
indices for control tokens, such as:

- [PAD] (Padding) -> 0
- [BOS] (Beginning of Sequence) -> 1
- [EOS] (End of Sequence) -> 2

By anchoring C0 at 1207, the encoder prevents collisions between control tokens and musical
content, ensuring a clean separation for the embedding layer.

Encoding Ceiling and Memory Efficiency:

The encoding covers the entire audible spectrum and the full MIDI 1.0 range while remaining
highly memory-efficient:

- MIDI Maximum (G9, MIDI 127): Maps to a binomial code of 12774.
  Calculation: `midi_octave` 10; `chromatic` 127; `diatonic` 74 -> (100 * 127) + 74.
- Notation Limit (B9, MIDI 131): Maps to 13176.
  Calculation:* `midi_octave` 10; `chromatic` 131; `diatonic` 76 -> (100 * 131) + 76.
- Audible Limit (~Eb10, MIDI 135): Maps to 13579.
  Calculation:* `midi_octave` 11; `chromatic` 135; `diatonic` 79 -> (100 * 135) + 79.

With the absolute ceiling under 14,000, the entire vocabulary (including the reserved
control range of 0-1206) fits within a 14-bit space (log_2(16384) = 14). This ensures
the data is fully compatible with standard 16-bit integer (uint16) containers, optimizing
memory throughput during large-scale training.
"""

import re

from melody_hpo.data.encoder.music import MusicCode


class PitchCode(MusicCode):
    """Encoder/decoder for pitch strings and integer codes.

    Class variables:
        pattern: Group-free regex string matching pitch notation (e.g. ``"C4"``,
            ``"Bb3"``). Safe for ``Series.str.contains()`` filtering.
        range: Dict with ``"min"`` and ``"max"`` integer code boundaries.
    """

    _letter_to_chromatic_pc = {
        "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11
    }
    _letter_to_diatonic_pc = {
        "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6
    }
    _diatonic_pc_to_letter = {
        0: "C", 1: "D", 2: "E", 3: "F", 4: "G", 5: "A", 6: "B"
    }

    pattern = r"[A-G][#b]*\d"

    range = {
        "min": 1207,  # Code for 'C0' -- see docstring
        "max": 13176,  # Code for 'B9'; practically impossible to need a higher value
    }

    @staticmethod
    def encode(pitch: str) -> int:
        """Encode a pitch string into an integer code."""

        # Using regex, parse the pitch string into its components.
        match = re.match(r"([A-G])([#b]*)(\d)", pitch)
        if match:
            letter_name = match.group(1)
            accidentals = match.group(2) or ""
            octave_asa = int(match.group(3))
        else:
            raise ValueError(f"Invalid pitch string: {pitch}")

        # Perform the necessary numerical conversions
        accidentals_int = accidentals.count("#") - accidentals.count("b")
        chromatic_pc_int = PitchCode._letter_to_chromatic_pc[letter_name] + accidentals_int
        diatonic_pc_int = PitchCode._letter_to_diatonic_pc[letter_name]
        octave_int = octave_asa + 1  # Convert ASA to MIDI octave designations

        # Calculate the chromatic and diatonic codes.
        chromatic_code = chromatic_pc_int + (octave_int * 12)
        diatonic_code = diatonic_pc_int + (octave_int * 7)

        # Using 100 for human-readable form: The last two digits of
        # the binomial code represent the diatonic code, and the
        # remaining leading digits represent the chromatic code.
        binomial_code = 100 * chromatic_code + diatonic_code

        # Check code value is within expected range
        if binomial_code < PitchCode.range["min"]:
            raise ValueError(
                f"Code for pitch string {pitch} = {binomial_code}"
                f" is below lower bound {PitchCode.range['min']}"
            )
        elif binomial_code > PitchCode.range["max"]:
            raise ValueError(
                f"Code for pitch string {pitch} = {binomial_code}"
                f" is above upper bound {PitchCode.range['max']}"
            )

        return binomial_code

    @staticmethod
    def decode(code: int) -> str:
        """Decode an integer code into a pitch string."""

        # Retrieve the chromatic and diatonic codes from the binomial code.
        chromatic_code = code // 100
        diatonic_code = code % 100

        # Retrieve the octave from the diatonic code; NB—*not* from the chromatic code,
        # because a negative accidental integer value can push a pitch like 'Cb4' into
        # the lower octave.
        octave_int = diatonic_code // 7
        diatonic_pc_int = diatonic_code % 7

        # Retrieve the letter name based on the diatonic PC integer.
        letter_name = PitchCode._diatonic_pc_to_letter[diatonic_pc_int]

        # Retrieve the chromatic code without accidentals.
        chromatic_code_no_acc = PitchCode._letter_to_chromatic_pc[letter_name] + (octave_int * 12)

        # Retrieve the accidentals integer based on the difference of the chromatic codes
        # with and without accidentals.
        accidentals_int = chromatic_code - chromatic_code_no_acc

        # Convert the accidentals integer to the corresponding string of accidentals.
        if accidentals_int == 0:
            accidentals = ""
        elif accidentals_int > 0:
            accidentals = "#" * accidentals_int
        else:
            accidentals = "b" * (-accidentals_int)

        # Convert the octave integer to the corresponding ASA octave designation.
        octave_asa = octave_int - 1

        # Return the pitch string.
        return f"{letter_name}{accidentals}{octave_asa}"
