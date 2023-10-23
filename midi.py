import itertools
import re
from functools import reduce
from typing import Generator

import mido
import numpy as np


def numpy_to_track(
    notes: np.ndarray,
    fs: float,
    frame_shift: int,
    velocity: int = 127,
    instrument: int = 0,
    tempo: int = 500000,
    ticks_per_beat: int = 480,
) -> mido.MidiTrack:
    # notesをstringとして保存する
    cast_notes = notes.astype(int).astype(str).T
    notes_strings = list(map(lambda row: "".join(row), cast_notes))

    init: list[mido.Message] = []

    # 1が連続して出てくる区間を探索し, その区間に変換する
    messages = reduce(
        lambda x, y: x
        + list(
            itertools.chain.from_iterable(  # flatten
                map(
                    lambda match: (
                        # note on message
                        mido.Message(
                            "note_on",
                            note=y[0],
                            velocity=velocity,
                            time=mido.second2tick(
                                match.span()[0] * frame_shift / fs,
                                ticks_per_beat,
                                tempo,
                            ),
                        ),
                        # note off message (velocity=0)
                        mido.Message(
                            "note_on",
                            note=y[0],
                            velocity=0,
                            time=mido.second2tick(
                                match.span()[1] * frame_shift / fs,
                                ticks_per_beat,
                                tempo,
                            ),
                        ),
                    ),
                    re.finditer(r"1+", y[1]),
                )
            )
        ),
        enumerate(notes_strings),
        init,
    )

    messages.sort(key=lambda m: m.time)
    messages = list(_to_rel_time(messages))

    prefix = [
        mido.MetaMessage("set_tempo", tempo=tempo, time=0),
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            notated_32nd_notes_per_beat=8,
            time=0,
        ),
        mido.Message("program_change", channel=0, program=instrument, time=0),
    ]

    suffix = [mido.MetaMessage("end_of_track", time=1)]
    track = mido.MidiTrack(prefix + messages + suffix)

    return track


def _to_rel_time(messages: list[mido.Message]) -> Generator[mido.Message, None, None]:
    current = 0
    for msg in messages:
        delta = msg.time - current
        yield msg.copy(time=int(delta))
        current = msg.time
