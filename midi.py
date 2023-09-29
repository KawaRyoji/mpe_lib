import itertools
from functools import reduce
import re
import mido
import numpy as np


def numpy_to_messages(
    notes: np.ndarray, fs: float, frame_shift: int, velocity: int = 127
) -> mido.MidiTrack:
    # notesをstringとして保存する
    cast_notes = notes.astype(int).astype(str).T
    notes_strings = list(map(lambda row: "".join(row), cast_notes))

    # 1が連続して出てくる区間を探索し, その区間に変換する
    # 最終的な出力は {note_on, note_off, note_number}のdict形式になる
    init: list[mido.Message] = []
    messages = reduce(
        lambda x, y: x
        + list(
            itertools.chain(
                map(
                    lambda match: (
                        # note on message
                        mido.Message(
                            "note_on",
                            note=y[0],
                            velocity=velocity,
                            time=match[0] * frame_shift / fs,
                        ),
                        # note off message (verocity=0)
                        mido.Message(
                            "note_on",
                            note=y[0],
                            velocity=0,
                            time=match[1] * frame_shift / fs,
                        ),
                    ),
                    re.finditer(r"1+", y[1]),
                )
            )
        ),
        enumerate(notes_strings),
        init,
    )

    track = mido.MidiTrack(messages)    
    
    return track


def numpy_to_midi(notes: np.ndarray):
    mido.MidiTrack()


if __name__ == "__main__":
    r = np.random.rand(64, 128)
    notes = np.where(r >= 0.8, np.ones_like(r), np.zeros_like(r))
    print(notes)
    numpy_to_messages(notes, 16000, 256)
