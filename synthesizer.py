import abc
import itertools
import re
from functools import reduce
from typing import List, Optional, Union

import numpy as np
from typing_extensions import Self


class Message:
    def __init__(
        self,
        track: int,
        onset: float,
        duration: float,
        frequency: float,
        amplitude: float,
    ) -> None:
        assert (
            track >= 0
            and onset >= 0
            and duration >= 0
            and frequency >= 0
            and amplitude >= 0
            and amplitude <= 1
        ), "Message の引数に期待されていない値が入力されました"

        self.__track = track
        self.__onset = float(onset)
        self.__duration = float(duration)
        self.__frequency = float(frequency)
        self.__amplitude = float(amplitude)

    @property
    def track(self) -> int:
        return self.__track

    @property
    def onset(self) -> float:
        return self.__onset

    @property
    def duration(self) -> float:
        return self.__duration

    @property
    def frequency(self) -> float:
        return self.__frequency

    @property
    def amplitude(self) -> float:
        return self.__amplitude

    @classmethod
    def from_midi(
        cls,
        track: int,
        onset: float,
        duration: float,
        note_number: int,
        amplitude: float,
    ) -> Self:
        assert note_number >= 0 and note_number <= 127
        frequency = 440 * 2 ** ((note_number - 69) / 12)

        return cls(track, onset, duration, frequency, amplitude)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"track:\t\t{self.track}",
                f"onset:\t\t{self.onset} sec",
                f"duration:\t{self.duration} sec",
                f"frequency:\t{self.frequency} Hz",
                f"amplitude:\t{self.amplitude}",
            ]
        )


class Score:
    def __init__(self, messages: List[Message] = []) -> None:
        self.__messages: List[Message] = messages

    @property
    def messages(self) -> List[Message]:
        return self.__messages

    @classmethod
    def from_numpy(
        cls,
        notes: np.ndarray,
        fs: float,
        frame_shift: int,
        amplitude: float,
        track: int,
    ) -> Self:
        # notesをstringとして保存する
        cast_notes = notes.astype(int).astype(str).T
        notes_strings = list(map(lambda row: "".join(row), cast_notes))

        # 1が連続して出てくる区間を探索し, その区間に変換する
        # 最終的な出力は {note_on, note_off, note_number}のdict形式になる
        matched_notes = list(
            itertools.chain.from_iterable(
                map(
                    lambda arg: map(
                        lambda match: dict(
                            zip(
                                ["note_on", "note_off", "note_number"],
                                (*match.span(), arg[0]),
                            )
                        ),
                        re.finditer(r"1+", arg[1]),
                    ),
                    enumerate(notes_strings),
                ),
            )
        )

        delta = frame_shift / fs

        messages = list(
            map(
                lambda d: Message.from_midi(
                    track=track,
                    onset=d["note_on"] * delta,
                    duration=(d["note_off"] - d["note_on"]) * delta,
                    note_number=d["note_number"],
                    amplitude=amplitude,
                ),
                matched_notes,
            )
        )

        return cls(messages)

    def time_length(self) -> float:
        return max(map(lambda message: message.onset + message.duration, self.messages))

    def copy_with(self, messages: Optional[List[Message]] = None) -> Self:
        return Score(messages=self.messages if messages is None else messages)

    def add(self, message: Message) -> Self:
        return self.copy_with(messages=self.messages + [message])

    def __str__(self) -> str:
        return "track, onset, dur, freq, amp\n" + "\n".join(
            [
                "{}, {}, {}, {}, {}".format(
                    message.track,
                    message.onset,
                    message.duration,
                    message.frequency,
                    message.amplitude,
                )
                for message in self.__messages
            ]
        )


class PeriodicSignal(abc.ABC):
    @abc.abstractmethod
    def generate(
        self, fs: float, frequency: float, amplitude: float, duration: float
    ) -> np.ndarray:
        raise NotImplementedError()


class AperiodicSignal(abc.ABC):
    @abc.abstractmethod
    def generate(self, fs: float, amplitude: float, duration: float) -> np.ndarray:
        raise NotImplementedError()


class SineWave(PeriodicSignal):
    def generate(
        self, fs: float, frequency: float, amplitude: float, duration: float
    ) -> np.ndarray:
        duration_point = int(fs * duration)
        t = np.arange(duration_point)

        return amplitude * np.sin(2 * np.pi * t * frequency / fs)


class Synthesizer:
    def __init__(self, fs: float, master_volume: float) -> None:
        self.__fs = fs
        self.__master_volume = master_volume

    @property
    def fs(self) -> float:
        return self.__fs

    @property
    def master_volume(self) -> float:
        return self.__master_volume

    def synth(
        self,
        score: Score,
        source: Union[PeriodicSignal, AperiodicSignal] = SineWave(),
        fade_time: float = 0.01,  # デフォルト: 10ms
    ) -> np.ndarray:
        time_length_sec = score.time_length()
        time_length_point = int(np.ceil(self.fs * time_length_sec))

        buffer = np.zeros(time_length_point)

        # 周期信号の場合と非周期信号の場合で生成を分ける
        # (生成した信号, onsetのポイント)をlistにしている
        signal_with_onset = list(
            map(
                lambda message: {
                    "signal": source.generate(
                        self.fs, message.frequency, message.amplitude, message.duration
                    ),
                    "onset": int(message.onset * self.fs),
                }
                if isinstance(source, PeriodicSignal)
                else {
                    "signal": source.generate(
                        self.fs, message.amplitude, message.duration
                    ),
                    "onset": int(message.onset * self.fs),
                },
                score.messages,
            )
        )

        processed_signals = list(
            map(
                lambda d: {
                    "signal": self.apply_fade(d["signal"], fade_time),
                    "onset": d["onset"],
                },
                signal_with_onset,
            )
        )

        # bufferに生成した信号を足す
        synthetic_sound = reduce(
            lambda buf, d: buf
            + np.pad(
                d["signal"],
                (
                    d["onset"],
                    time_length_point - (d["onset"] + len(d["signal"])),
                ),
            ),
            processed_signals,
            buffer,
        )

        return synthetic_sound / np.max(np.abs(synthetic_sound)) * self.master_volume

    def apply_fade(self, signal: np.ndarray, fade_time: float) -> np.ndarray:
        fade_point = int(fade_time * self.fs)
        t = np.arange(fade_point)
        func = np.ones(len(signal))

        func[:fade_point] = 1 / fade_point * t
        func[-fade_point:] = -1 / fade_point * t + 1

        return signal * func
