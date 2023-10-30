from functools import reduce
from typing import Any, Callable, Generator, Literal, Optional, TypeVar

import librosa
import mido
import numpy as np
from typing_extensions import Self, TypedDict

from audio_processing.utils import center_frame_samples

_T = TypeVar("_T")


class MIDIAnnotation(TypedDict):
    """
    Multi-pitch estimationで用いられるMIDIのアノテーション
    """

    note_on: float
    """ノートオンのタイミング (sec)"""
    note_off: float
    """ノートオフのタイミング (sec)"""
    note_number: int
    """ノートオンからノートオフの間に鳴っている音高番号"""
    instrument: int
    """ノートオンからノートオフの間に鳴っている楽器番号"""


class MIDIAnnotations(list[MIDIAnnotation]):
    """
    MIDIファイルのアノテーションを扱うクラスです
    """

    @staticmethod
    def pre_processing(
        track: mido.MidiTrack, filter: list[str]
    ) -> list[dict[str, Any]]:
        times = 0
        events = []
        sustain = False

        for message in track:
            times += message.time

            if message.type == "control_change" and message.control == 64:
                sustain = message.value >= 64

            if message.type in filter or message.is_meta:
                event = message.dict()
                event.update(time=times, sustain=sustain)
                events.append(event)

        return events

    @classmethod
    def from_midi(cls, file_path: str) -> Self:
        """
        MIDIファイルからインスタンスを生成します

        Args:
            file_path (str): MIDIファイルへのパス

        Returns:
            Self: 生成したインスタンス
        """
        file = mido.MidiFile(file_path)
        track = mido.merge_tracks(file.tracks)
        track = cls.pre_processing(
            track, ["program_change", "control_change", "note_on", "note_off"]
        )

        ticks_per_beat = file.ticks_per_beat
        tempo = 500000
        instrument = 0
        annotations = cls()

        for i, message in enumerate(track):
            if message["type"] == "set_tempo":
                tempo = message["tempo"]
            elif message["type"] == "program_change":
                instrument = message["program"]
            elif message["type"] == "note_on":
                # ここでは note_on に着目し, note_onであれば
                # その音高の次のnote_onを探して, note_offとする

                if message["velocity"] == 0:  # velocity = 0 は note_off を表すためスキップ
                    continue

                # 次に出現する同じ音高のメッセージがnote_offになる
                index, note_off = next(
                    filter(
                        lambda x: x[1]["type"] == "note_on"
                        and x[1]["note"] == message["note"]
                        or x[1] is track[-1],
                        enumerate(track[i + 1 :]),
                    )
                )

                if note_off["sustain"] and note_off is not track[-1]:
                    # sustain (ペダル)がonの場合, note_off はペダルが離されるまで伸びる
                    note_off = next(
                        filter(
                            lambda x: x["type"] == "control_change"
                            and x["control"] == 64
                            and x["value"] < 64
                            or x is track[-1],
                            track[i + index + 1 :],
                        )
                    )

                annotations.append(
                    MIDIAnnotation(
                        note_on=mido.tick2second(
                            message["time"], ticks_per_beat, tempo
                        ),
                        note_off=mido.tick2second(
                            note_off["time"], ticks_per_beat, tempo
                        ),
                        note_number=message["note"],
                        instrument=instrument,
                    )
                )

        return annotations

    @classmethod
    def from_numpy(
        cls, notes: np.ndarray, fs: int, frame_shift: int, instrument: int = 0
    ) -> Self:
        messages = notes_to_span(
            notes,
            lambda note_on, note_off, note_num: MIDIAnnotation(
                note_on=note_on * frame_shift / fs,
                note_off=note_off * frame_shift / fs,
                note_number=note_num,
                instrument=instrument,
            ),
        )

        return cls(messages)

    def length(self) -> float:
        """
        MIDIの演奏時間を取得します

        Returns:
            float: 演奏時間
        """
        return max(map(lambda m: m["note_off"], self))

    def search_exist_notes(self, sec: float) -> Self:
        """
        指定した秒数に存在するノートを返します

        Args:
            sec (float): 検索する秒数

        Raises:
            ValueError: 指定した秒数が不正の場合

        Returns:
            Self: 指定した秒数に存在するノート
        """
        if sec < 0:
            raise ValueError()

        return MIDIAnnotations(
            filter(lambda x: x["note_on"] <= sec and x["note_off"] >= sec, self)
        )

    def search(
        self, kind: Literal["note_on", "note_off"], sec_start: float, sec_end: float
    ) -> Self:
        """
        指定した秒数にあるアノテーションを返します

        Args:
            kind (Literal[&quot;note_on&quot;, &quot;note_off&quot;]): 'note_on'または'note_off'のどちらを探すか
            sec_start (float): 探し始めの秒数
            sec_end (float): 探し終わりの秒数

        Returns:
            Self: 指定した秒数にあるアノテーション
        """
        return MIDIAnnotations(
            filter(lambda x: x[kind] >= sec_start and x[kind] < sec_end, self)
        )

    def to_frames(
        self,
        fs: int,
        num_frames: int,
        frame_shift: int,
        frame_length: Optional[int] = None,
        offset: int = 0,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        音高アノテーションをフレーム単位に整形します.
        音高はベクトル形式となり, 鳴っている音高は1, 鳴っていない音高は0となります.
        返されるのは (フレーム数, 128) の二次元のnp.ndarrayです.

        Args:
            fs (int): サンプリング周波数
            num_frames (int): フレーム数
            frame_shift (int): フレームシフト (sample)
            frame_length (Optional[int], optional): フレーム長 (sample). 指定されていない場合, 0 からフレーム長ごとのサンプルを検索します.
            offset (int, optional): 検索を開始する位置のオフセット
            dtype (_type_, optional): 返されるnp.ndarrayのデータタイプ

        Returns:
            np.ndarray: フレーム単位に整形されたアノテーション
        """
        if frame_length is None:
            center_samples = np.fromiter(
                map(lambda i: i * frame_shift, range(num_frames)), int
            )
        else:
            center_samples = center_frame_samples(
                num_frames, frame_length, frame_shift, offset=offset
            )

        notes = [
            self.note_to_vector(
                np.array(
                    [x["note_number"] for x in self.search_exist_notes(sample)],
                    dtype=int,
                )
            )
            for sample in center_samples / fs
        ]

        return np.array(notes, dtype=dtype)

    def to_onset_and_offset(
        self,
        fs: int,
        num_frames: int,
        frame_shift: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        to_note_vector = lambda kind: lambda i: self.note_to_vector(
            np.fromiter(
                map(
                    lambda x: x["note_number"],
                    self.search(
                        kind,
                        (i * frame_shift) / fs,
                        ((i + 1) * frame_shift) / fs,
                    ),
                ),
                dtype=int,
            ),
        ).reshape((1, -1))

        onsets = np.concatenate(
            list(map(to_note_vector("note_on"), range(num_frames))),
            axis=0,
            dtype=np.float32,
        )
        offsets = np.concatenate(
            list(map(to_note_vector("note_off"), range(num_frames))),
            axis=0,
            dtype=np.float32,
        )

        return onsets, offsets

    @staticmethod
    def note_to_vector(note: np.ndarray) -> np.ndarray:
        """
        音高番号をベクトル形式にします.

        Args:
            note (np.ndarray): 音高番号 (複数可)

        Returns:
            np.ndarray: ベクトル形式の音高番号
        """
        vector = np.zeros(128)
        vector[note] = 1
        vector = vector.astype(np.float32)

        return vector


def filter_message(track: mido.MidiTrack, filter: list[str]) -> mido.MidiTrack:
    """
    `mido.MidiTrack`から指定したメッセージタイプでフィルタリングした`mido.MidiTrack`を返します.

    Args:
        track (mido.MidiTrack): フィルタリングするトラック
        filter (list[str]): フィルタリングしたいメッセージタイプのリスト

    Returns:
        mido.MidiTrack: フィルタリングされたトラック
    """
    times = 0
    messages = []

    for message in track:
        if message.type in filter or message.is_meta:
            messages.append(message.copy(time=message.time + times))
            times = 0
        else:
            times += message.time

    return mido.MidiTrack(messages)


def notes_to_span(
    notes: np.ndarray, constructor: Callable[[int, int, int], _T]
) -> list[_T]:
    """
    numpy配列に保存されている音高((T, 128)を想定)をnote_on, note_offの区間に変換します

    Args:
        notes (np.ndarray): 音高の配列
        constructor (Callable[[int, int, int], _T]): 区間と音高からオブジェクトを作成するコンストラクタ.
        このクロージャは`constructor(note_on_index, note_off_index, note_number)`として呼び出されます

    Returns:
        list[_T]: `constructor`で変換したオブジェクトの配列
    """
    import re

    # notesをstringとして保存する
    cast_notes = notes.astype(int).astype(str).T
    notes_strings = list(map(lambda row: "".join(row), cast_notes))

    init: list[_T] = []

    # 1が連続して出てくる区間を探索し, その区間に変換する
    messages = reduce(
        lambda x, y: x
        + list(
            map(
                lambda match: constructor(*match.span(), y[0]),
                re.finditer(r"1+", y[1]),
            )
        ),
        enumerate(notes_strings),
        init,
    )

    return messages


def numpy_to_track(
    notes: np.ndarray,
    fs: float,
    frame_shift: int,
    velocity: int = 127,
    instrument: int = 0,
    tempo: int = 500000,
    ticks_per_beat: int = 480,
) -> mido.MidiTrack:
    import itertools

    messages = notes_to_span(
        notes,
        lambda note_on, note_off, note_num: (  # note on message
            mido.Message(
                "note_on",
                note=note_num,
                velocity=velocity,
                time=mido.second2tick(
                    note_on * frame_shift / fs,
                    ticks_per_beat,
                    tempo,
                ),
            ),
            # note off message (velocity=0)
            mido.Message(
                "note_on",
                note=note_num,
                velocity=0,
                time=mido.second2tick(
                    note_off * frame_shift / fs,
                    ticks_per_beat,
                    tempo,
                ),
            ),
        ),
    )

    messages = list(itertools.chain.from_iterable(messages))  # flatten

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


class SimpleSynthesizer:
    def __init__(self, fs: int) -> None:
        self.__fs = fs

    @property
    def fs(self) -> int:
        return self.__fs

    def sine_wave(
        self, frequency: float, amplitude: float, duration: float
    ) -> np.ndarray:
        duration_point = int(self.fs * duration)
        t = np.arange(duration_point)

        return amplitude * np.sin(2 * np.pi * t * frequency / self.fs)

    def apply_fade(self, signal: np.ndarray, fade_time: float) -> np.ndarray:
        fade_point = int(fade_time * self.fs)
        t = np.arange(fade_point)
        func = np.ones(len(signal))

        func[:fade_point] = 1 / fade_point * t
        func[-fade_point:] = -1 / fade_point * t + 1

        return signal * func

    def synth(
        self, score: MIDIAnnotations, master_volume: float, fade_time: float = 0.01
    ) -> np.ndarray:
        to_point = lambda x: int(x * self.fs)

        time_length = score.length()
        time_length_point = to_point(time_length)

        buffer = np.zeros(time_length_point)

        synthetic_sound = reduce(
            lambda buf, m: buf
            + np.pad(  # bufferと同じ長さになるようにsin波をパディングする
                self.apply_fade(
                    self.sine_wave(
                        librosa.midi_to_hz(m["note_number"]),  # frequency
                        1.0,  # amplitude
                        m["note_off"] - m["note_on"],  # duration
                    ),
                    fade_time,
                ),
                (
                    to_point(m["note_on"]),  # offset
                    time_length_point
                    - to_point(m["note_on"])
                    - to_point(m["note_off"] - m["note_on"]),
                ),
            ),
            score,
            buffer,
        )

        return synthetic_sound / np.max(np.abs(synthetic_sound)) * master_volume
