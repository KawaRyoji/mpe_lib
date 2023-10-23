import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Union

import mido
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from typing_extensions import Self, TypedDict

from audio_processing.utils import center_frame_samples


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
                        ((i, n) for i, n in enumerate(track[i + 1 :])),
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
                            (n for n in track[i + index + 1 :]),
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


class MusicNetAnnotations(MIDIAnnotations):
    """
    MusicNet用のアノテーションクラス
    """

    @classmethod
    def from_csv(cls, path: str, fs: int) -> Self:
        """
        MusicNetのアノテーションファイル(.csv)からアノテーションインスタンスを生成します.

        Args:
            path (str): アノテーションファイルのパス
            fs (int): サンプリング周波数

        Returns:
            Self: 生成したアノテーションインスタンス
        """
        label = np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )

        return cls(
            map(
                lambda x: MIDIAnnotation(
                    note_on=x[0] / fs,
                    note_off=x[1] / fs,
                    instrument=x[2],
                    note_number=x[3],
                ),
                zip(
                    label["start_time"],
                    label["end_time"],
                    label["instrument"],
                    label["note"],
                ),
            )
        )
