import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mido
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from typing_extensions import Self, TypedDict

from audio_processing.utils import center_frame_samples


def filter_message(track: mido.MidiTrack, filter: List[str]) -> mido.MidiTrack:
    """
    `mido.MidiTrack`から指定したメッセージタイプでフィルタリングした`mido.MidiTrack`を返します.

    Args:
        track (mido.MidiTrack): フィルタリングするトラック
        filter (List[str]): フィルタリングしたいメッセージタイプのリスト

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


class MIDIAnnotations(List[MIDIAnnotation]):
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

    def search(self, sec: float) -> Self:
        """
        指定した秒数にあるアノテーションを返します

        Args:
            sec (float): 検索する秒数

        Raises:
            ValueError: 指定した秒数が不正の場合

        Returns:
            Self: 指定した秒数にあるアノテーション
        """
        if sec < 0:
            raise ValueError()

        return MIDIAnnotations(
            filter(lambda x: x["note_on"] <= sec and x["note_off"] >= sec, self)
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
                np.array([x["note_number"] for x in self.search(sample)], dtype=int)
            )
            for sample in center_samples / fs
        ]

        return np.array(notes, dtype=dtype)

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


class DatasetConstructor(ABC):
    """
    データセットを構築する処理を行う抽象クラスです. 使用には`procedure`メソッドをオーバーライドしてください.
    """

    @abstractmethod
    def procedure(
        self, data_path: str, annotation_path: str, **params: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        データへのパスとアノテーションへのパスから所望のデータ、アノテーションへの変換処理を記述する抽象メソッドです.

        Args:
            data_path (str): データへのパス
            annotation_path (str): アノテーションへのパス

        Raises:
            NotImplementedError: 実装されていない場合

        Returns:
            Tuple[np.ndarray, np.ndarray]: データとアノテーションの組
        """
        raise NotImplementedError()

    def construct(
        self,
        data_paths: List[str],
        annotation_paths: List[str],
        dtype=np.float32,
        **params: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        `procedure`メソッドを使用して、データセットを構築します.

        Args:
            data_paths (List[str]): データへのパスのリスト
            annotation_paths (List[str]): アノテーションへのパスのリスト
            dtype (_type_, optional): 変換したデータとアノテーションのデータタイプ
            params: (Dict[str, Any]): 構築する際のパラメータ

        Returns:
            Tuple[np.ndarray, np.ndarray]: 構築したデータセット
        """
        data: List[np.ndarray] = []
        annotations: List[np.ndarray] = []

        for i, path in enumerate(zip(data_paths, annotation_paths)):
            print("\rnow processing...: {} / {}".format(i + 1, len(data_paths)), end="")
            data_path, annotation_path = path

            x, y = self.procedure(data_path, annotation_path, **params)
            data.extend(x)
            annotations.extend(y)
        print()

        data = np.array(data, dtype=dtype)
        annotations = np.array(annotations, dtype=dtype)

        return data, annotations


class MIDIDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        annotations: np.ndarray,
        params: Dict[str, Any] = {},
    ) -> None:
        """
        Args:
            data (np.ndarray): 音信号データ
            annotations (np.ndarray): 音信号データと対となるアノテーション
            params (Dict[str, Any], optional): データとアノテーションを生成したパラメータ

        Raises:
            ValueError: `data`と`annotations`の最初の次元が合わない場合
        """
        if data.shape[0] != annotations.shape[0]:
            raise ValueError("データとアノテーションの次元は一致させてください")

        super().__init__()

        self.data = data
        self.annotation = annotations
        self.params = params

    @classmethod
    def use_constructor(
        cls,
        constructor: DatasetConstructor,
        data_paths: List[str],
        annotation_paths: List[str],
        dtype=np.float32,
        param_path: Optional[str] = None,
        **params: Any
    ) -> Self:
        """
        `DatasetConstructor`を使用してデータセットを構築します.

        Args:
            constructor (DatasetConstructor): `procedure`を実装したインスタンス
            data_paths (List[str]): データへのパスのリスト
            annotation_paths (List[str]): アノテーションへのパスのリスト
            dtype (_type_, optional): 変換したデータとアノテーションのデータタイプ
            param_path (Optional[str], optional): パラメータを記述したjsonファイルへのパス. ない場合`params`が使用されます.
            params (Dict[str, Any]): 構築に使用するパラメータ

        Returns:
            Self: 構築したデータセット
        """
        if param_path is not None:
            with open(param_path, "r") as f:
                params = json.load(f)

        return cls(
            *constructor.construct(data_paths, annotation_paths, dtype=dtype, **params),
            params
        )

    @classmethod
    def load_from_npz(cls, npz_path: str, shuffle: bool = False) -> Self:
        """
        保存したnpzファイルから復元します.

        Args:
            npz_path (str): npzファイルへのパス
            shuffle (bool, optional): 読み込みの際にシャッフルするかどうか.

        Returns:
            Self: 復元したデータセット
        """
        data = np.load(npz_path, allow_pickle=True)
        x: np.ndarray = data["x"]
        y: np.ndarray = data["y"]

        if shuffle:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

        return cls(x, y)

    def to_dataloader(
        self, batch_size: int, shuffle: bool, num_workers: int = 0, **kwargs: Any
    ) -> DataLoader:
        """
        このデータセットをpytorchの`DataLoader`へ変換します.

        Args:
            batch_size (int): バッチサイズ
            shuffle (bool): シャッフルするかどうか
            num_workers (int, optional): ワーカー数
            kwargs (Dict[str, Any]): `DataLoader`のパラメータ

        Returns:
            DataLoader: _description_
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )

    def save_parameter(self, path: Union[str, Path]) -> None:
        """
        このデータセットを生成したパラメータを保存します.

        Args:
            path (Union[str, Path]): 保存先のパス (json形式)
        """
        with open(path, "w") as f:
            json.dump(self.params, f)

    def save_to_npz(
        self, path: str, compress: bool = False, save_params: bool = True
    ) -> None:
        """
        データセットをnpzファイルに保存します.

        Args:
            path (str): 保存先のパス
            compress (bool, optional): 圧縮するかどうか
            save_params (bool, optional): 生成したパラメータを保存するかどうか
        """
        if path.endswith(".npz"):
            path = path[:-4]

        directory = Path(path).parent
        Path.mkdir(directory, parents=True, exist_ok=True)

        if save_params:
            self.save_parameter(directory / (Path(path).name + ".json"))

        if compress:
            np.savez_compressed(path, x=self.data, y=self.annotation)
        else:
            np.savez(path, x=self.data, y=self.annotation)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data[index], self.annotation[index]
