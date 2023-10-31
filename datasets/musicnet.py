import os
from typing import Callable

import numpy as np
from typing_extensions import Self

from audio_processing import Audio

from ..midi import MIDIAnnotation, MIDIAnnotations


class MusicNet:
    """
    MusicNetデータセットの楽曲, ラベルのパスを保存するクラス
    """

    def __init__(self, root_dir: str) -> None:
        """
        Args:
            root_dir (str): MusicNetのルートディレクトリ
        """
        self.root_dir = root_dir

        train_data_paths = list(
            sorted(
                os.path.join(self.root_dir, "train_data", path)
                for path in os.listdir(os.path.join(self.root_dir, "train_data"))
            )
        )
        train_label_paths = list(
            sorted(
                os.path.join(self.root_dir, "train_labels", path)
                for path in os.listdir(os.path.join(self.root_dir, "train_labels"))
            )
        )
        test_data_paths = list(
            sorted(
                os.path.join(self.root_dir, "test_data", path)
                for path in os.listdir(os.path.join(self.root_dir, "test_data"))
            )
        )
        test_label_paths = list(
            sorted(
                os.path.join(self.root_dir, "test_labels", path)
                for path in os.listdir(os.path.join(self.root_dir, "test_labels"))
            )
        )

        self.__train_data_paths = train_data_paths
        self.__train_label_paths = train_label_paths
        self.__test_data_paths = test_data_paths
        self.__test_label_paths = test_label_paths

    @property
    def train_data_paths(self) -> list[str]:
        """
        学習用データのパスのリスト
        """
        return self.__train_data_paths

    @property
    def train_label_paths(self) -> list[str]:
        """
        学習用ラベルのパスのリスト
        """
        return self.__train_label_paths

    @property
    def test_data_paths(self) -> list[str]:
        """
        テスト用データのパスのリスト
        """
        return self.__test_data_paths

    @property
    def test_label_paths(self) -> list[str]:
        """
        テスト用ラベルのパスのリスト
        """
        return self.__test_label_paths

    def split(
        self, train_ratio: float, shuffle: bool = True
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        学習用データセットを学習用・検証用に分割します

        Args:
            train_ratio (float): 学習用データセットの割合. 範囲は (0, 1)
            shuffle (bool, optional): 分割するときにシャッフルするかどうか

        Returns:
            tuple[list[str], list[str], list[str], list[str]]: 学習用データパス, 学習用ラベルパス, 検証用データパス, 検証用ラベルパス
        """
        indices = list(range(len(self.train_data_paths)))
        train_point = int(len(self.train_data_paths) * train_ratio)

        if shuffle:
            indices = np.random.permutation(indices)

        train_indices = indices[:train_point]
        valid_indices = indices[train_point:]

        train_data_paths = list(map(lambda i: self.train_data_paths[i], train_indices))
        train_label_paths = list(
            map(lambda i: self.train_label_paths[i], train_indices)
        )

        valid_data_paths = list(map(lambda i: self.train_data_paths[i], valid_indices))
        valid_label_paths = list(
            map(lambda i: self.train_label_paths[i], valid_indices)
        )

        return train_data_paths, train_label_paths, valid_data_paths, valid_label_paths

    def search(self, *music_ids: int) -> tuple[list[str], list[str]]:
        """
        学習・検証・テストから指定したidのデータとアノテーションのパスを検索します.

        Args:
            music_ids (tuple[int, ...]): 検索するidのリスト

        Returns:
            tuple[list[str], list[str]]: 検索したデータとアノテーションのパスの組
        """
        all_data_paths = self.train_data_paths + self.test_data_paths
        all_label_paths = self.train_label_paths + self.test_label_paths

        return (
            list(
                filter(lambda path: self.path_to_id(path) in music_ids, all_data_paths)
            ),
            list(
                filter(lambda path: self.path_to_id(path) in music_ids, all_label_paths)
            ),
        )

    @staticmethod
    def path_to_id(path: str) -> int:
        """
        パスからMusicNetの楽曲番号を抽出します

        Args:
            path (str): 抽出するパス

        Returns:
            int: MusicNetの楽曲番号
        """
        return int(os.path.basename(path)[: -len(os.path.splitext(path)[1])])


class MusicNetNPZ:
    def __init__(self, train_set_dir: str, test_set_dir: str) -> None:
        """
        Args:
            train_set_dir (str): 学習用データのルートディレクトリ
            test_set_dir (str): テスト用データのルートディレクトリ
        """
        self.train_paths = list(
            map(
                lambda path: os.path.join(train_set_dir, path),
                filter(
                    lambda path: os.path.splitext(path)[1] == ".npz",
                    os.listdir(train_set_dir),
                ),
            )
        )
        self.test_paths = list(
            map(
                lambda path: os.path.join(test_set_dir, path),
                filter(
                    lambda path: os.path.splitext(path)[1] == ".npz",
                    os.listdir(test_set_dir),
                ),
            )
        )

    def split(
        self, train_ratio: float, shuffle: bool = True
    ) -> tuple[list[str], list[str]]:
        """
        学習用データセットを学習用・検証用に分割します

        Args:
            train_ratio (float): 学習用データセットの割合. 範囲は (0, 1)
            shuffle (bool, optional): 分割するときにシャッフルするかどうか

        Returns:
            tuple[list[str], list[str], list[str], list[str]]: 学習用データパス, 学習用ラベルパス, 検証用データパス, 検証用ラベルパス
        """
        assert train_ratio > 0 and train_ratio < 1

        indices = list(range(len(self.train_paths)))
        train_point = int(len(self.train_paths) * train_ratio)

        if shuffle:
            indices = np.random.permutation(indices)

        train_indices = indices[:train_point]
        valid_indices = indices[train_point:]

        train_splitted_paths = list(map(lambda i: self.train_paths[i], train_indices))
        valid_splitted_paths = list(map(lambda i: self.train_paths[i], valid_indices))

        return train_splitted_paths, valid_splitted_paths


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


def data_procedure(
    path: str,
    frame_shift: int,
    fmin: float,
    octaves: int,
    bins_per_octave: int,
    power: bool,
) -> np.ndarray:
    data = (
        Audio.read(path)
        .to_cqt(
            frame_shift=frame_shift,
            fmin=fmin,
            n_bins=octaves * bins_per_octave,
            bins_per_octave=bins_per_octave,
        )
        .to_amplitude()
    )
    data = data.linear_to_power() if power else data

    return data.frame_series


def label_procedure(
    path: str, frame_shift: int, fs: int, num_frames: int, trim: tuple[int, int]
) -> np.ndarray:
    return MusicNetAnnotations.from_csv(path, fs).to_frames(
        fs=fs, num_frames=num_frames, frame_shift=frame_shift
    )[:, slice(*trim)]


def create_procedure(
    frame_shift: int,
    fmin: float,
    octaves: int,
    bins_per_octave: int,
    power: bool = False,
    fs: int = 16000,
    trim: tuple[int, int] = (21, 109),
) -> Callable[[str, str], tuple[np.ndarray, np.ndarray]]:
    def procedure(data_path: str, label_path: str) -> tuple[np.ndarray, np.ndarray]:
        data = data_procedure(
            data_path, frame_shift, fmin, octaves, bins_per_octave, power
        )
        label = label_procedure(label_path, frame_shift, fs, data.shape[0], trim)
        return data, label

    return procedure


if __name__ == "__main__":
    import os

    import librosa

    from .mpe_datasets import NNMPEDatasetOnMemory

    cqt_parameters = {
        "frame_shift": 256,
        "fmin": librosa.note_to_hz("A0"),
        "octaves": 8,
        "bins_per_octave": 36,
        "power": False,
        "fs": 16000,
        "trim": (21, 109),
    }
    frames = 64

    cqt_file_name = "cqt_s{}_o{}_b{}_p{}".format(
        cqt_parameters["frame_shift"],
        cqt_parameters["octaves"],
        cqt_parameters["bins_per_octave"],
        cqt_parameters["power"],
    )

    mun = MusicNet("./resources/musicnet16k")

    procedure = create_procedure(**cqt_parameters)

    train_save_dir = "./resources/datasets/musicnet/cqt_s{}_o{}_b{}_p{}".format(
        cqt_parameters["frame_shift"],
        cqt_parameters["octaves"],
        cqt_parameters["bins_per_octave"],
        cqt_parameters["power"],
    )

    # 学習データの生成
    for train_data_path, train_label_path in zip(
        mun.train_data_paths, mun.train_label_paths
    ):
        save_file_path = os.path.join(
            train_save_dir,
            "train",
            os.path.basename(train_data_path)[
                : -len(os.path.splitext(train_data_path)[1])
            ]
            + ".npz",
        )

        if os.path.exists(save_file_path):
            continue

        print("processing {}, {}".format(train_data_path, train_label_path))
        dataset = NNMPEDatasetOnMemory.construct(
            train_data_path, train_label_path, frames, procedure
        )
        dataset.save_to_npz(save_file_path)
        print("save completed {}, {}".format(train_data_path, train_label_path))

    # テストデータの生成
    for test_data_path, test_label_path in zip(
        mun.test_data_paths, mun.test_label_paths
    ):
        save_file_path = os.path.join(
            train_save_dir,
            "test",
            os.path.basename(test_data_path)[
                : -len(os.path.splitext(test_data_path)[1])
            ]
            + ".npz",
        )

        if os.path.exists(save_file_path):
            continue

        print("processing {}, {}".format(test_data_path, test_label_path))
        dataset = NNMPEDatasetOnMemory.construct(
            test_data_path, test_label_path, frames, procedure
        )
        dataset.save_to_npz(save_file_path)
        print("save completed {}, {}".format(test_data_path, test_label_path))
