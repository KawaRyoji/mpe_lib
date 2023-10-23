import os
from typing import Optional

import numpy as np
from typing_extensions import override

from audio_processing import Audio

from .annotations import MusicNetAnnotations
from .base import DatasetConstructor, DatasetSplit
from .mpe_datasets import (
    N1MPEConcatDatasetOnMemory,
    N1MPEDatasetOnMemory,
    NNMPEConcatDatasetOnMemory,
    NNMPEDatasetOnMemory,
)


class MusicNet(DatasetSplit):
    def __init__(
        self,
        root_dir: str,
        valid_music_ids: Optional[list[int]] = None,
        valid_ratio: Optional[float] = None,
    ) -> None:
        """
        Args:
            root_dir (str): MusicNetのルートディレクトリ
            valid_music_number (Optional[list[int]], optional): 検証用に用いる曲番号のリスト. 指定されている場合これが優先されます.
            valid_ratio (Optional[float], optional): 検証用に用いる曲の割合. 学習用データセットから後ろ何割が使用されます.
        """
        self.root_dir = root_dir

        train_data_paths = set(
            sorted(
                os.path.join(self.root_dir, "train_data", path)
                for path in os.listdir(os.path.join(self.root_dir, "train_data"))
            )
        )
        train_labels_paths = set(
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
        test_labels_paths = list(
            sorted(
                os.path.join(self.root_dir, "test_labels", path)
                for path in os.listdir(os.path.join(self.root_dir, "test_labels"))
            )
        )

        if valid_music_ids is not None:
            is_valid = lambda path: int(os.path.basename(path)[:-4]) in valid_music_ids
            valid_data_paths = set(filter(is_valid, train_data_paths))
            valid_labels_paths = set(filter(is_valid, train_labels_paths))

            train_data_paths = sorted(train_data_paths - valid_data_paths)
            train_labels_paths = sorted(train_labels_paths - valid_labels_paths)
            valid_data_paths = sorted(valid_data_paths)
            valid_labels_paths = sorted(valid_labels_paths)
        elif valid_ratio is not None:
            valid_point = int(valid_ratio * len(train_data_paths))

            train_data_list = sorted(train_data_paths)
            train_labels_list = sorted(train_labels_paths)

            train_data_paths = train_data_list[:-valid_point]
            train_labels_paths = train_labels_list[:-valid_point]
            valid_data_paths = train_data_list[-valid_point:]
            valid_labels_paths = train_labels_list[-valid_point:]
        else:
            train_data_paths = sorted(train_data_paths)
            train_labels_paths = sorted(train_labels_paths)
            valid_data_paths = []
            valid_labels_paths = []

        self.__train_data_paths = train_data_paths
        self.__train_labels_paths = train_labels_paths
        self.__valid_data_paths = valid_data_paths
        self.__valid_labels_paths = valid_labels_paths
        self.__test_data_paths = test_data_paths
        self.__test_labels_paths = test_labels_paths

    @property
    @override
    def train_data_paths(self) -> list[str]:
        return self.__train_data_paths

    @property
    @override
    def train_labels_paths(self) -> list[str]:
        return self.__train_labels_paths

    @property
    @override
    def valid_data_paths(self) -> list[str]:
        return self.__valid_data_paths

    @property
    @override
    def valid_labels_paths(self) -> list[str]:
        return self.__valid_labels_paths

    @property
    @override
    def test_data_paths(self) -> list[str]:
        return self.__test_data_paths

    @property
    @override
    def test_labels_paths(self) -> list[str]:
        return self.__test_labels_paths

    def search(self, *music_ids: int) -> tuple[list[str], list[str]]:
        """
        学習・検証・テストから指定したidのデータとアノテーションのパスを検索します.

        Args:
            music_ids (tuple[int, ...]): 検索するidのリスト

        Returns:
            tuple[list[str], list[str]]: 検索したデータとアノテーションのパスの組
        """
        all_data_paths = (
            self.train_data_paths + self.valid_data_paths + self.test_data_paths
        )
        all_labels_paths = (
            self.train_labels_paths + self.valid_labels_paths + self.test_labels_paths
        )

        return list(
            filter(
                lambda path: int(os.path.basename(path)[:-4]) in music_ids,
                all_data_paths,
            )
        ), list(
            filter(
                lambda path: int(os.path.basename(path)[:-4]) in music_ids,
                all_labels_paths,
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
    path: str, frame_shift: int, fs: int, num_frames: int
) -> np.ndarray:
    return MusicNetAnnotations.from_csv(path, fs).to_frames(
        fs=fs, num_frames=num_frames, frame_shift=frame_shift
    )


class N1CQTMuNDatasetConstructor(DatasetConstructor):
    @override
    def procedure(
        self,
        data_paths: list[str],
        label_paths: list[str],
        frame_shift: int,
        fmin: float,
        octaves: int,
        bins_per_octave: int,
        context_frame: int,
        power: bool = False,
        fs: int = 16000,
    ) -> N1MPEConcatDatasetOnMemory:
        def to_dataset(data_path: str, label_path: str) -> N1MPEDatasetOnMemory:
            data = data_procedure(
                path=data_path,
                frame_shift=frame_shift,
                fmin=fmin,
                octaves=octaves,
                bins_per_octave=bins_per_octave,
                power=power,
            )
            label = label_procedure(
                path=label_path,
                frame_shift=frame_shift,
                fs=fs,
                num_frames=data.shape[0],
            )

            return N1MPEDatasetOnMemory(data, label, context_frame)

        return N1MPEConcatDatasetOnMemory(map(to_dataset, zip(data_paths, label_paths)))


class NNCQTMuNDatasetConstructor(DatasetConstructor):
    @override
    def procedure(
        self,
        data_paths: list[str],
        label_paths: list[str],
        frame_shift: int,
        fmin: float,
        octaves: int,
        bins_per_octave: int,
        frames: int,
        power: bool = False,
        fs: int = 16000,
    ) -> NNMPEConcatDatasetOnMemory:
        def to_dataset(data_path: str, label_path: str) -> NNMPEDatasetOnMemory:
            data = data_procedure(
                path=data_path,
                frame_shift=frame_shift,
                fmin=fmin,
                octaves=octaves,
                bins_per_octave=bins_per_octave,
                power=power,
            )
            label = label_procedure(
                path=label_path,
                frame_shift=frame_shift,
                fs=fs,
                num_frames=data.shape[0],
            )

            return NNMPEDatasetOnMemory(data, label, frames)

        return NNMPEConcatDatasetOnMemory(map(to_dataset, zip(data_paths, label_paths)))
