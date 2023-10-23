import os

import numpy as np
import pandas as pd
from typing_extensions import override

from audio_processing import Audio
from mpe_lib.datasets.annotations import MIDIAnnotations
from mpe_lib.datasets.base import DatasetConstructor, DatasetSplit
from mpe_lib.datasets.mpe_datasets import (
    N1MPEConcatDatasetOnMemory,
    N1MPEDatasetOnMemory,
    NNMPEConcatDatasetOnMemory,
    NNMPEDatasetOnMemory,
)


class MAESTRO(DatasetSplit):
    def __init__(self, root_dir: str, version: int = 3) -> None:
        meta_data = pd.read_csv(
            os.path.join(root_dir, "maestro-v{}.0.0.csv".format(version))
        )

        train_split = meta_data.where(lambda x: x["split"] == "train").dropna()
        valid_split = meta_data.where(lambda x: x["split"] == "validation").dropna()
        test_split = meta_data.where(lambda x: x["split"] == "test").dropna()

        self.__train_data_paths = list(
            map(
                lambda path: os.path.join(root_dir, path), train_split["audio_filename"]
            )
        )
        self.__train_labels_paths = list(
            map(lambda path: os.path.join(root_dir, path), train_split["midi_filename"])
        )
        self.__valid_data_paths = list(
            map(
                lambda path: os.path.join(root_dir, path), valid_split["audio_filename"]
            )
        )
        self.__valid_labels_paths = list(
            map(lambda path: os.path.join(root_dir, path), valid_split["midi_filename"])
        )
        self.__test_data_paths = list(
            map(lambda path: os.path.join(root_dir, path), test_split["audio_filename"])
        )
        self.__test_labels_paths = list(
            map(lambda path: os.path.join(root_dir, path), test_split["midi_filename"])
        )

    @property
    @override
    def train_data_paths(self) -> list[str]:
        """
        学習用データのパスのリスト
        """
        return self.__train_data_paths

    @property
    @override
    def train_labels_paths(self) -> list[str]:
        """
        学習用アノテーションのパスのリスト
        """
        return self.__train_labels_paths

    @property
    @override
    def valid_data_paths(self) -> list[str]:
        """
        検証用データのパスのリスト
        """
        return self.__valid_data_paths

    @property
    @override
    def valid_labels_paths(self) -> list[str]:
        """
        検証用アノテーションのパスのリスト
        """
        return self.__valid_labels_paths

    @property
    @override
    def test_data_paths(self) -> list[str]:
        """
        テスト用データのパスのリスト
        """
        return self.__test_data_paths

    @property
    @override
    def test_labels_paths(self) -> list[str]:
        """
        テスト用アノテーションのパスのリスト
        """
        return self.__test_labels_paths


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
    return MIDIAnnotations.from_midi(path).to_frames(
        fs=fs, num_frames=num_frames, frame_shift=frame_shift
    )


class N1CQTMaestroDatasetConstructor(DatasetConstructor):
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


class NNCQTMaestroDatasetConstructor(DatasetConstructor):
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
