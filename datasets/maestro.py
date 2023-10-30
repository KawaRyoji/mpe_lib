import os
from typing import Callable

import numpy as np
import pandas as pd

from audio_processing import Audio
from mpe_lib.datasets.midi import MIDIAnnotations


class MAESTRO:
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
    def train_data_paths(self) -> list[str]:
        """
        学習用データのパスのリスト
        """
        return self.__train_data_paths

    @property
    def train_labels_paths(self) -> list[str]:
        """
        学習用アノテーションのパスのリスト
        """
        return self.__train_labels_paths

    @property
    def valid_data_paths(self) -> list[str]:
        """
        検証用データのパスのリスト
        """
        return self.__valid_data_paths

    @property
    def valid_labels_paths(self) -> list[str]:
        """
        検証用アノテーションのパスのリスト
        """
        return self.__valid_labels_paths

    @property
    def test_data_paths(self) -> list[str]:
        """
        テスト用データのパスのリスト
        """
        return self.__test_data_paths

    @property
    def test_labels_paths(self) -> list[str]:
        """
        テスト用アノテーションのパスのリスト
        """
        return self.__test_labels_paths


class MAESTRONPZ:
    def __init__(
        self, train_data_dir: str, valid_data_dir: str, test_data_dir: str
    ) -> None:
        self.train_paths = list(
            filter(
                lambda path: os.path.splitext(path)[1] == ".npz",
                os.listdir(train_data_dir),
            )
        )
        self.valid_paths = list(
            filter(
                lambda path: os.path.splitext(path)[1] == ".npz",
                os.listdir(valid_data_dir),
            )
        )
        self.test_paths = list(
            filter(
                lambda path: os.path.splitext(path)[1] == ".npz",
                os.listdir(test_data_dir),
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
    return MIDIAnnotations.from_midi(path).to_frames(
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
