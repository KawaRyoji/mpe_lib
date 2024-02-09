import abc
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from typing_extensions import Self, override


class NPZConcatDataset(ConcatDataset, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_npz(cls, paths: list[str]) -> Self:
        raise NotImplementedError()


class N1MPEDataset(Dataset):
    """
    Multi-Pitch Estimation 用のオンメモリデータセット.
    このデータセットはスペクトログラムの中心1フレームの音高推定を行うためのN対1データセットです.
    N対Nのデータセットは`NNMPEDatasetOnMemory`を参照してください.

    このデータセットが保存するのは一曲分のデータです.
    データは`context_frame`分のパディングを前後に挿入してください.
    """

    def __init__(self, data: np.ndarray, label: np.ndarray, context_frame: int) -> None:
        """
        Args:
            data (np.ndarray): 一曲分のデータ (スペクトログラムを想定)
            label (np.ndarray): 一曲分のラベル
            context_frame (int): コンテキストフレーム. データの形状は `(context_frame * 2 + 1, F)` になります.
        """
        super().__init__()

        # context_frame 分のパディングを除くと、データの時間長とラベルの時間長は等しくなる
        assert data.shape[0] - 2 * context_frame == label.shape[0]

        self.data = data
        self.label = label
        self.context_frame = context_frame

    @classmethod
    def from_npz(cls, path: str) -> Self:
        file = np.load(path, allow_pickle=True)

        return cls(file["x"], file["y"], file["context_frame"])

    @classmethod
    def construct(
        cls,
        data_path: str,
        label_path: str,
        context_frame: int,
        constructor: Callable[[str, str], tuple[np.ndarray, np.ndarray]],
    ) -> Self:
        data, label = constructor(data_path, label_path)
        return cls(data, label, context_frame)

    def save_to_npz(self, path: str, compress: bool = False) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if compress:
            np.savez_compressed(
                path, x=self.data, y=self.label, context_frame=self.context_frame
            )
        else:
            np.savez(path, x=self.data, y=self.label, context_frame=self.context_frame)

    def __len__(self) -> int:
        return self.data.shape[0] - 2 * self.context_frame

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        data = self.data[index : index + 2 * self.context_frame + 1]
        label = self.label[index]

        return data, label


class N1MPEConcatDataset(NPZConcatDataset):
    def __init__(self, datasets: Iterable[N1MPEDataset]) -> None:
        assert all(map(lambda dataset: isinstance(dataset, N1MPEDataset), datasets))
        super().__init__(datasets)

    @classmethod
    @override
    def from_npz(cls, paths: list[str]) -> Self:
        return cls(list(map(lambda path: N1MPEDataset.from_npz(path), paths)))

    @classmethod
    def construct(
        cls,
        data_paths: list[str],
        label_paths: list[str],
        context_frame: int,
        procedure: Callable[[str, str], tuple[np.ndarray, np.ndarray]],
    ) -> Self:
        return cls(
            list(
                map(
                    lambda path: N1MPEDataset.construct(
                        *path, context_frame, procedure
                    ),
                    zip(data_paths, label_paths),
                )
            )
        )


class NNMPEDataset(Dataset):
    """
    Multi-Pitch Estimation 用のオンメモリデータセット.
    このデータセットはスペクトログラムのフレーム数分音高推定を行うためのN対Nデータセットです.
    N対1のデータセットは`N1MPEDatasetOnMemory`を参照してください.

    このデータセットが保存するのは一曲分のデータです.
    """

    def __init__(self, data: np.ndarray, label: np.ndarray, frames: int) -> None:
        """
        Args:
            data (np.ndarray): 一曲分のデータ (スペクトログラムを想定)
            label (np.ndarray): 一曲分のラベル
            frames (int): フレーム数. データの形状は `(frames, F)` になります.
        """
        super().__init__()

        # データの時間長とラベルの時間長は等しくなる
        assert (
            data.shape[0] == label.shape[0]
        ), "data shape: {}, label shape: {}".format(data.shape, label.shape)

        self.data = data
        self.label = label
        self.frames = frames

    @classmethod
    def from_npz(cls, path: str) -> Self:
        file = np.load(path, allow_pickle=True)

        return cls(file["x"], file["y"], file["frames"])

    @classmethod
    def construct(
        cls,
        data_path: str,
        label_path: str,
        frames: int,
        procedure: Callable[[str, str], tuple[np.ndarray, np.ndarray]],
    ) -> Self:
        data, label = procedure(data_path, label_path)
        return cls(data, label, frames)

    def save_to_npz(self, path: str, compress: bool = False) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if compress:
            np.savez_compressed(path, x=self.data, y=self.label, frames=self.frames)
        else:
            np.savez(path, x=self.data, y=self.label, frames=self.frames)

    def __len__(self) -> int:
        return self.data.shape[0] - self.frames

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        data = self.data[index : index + self.frames]
        label = self.label[index : index + self.frames]

        return data, label


class NNMPEConcatDataset(NPZConcatDataset):
    def __init__(self, datasets: Iterable[NNMPEDataset]) -> None:
        assert all(map(lambda dataset: isinstance(dataset, NNMPEDataset), datasets))
        super().__init__(datasets)

    @classmethod
    @override
    def from_npz(cls, paths: list[str]) -> Self:
        return cls(list(map(lambda path: NNMPEDataset.from_npz(path), paths)))

    @classmethod
    def construct(
        cls,
        data_paths: list[str],
        label_paths: list[str],
        frames: int,
        procedure: Callable[[str, str], tuple[np.ndarray, np.ndarray]],
    ) -> Self:
        return cls(
            list(
                map(
                    lambda path: N1MPEDataset.construct(*path, frames, procedure),
                    zip(data_paths, label_paths),
                )
            )
        )
