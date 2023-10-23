import os
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from typing_extensions import Self


class N1MPEDatasetOnMemory(Dataset):
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

    def save_to_npz(self, path: str, compress: bool = False) -> None:
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


class N1MPEConcatDatasetOnMemory(ConcatDataset):
    def __init__(self, datasets: Iterable[N1MPEDatasetOnMemory]) -> None:
        assert all(
            map(lambda dataset: isinstance(dataset, N1MPEDatasetOnMemory), datasets)
        )
        super().__init__(datasets)

    @classmethod
    def from_npz(cls, root_dir_path: str) -> Self:
        return cls(
            map(
                lambda path: N1MPEDatasetOnMemory.from_npz(path),
                os.listdir(root_dir_path),
            )
        )

    def save_to_npz(self, root_dir_path: str, compress: bool = False) -> None:
        root_dir = Path(root_dir_path)
        root_dir.mkdir(parents=True, exist_ok=True)

        for i, dataset in enumerate(self.datasets):
            if not isinstance(dataset, N1MPEDatasetOnMemory):
                raise TypeError()

            dataset.save_to_npz(root_dir / str(i), compress=compress)


class NNMPEDatasetOnMemory(Dataset):
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
        assert data.shape[0] == label.shape[0]

        self.data = data
        self.label = label
        self.frames = frames

    @classmethod
    def from_npz(cls, path: str) -> Self:
        file = np.load(path, allow_pickle=True)

        return cls(file["x"], file["y"], file["frames"])

    def save_to_npz(self, path: str, compress: bool = False) -> None:
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


class NNMPEConcatDatasetOnMemory(ConcatDataset):
    def __init__(self, datasets: Iterable[NNMPEDatasetOnMemory]) -> None:
        assert all(
            map(lambda dataset: isinstance(dataset, NNMPEDatasetOnMemory), datasets)
        )
        super().__init__(datasets)

    @classmethod
    def from_npz(cls, root_dir_path: str) -> Self:
        return cls(
            map(
                lambda path: NNMPEDatasetOnMemory.from_npz(path),
                os.listdir(root_dir_path),
            )
        )

    def save_to_npz(self, root_dir_path: str, compress: bool = False) -> None:
        root_dir = Path(root_dir_path)
        root_dir.mkdir(parents=True, exist_ok=True)

        for i, dataset in enumerate(self.datasets):
            if not isinstance(dataset, NNMPEDatasetOnMemory):
                raise TypeError()

            dataset.save_to_npz(root_dir / str(i), compress=compress)
