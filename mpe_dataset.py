import abc
import os
from typing import List, Optional, Tuple

import pandas as pd
from typing_extensions import override


class MPEDataset(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def train_data_paths(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractproperty
    def train_labels_paths(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractproperty
    def valid_data_paths(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractproperty
    def valid_labels_paths(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractproperty
    def test_data_paths(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractproperty
    def test_labels_paths(self) -> List[str]:
        raise NotImplementedError()


class MusicNet(MPEDataset):
    def __init__(
        self,
        root_dir: str,
        valid_music_ids: Optional[List[int]] = None,
        valid_ratio: Optional[float] = None,
    ) -> None:
        """
        Args:
            root_dir (str): MusicNetのルートディレクトリ
            valid_music_number (Optional[List[int]], optional): 検証用に用いる曲番号のリスト. 指定されている場合これが優先されます.
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
    def train_data_paths(self) -> List[str]:
        """
        学習用データのパスのリスト
        """
        return self.__train_data_paths

    @property
    @override
    def train_labels_paths(self) -> List[str]:
        """
        学習用アノテーションのパスのリスト
        """
        return self.__train_labels_paths

    @property
    @override
    def valid_data_paths(self) -> List[str]:
        """
        検証用データのパスのリスト
        """
        return self.__valid_data_paths

    @property
    @override
    def valid_labels_paths(self) -> List[str]:
        """
        検証用アノテーションのパスのリスト
        """
        return self.__valid_labels_paths

    @property
    @override
    def test_data_paths(self) -> List[str]:
        """
        テスト用データのパスのリスト
        """
        return self.__test_data_paths

    @property
    @override
    def test_labels_paths(self) -> List[str]:
        """
        テスト用アノテーションのパスのリスト
        """
        return self.__test_labels_paths

    def search(self, *music_ids: int) -> Tuple[List[str], List[str]]:
        """
        学習・検証・テストから指定したidのデータとアノテーションのパスを検索します.

        Args:
            music_ids (Tuple[int, ...]): 検索するidのリスト

        Returns:
            Tuple[List[str], List[str]]: 検索したデータとアノテーションのパスの組
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


class MAESTRO(MPEDataset):
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
    def train_data_paths(self) -> List[str]:
        """
        学習用データのパスのリスト
        """
        return self.__train_data_paths

    @property
    @override
    def train_labels_paths(self) -> List[str]:
        """
        学習用アノテーションのパスのリスト
        """
        return self.__train_labels_paths

    @property
    @override
    def valid_data_paths(self) -> List[str]:
        """
        検証用データのパスのリスト
        """
        return self.__valid_data_paths

    @property
    @override
    def valid_labels_paths(self) -> List[str]:
        """
        検証用アノテーションのパスのリスト
        """
        return self.__valid_labels_paths

    @property
    @override
    def test_data_paths(self) -> List[str]:
        """
        テスト用データのパスのリスト
        """
        return self.__test_data_paths

    @property
    @override
    def test_labels_paths(self) -> List[str]:
        """
        テスト用アノテーションのパスのリスト
        """
        return self.__test_labels_paths
