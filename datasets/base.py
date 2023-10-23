import abc
from typing import Any

from torch.utils.data import Dataset


class DatasetConstructor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def procedure(
        self, data_paths: list[str], label_paths: list[str], *args: Any, **kwargs: Any
    ) -> Dataset:
        raise NotImplementedError()


class DatasetSplit(metaclass=abc.ABCMeta):
    """
    データセットのファイルパスを学習・検証・テストの3つのスプリットに分けて保存する抽象クラスです.
    DatasetConstructorを使ってDatasetを構築することができます.
    """

    @abc.abstractproperty
    def train_data_paths(self) -> list[str]:
        """
        学習用データのパスのリスト
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def train_label_paths(self) -> list[str]:
        """
        学習用アノテーションのパスのリスト
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def valid_data_paths(self) -> list[str]:
        """
        検証用データのパスのリスト
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def valid_label_paths(self) -> list[str]:
        """
        検証用アノテーションのパスのリスト
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def test_data_paths(self) -> list[str]:
        """
        テスト用データのパスのリスト
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def test_label_paths(self) -> list[str]:
        """
        テスト用アノテーションのパスのリスト
        """
        raise NotImplementedError()

    def construct_train_dataset(
        self, constructor: DatasetConstructor, *args: Any, **kwargs: Any
    ) -> Dataset:
        """
        学習用データのパスのリストから学習用データセットを構築します.
        学習・検証・テストが同じ`constructor`を用いる場合、メソッド`construct`を使用することもできます.

        Args:
            constructor (DatasetConstructor): データセットを構築するコンストラクタ

        Returns:
            Dataset: 構築したデータセット
        """
        return constructor.procedure(
            self.train_data_paths, self.train_label_paths, *args, **kwargs
        )

    def construct_valid_dataset(
        self, constructor: DatasetConstructor, *args: Any, **kwargs: Any
    ) -> Dataset:
        """
        検証用データのパスのリストから検証用データセットを構築します.
        学習・検証・テストが同じ`constructor`を用いる場合、メソッド`construct`を使用することもできます.

        Args:
            constructor (DatasetConstructor): データセットを構築するコンストラクタ

        Returns:
            Dataset: 構築したデータセット
        """
        return constructor.procedure(
            self.valid_data_paths, self.valid_label_paths, *args, **kwargs
        )

    def construct_test_dataset(
        self, constructor: DatasetConstructor, *args: Any, **kwargs: Any
    ) -> Dataset:
        """
        テスト用データのパスのリストからテスト用データセットを構築します.
        学習・検証・テストが同じ`constructor`を用いる場合、メソッド`construct`を使用することもできます.

        Args:
            constructor (DatasetConstructor): データセットを構築するコンストラクタ

        Returns:
            Dataset: 構築したデータセット
        """
        return constructor.procedure(
            self.test_data_paths, self.test_label_paths, *args, **kwargs
        )

    def construct(
        self,
        constructor: DatasetConstructor,
        train_params: dict[str, Any],
        valid_params: dict[str, Any],
        test_params: dict[str, Any],
    ) -> tuple[Dataset, Dataset, Dataset]:
        """
        学習・検証・テストのパスのリストから同一の`constructor`を使用して、それぞれのデータセットを構築します.
        学習・検証・テストが同一の処理でない場合、`construct_(train|valid|test)_dataset`メソッドを使用してください.

        Args:
            constructor (DatasetConstructor): データセットを構築するコンストラクタ
            train_params (dict[str, Any]): 学習用データセットのパラメータ
            valid_params (dict[str, Any]): 検証用データセットのパラメータ
            test_params (dict[str, Any]): テスト用データセットのパラメータ

        Returns:
            tuple[Dataset, Dataset, Dataset]: 学習用データセット, 検証用データセット, テスト用データセット
        """
        train_dataset = self.construct_train_dataset(constructor, **train_params)
        valid_dataset = self.construct_valid_dataset(constructor, **valid_params)
        test_dataset = self.construct_test_dataset(constructor, **test_params)

        return train_dataset, valid_dataset, test_dataset
