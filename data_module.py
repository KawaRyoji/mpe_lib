from typing import Generator

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset


class HoldOutDataModule(LightningDataModule):
    """
    学習・検証・テストを行うデータモジュール
    """

    def __init__(
        self,
        train_set: Dataset,
        val_set: Dataset,
        test_set: Dataset,
        batch_size: int,
        num_workers: int,
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        test_shuffle: bool = False,
    ) -> None:
        """
        Args:
            train_set (Dataset): 学習用データセット
            val_set (Dataset): 検証用データセット
            test_set (Dataset): テスト用データセット
            batch_size (int): バッチサイズ
            num_workers (int): ワーカー数
            train_shuffle (bool, optional): 学習用データセットをシャッフルするかどうか
            val_shuffle (bool, optional): 検証用データセットをシャッフルするかどうか
            test_shuffle (bool, optional): テスト用データセットをシャッフルするかどうか
        """
        super().__init__()

        self.train_set = train_set
        self.valid_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.valid_shuffle = val_shuffle
        self.test_shuffle = test_shuffle

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=self.valid_shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
        )


class KFoldDataModuleGenerator:
    """
    K分割交差検証のデータを生成するジェネレータ.
    """

    def __init__(
        self,
        k: int,
        train_set: Dataset,
        test_set: Dataset,
        batch_size: int,
        num_workers: int,
        train_shuffle: bool = True,
        valid_shuffle: bool = False,
        test_shuffle: bool = False,
    ) -> None:
        """
        Args:
            k (int): 何分割するか
            train_set (Dataset): 学習用データセット
            test_set (Dataset): テスト用データセット
            batch_size (int): バッチサイズ
            num_workers (int): ワーカー数
            train_shuffle (bool, optional): 学習用データをシャッフルするかどうか
            valid_shuffle (bool, optional): 検証用データをシャッフルするかどうか
            test_shuffle (bool, optional): テスト用データをシャッフルするかどうか
        """
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_shuffle = train_shuffle
        self.valid_shuffle = valid_shuffle
        self.test_shuffle = test_shuffle

        self.train_set = train_set
        self.test_set = test_set

        self.splits = [split for split in KFold(k).split(range(len(self.train_set)))]

    def generate(self) -> Generator[tuple[int, "HoldOutDataModule"], None, None]:
        """
        K分割交差検証用のデータを生成します.

        Yields:
            (int, HoldOutDataModule): fold, データモジュール
        """
        for fold in range(self.k):
            train_indices, valid_indices = self.splits[fold]
            train_fold = Subset(self.train_set, train_indices)
            valid_fold = Subset(self.train_set, valid_indices)

            yield fold, HoldOutDataModule(
                train_set=train_fold,
                val_set=valid_fold,
                test_set=self.test_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                train_shuffle=self.train_shuffle,
                val_shuffle=self.valid_shuffle,
                test_shuffle=self.test_shuffle,
            )
