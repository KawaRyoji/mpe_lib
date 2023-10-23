import os
from abc import abstractmethod, abstractproperty
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type

import optuna
import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from .data_module import KFoldDataModuleGenerator


class Experiment:
    """
    Pytorch lightning によるDNN実験クラスです. 必要に応じて抽象メソッド・プロパティをオーバーライドしてください.
    """

    def __init__(self, root_dir: str, monitor: str) -> None:
        """
        Args:
            root_dir (str): 実験結果を保存するディレクトリパス
            monitor (str): 学習の際に監視するメトリクス.
        """
        self.__root_dir = root_dir
        self.__trainer = Trainer(
            default_root_dir=self.root_dir, **self.trainer_args(self.root_dir, monitor)
        )
        self.__monitor = monitor

    @property
    def root_dir(self) -> str:
        """
        実験結果を保存するディレクトリパス
        """
        return self.__root_dir

    @property
    def monitor(self) -> str:
        """
        学習の際に監視するメトリクス
        """
        return self.__monitor

    @abstractproperty
    def model_weight_dir_name(self) -> str:
        """
        モデルの重みを保存するディレクトリ名
        """
        raise NotImplementedError()

    @abstractproperty
    def csv_dir_name(self) -> str:
        """
        学習のログを保存するディレクトリ名
        """
        raise NotImplementedError()

    @abstractproperty
    def tensor_board_dir_name(self) -> str:
        """
        TensorBoardのログを保存するディレクトリ名
        """
        raise NotImplementedError()

    def data_module_args(self) -> Dict[str, Any]:
        """
        Pytorch lightning の`LigtningDataModule`に渡す引数を返します.
        このメソッドは`load_data_module`に使用されます.
        必要に応じてオーバーライドしてください.

        Returns:
            Dict[str, Any]: DataModuleに渡す引数
        """
        return {}

    def load_data_module(self) -> LightningDataModule:
        """
        `data_module_args`を使って、定義された`LigtningDataModule`を生成します.

        Returns:
            LightningDataModule: _description_
        """
        return self.define_data_module(**self.data_module_args())

    @abstractmethod
    def define_data_module(self, *args: Any, **kwargs: Any) -> LightningDataModule:
        """
        Pytorch lightning の`LigtningDataModule`を定義します. このメソッドは`load_data_module`に使用され, このメソッドの引数は`data_module_args`で定義します.
        必要に応じてオーバーライドしてください.

        Returns:
            LightningDataModule: 定義した`LigtningDataModule`
        """
        raise NotImplementedError()

    def k_fold_data_generator_args(self) -> Dict[str, Any]:
        """
        k分割交差検証用のデータジェネレータに渡す引数を定義します.
        このメソッドは`k_fold_data_generator`に使用されます.
        必要に応じてオーバーライドしてください.

        Returns:
            Dict[str, Any]: k分割交差検証用のデータジェネレータに渡す引数
        """
        return {}

    @abstractmethod
    def k_fold_data_generator(
        self, *args: Any, **kwargs: Any
    ) -> KFoldDataModuleGenerator:
        """
        k分割交差検証用のデータジェネレータを定義します. このメソッドは`run_k_fold`に使用されます.
        必要に応じてオーバーライドしてください.

        Returns:
            KFoldDataModuleGenerator: k分割交差検証用のデータジェネレータ
        """
        raise NotImplementedError()

    def trainer_args(self, root_dir: str, monitor: str) -> Dict[str, Any]:
        """
        Pytorch lightning の`Trainer`に渡す引数を定義します.
        デフォルトでTensorBoardロガーとCSVLogger, ModelCheckpointを定義しています.
        オーバーライドをするときは以下のように行ってください.

        ```python
        @override
        def trainer_args(self, root_dir: str, monitor: str) -> Dict[str, Any]:
            args = super().trainer_args(root_dir, monitor)
            args.update(
                {
                    "max_epoch": 100,
                    ...
                }
            )
            return args
        ```

        Args:
            root_dir (str): 実験結果を保存するディレクトリパス
            monitor (str): 学習の際に監視するメトリクス

        Returns:
            Dict[str, Any]: `Trainer`に渡す引数
        """
        return {
            "logger": [
                TensorBoardLogger(root_dir, name=self.tensor_board_dir_name),
                CSVLogger(root_dir, name=self.csv_dir_name),
            ],
            "callbacks": [
                ModelCheckpoint(
                    dirpath=os.path.join(root_dir, self.model_weight_dir_name),
                    filename="best",
                    monitor=monitor,
                ),
                ModelCheckpoint(
                    dirpath=os.path.join(root_dir, self.model_weight_dir_name),
                    filename="last",
                ),
            ],
        }

    def run_hold_out(
        self, model: LightningModule, data_module: LightningDataModule
    ) -> None:
        """
        学習とテストを行います.

        Args:
            model (LightningModule): 学習するモデル
            data_module (LightningDataModule): 学習に使用する`LigtningDataModule`
        """
        self.__trainer.fit(model, datamodule=data_module)
        self.test(model, data_module=data_module)

    def run_k_fold(self, model: LightningModule, version: int = 0) -> None:
        """
        k分割交差検証を行います. なおkは`k_fold_data_generator_args`で定義してください.

        Args:
            model (LightningModule): 学習するモデル
            version (int, optional): 平均を算出するバージョン数. デフォルトは0です.
        """
        init_state = deepcopy(model.state_dict())
        histories: List[pd.DataFrame] = []

        generator = self.k_fold_data_generator(**self.k_fold_data_generator_args())

        for fold, fold_module in generator.generate():
            fold_dir = os.path.join(self.root_dir, "fold{}".format(fold))

            trainer = Trainer(
                default_root_dir=fold_dir, **self.trainer_args(fold_dir, self.monitor)
            )
            trainer.fit(model=model, datamodule=fold_module)
            trainer.test(
                model=model,
                datamodule=fold_module,
                ckpt_path=os.path.join(
                    fold_dir, self.model_weight_dir_name, "best.ckpt"
                ),
            )

            histories.append(
                pd.read_csv(os.path.join(fold_dir, f"version_{version}", "metrics.csv"))
            )
            model.load_state_dict(init_state)  # モデルの重みの初期化

        histories_concat = pd.concat(histories, axis=0)
        test_metrics = list(
            filter(lambda c: c.startswith("test_"), histories_concat.columns.values)
        )
        test_metrics_histories = histories_concat.loc[:, test_metrics].dropna()
        pd.DataFrame(test_metrics_histories.mean()).T.to_csv(
            os.path.join(self.root_dir, "average.csv")
        )

    @abstractmethod
    def suggestion_parameter(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        optunaでチューニングするパラメータを定義するメソッドです.
        チューニングするパラメータは`trial.suggest_*`で定義してください.

        Args:
            trial (optuna.Trial): optunaのトライアル

        Raises:
            NotImplementedError: このメソッドがオーバーライドされてない場合

        Returns:
            dict[str, Any]: 定義したパラメータ
        """
        raise NotImplementedError()

    def create_objective_func(
        self,
        model_type: Type[LightningModule],
        data_module: LightningDataModule,
        tune_monitor: Optional[str] = None,
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        optunaで用いる目標関数を生成します. このメソッドは`run_optuna`で使用します.

        Args:
            model_type (Type[LightningModule]): モデルのタイプ
            data_module (LightningDataModule): 学習に使用する`LigtningDataModule`
            tune_monitor (Optional[str], optional): 最適化するメトリクス. `None`の場合, 学習の際に監視するメトリクスと同じになります

        Returns:
            Callable[[optuna.trial.Trial], float]: 生成した目標関数
        """

        def objective(trial: optuna.trial.Trial) -> float:
            trial_dir = os.path.join(self.root_dir, "optuna", f"trial{trial.number}")
            model = model_type(**self.suggestion_parameter(trial))

            trainer = Trainer(
                default_root_dir=trial_dir, **self.trainer_args(trial_dir, self.monitor)
            )
            trainer.fit(model, data_module)
            trainer.test(
                model=model,
                datamodule=data_module,
                ckpt_path=os.path.join(
                    trial_dir, self.model_weight_dir_name, "best.ckpt"
                ),
            )

            monitor = self.monitor if tune_monitor is None else tune_monitor

            return trainer.callback_metrics[monitor].item()

        return objective

    def run_optuna(
        self,
        model_type: Type[LightningModule],
        data_module: LightningDataModule,
        tune_monitor: Optional[str] = None,
        trials: int = 20,
        direction: str = "maximize",
        pruner: Optional[optuna.pruners.BasePruner] = optuna.pruners.MedianPruner(),
        study_params: Dict[str, Any] = {},
        optimize_params: Dict[str, Any] = {},
    ) -> None:
        """
        optunaによりハイパーパラメータチューニングを行います.

        Args:
            model_type (Type[LightningModule]): モデルのタイプ
            data_module (LightningDataModule): 学習に使用する`LigtningDataModule`
            tune_monitor (Optional[str], optional): 最適化するメトリクス. `None`の場合, 学習の際に監視するメトリクスと同じになります
            trials (int, optional): 試行回数
            direction (str, optional): 最適化する方向
            pruner (Optional[optuna.pruners.BasePruner], optional): 枝刈りを行う基準
            study_params (Dict[str, Any], optional): optunaのstudyに渡すパラメータ
            optimize_params (Dict[str, Any], optional): optunaのoptimizeに渡すパラメータ
        """
        study = optuna.create_study(direction=direction, pruner=pruner, **study_params)

        objective = self.create_objective_func(
            model_type=model_type,
            data_module=data_module,
            tune_monitor=tune_monitor,
        )

        study.optimize(objective, n_trials=trials, **optimize_params)
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial
        print("  trial: {}".format(trial.number))
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def test(self, model: LightningModule, data_module: LightningDataModule) -> None:
        """
        モデルのテストを行います.

        Args:
            model (LightningModule): テストを行うモデル
            data_module (LightningDataModule): テストに使用する`LigtningDataModule`
        """
        self.__trainer.test(
            model,
            datamodule=data_module,
            ckpt_path=os.path.join(
                self.root_dir, self.model_weight_dir_name, "best.ckpt"
            ),
        )
