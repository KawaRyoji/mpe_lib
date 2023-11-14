import os
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Iterable, Optional, Type, final

import optuna
import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from typing_extensions import override

from .data_module import KFoldDataModuleGenerator


class Experiment:
    """
    Pytorch lightning によるDNN実験クラスです. 必要に応じて抽象メソッド・プロパティをオーバーライドしてください.
    """

    def __init__(
        self,
        root_dir: str,
        monitor: str,
        model_weight_dir_name: str = "model_weights",
        csv_dir_name: str = "history",
        tensor_board_dir_name: str = "tb_log",
    ) -> None:
        """
        Args:
            root_dir (str): 実験結果を保存するディレクトリパス
            monitor (str): 学習の際に監視するメトリクス.
            model_weight_dir_name (str): モデルの重みを保存するディレクトリ名
            csv_dir_name (str): 学習のログを保存するディレクトリ名
            tensor_board_dir_name (str): TensorBoardのログを保存するディレクトリ名
        """
        self.__root_dir = root_dir
        self.__monitor = monitor
        self.__model_weight_dir_name = model_weight_dir_name
        self.__csv_dir_name = csv_dir_name
        self.__tensor_board_dir_name = tensor_board_dir_name

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

    @property
    def model_weight_dir_name(self) -> str:
        """
        モデルの重みを保存するディレクトリ名
        """
        return self.__model_weight_dir_name

    @property
    def csv_dir_name(self) -> str:
        """
        学習のログを保存するディレクトリ名
        """
        return self.__csv_dir_name

    @property
    def tensor_board_dir_name(self) -> str:
        """
        TensorBoardのログを保存するディレクトリ名
        """
        return self.__tensor_board_dir_name

    @abstractmethod
    def model_params(self) -> dict[str, Any]:
        """
        モデルに渡すパラメータを定義します.
        定義したパラメータは`define_model`メソッドに使用されます.
        オーバーライドして使用してください.

        Returns:
            dict[str, Any]: 定義したパラメータ
        """
        raise NotImplementedError()

    @abstractmethod
    def define_model(self, model_params: dict[str, Any]) -> LightningModule:
        """
        モデルを定義し, 生成します.
        このメソッドに渡される引数は`model_params`メソッドで定義したパラメータになります.
        オーバーライドして使用してください.

        Args:
            model_params (dict[str, Any]): モデルのパラメータ

        Returns:
            LightningModule: 生成したモデル
        """
        raise NotImplementedError()

    def create_model(self) -> LightningModule:
        """
        定義したモデルとパラメータからモデルを定義します.

        Returns:
            LightningModule: 生成したモデル
        """
        return self.define_model(self.model_params())

    @final
    def default_trainer_args(self, root_dir: str, monitor: str) -> dict[str, Any]:
        """
        実験に必要となる学習器のデフォルトのパラメータを定義します.

        Args:
            root_dir (str): 実験を保存するルートディレクトリパス
            monitor (str): 監視するメトリクス

        Returns:
            dict[str, Any]: 学習器のデフォルトのパラメータ
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

    def trainer_args(self, root_dir: str, monitor: str) -> dict[str, Any]:
        """
        `default_trainer_args`で定義されているデフォルトのパラメータに追加する学習器のパラメータを定義します.
        オーバーライドして使用してください.

        Args:
            root_dir (str): 実験を保存するルートディレクトリパス
            monitor (str): 監視するメトリクス

        Returns:
            dict[str, Any]: 学習器のパラメータ
        """
        return {}

    def _merge_trainer_args(self, root_dir: str, monitor: str) -> dict[str, Any]:
        default = self.default_trainer_args(root_dir, monitor)
        defined = self.trainer_args(root_dir, monitor)

        maybe_callbacks = defined.pop("callbacks", None)
        if maybe_callbacks is not None:
            if isinstance(maybe_callbacks, (list, tuple, Iterable)):
                default.update(callbacks=default["callbacks"] + list(maybe_callbacks))
            else:
                default.update(callbacks=default["callbacks"] + [maybe_callbacks])

        maybe_logger = defined.pop("logger", None)
        if maybe_logger is not None:
            if isinstance(maybe_logger, (list, tuple, Iterable)):
                default.update(logger=default["logger"] + list(maybe_logger))
            else:
                default.update(logger=default["logger"] + [maybe_logger])

        default.update(**defined)
        return default

    def create_trainer(self, root_dir: str, monitor: str) -> Trainer:
        """
        定義したパラメータを使用して学習器を生成します.

        Args:
            root_dir (str): 実験を保存するルートディレクトリパス
            monitor (str): 監視するメトリクス

        Returns:
            Trainer: 生成した学習器
        """
        return Trainer(
            default_root_dir=root_dir, **self._merge_trainer_args(root_dir, monitor)
        )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()


class HoldOutExperiment(Experiment):
    @abstractmethod
    def data_module_args(self) -> dict[str, Any]:
        """
        データセットを作成するためのパラメータを定義します.
        オーバーライドして使用してください.

        Returns:
            dict[str, Any]: データセットのパラメータ
        """
        raise NotImplementedError()

    @abstractmethod
    def define_data_module(self, *args: Any, **kwargs: Any) -> LightningDataModule:
        """
        データセットを定義し, 生成します.
        このメソッドに渡される引数は`data_module_args`で定義したパラメータです.
        オーバーライドして使用してください.

        Args:
            data_module_args (dict[str, Any]): データセットのパラメータ. `data_module_args`で設定した値となります.

        Returns:
            LightningDataModule: 生成したデータセット
        """
        raise NotImplementedError()

    def create_data_module(self) -> LightningDataModule:
        """
        定義したデータセットとパラメータでデータセットを生成します.

        Returns:
            LightningDataModule: 生成したデータセット
        """
        return self.define_data_module(**self.data_module_args())

    def run(self) -> None:
        """
        Hold-Out法によりモデルを学習・テストします.
        """
        trainer = self.create_trainer(self.root_dir, self.monitor)
        model = self.create_model()
        data_module = self.create_data_module()

        trainer.fit(model, datamodule=data_module)
        trainer.test(
            model,
            datamodule=data_module,
            ckpt_path=os.path.join(
                self.root_dir, self.model_weight_dir_name, "best.ckpt"
            ),
        )


class KFoldExperiment(Experiment):
    @abstractmethod
    def k_fold_data_generator_args(self) -> dict[str, Any]:
        """
        k分割交差検証用のデータジェネレータに渡す引数を定義します.
        このメソッドは`k_fold_data_generator`に使用されます.
        オーバーライドして使用してください.

        Returns:
            dict[str, Any]: k分割交差検証用のデータジェネレータに渡す引数
        """
        raise NotImplementedError()

    @abstractmethod
    def define_k_fold_data_generator(
        self, *args: Any, **kwargs: Any
    ) -> KFoldDataModuleGenerator:
        """
        k分割交差検証用のデータジェネレータを定義します. このメソッドは`run_k_fold`に使用されます.
        オーバーライドして使用してください.

        Returns:
            KFoldDataModuleGenerator: k分割交差検証用のデータジェネレータ
        """
        raise NotImplementedError()

    def create_k_fold_data_generator(self) -> KFoldDataModuleGenerator:
        """
        定義したパラメータとデータジェネレータからデータジェネレータを生成します.

        Returns:
            KFoldDataModuleGenerator: 生成したデータジェネレータ
        """
        return self.define_k_fold_data_generator(**self.k_fold_data_generator_args())

    @override
    def run(self) -> None:
        """
        k分割交差検証を行います.
        """
        model = self.create_model()
        init_state = deepcopy(model.state_dict())
        histories: list[pd.DataFrame] = []

        generator = self.create_k_fold_data_generator()

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
                pd.read_csv(os.path.join(fold_dir, "version_0", "metrics.csv"))
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


class OptunaExperiment(Experiment):
    @abstractmethod
    def data_module_args(self) -> dict[str, Any]:
        """
        データセットを作成するためのパラメータを定義します.
        オーバーライドして使用してください.

        Returns:
            dict[str, Any]: データセットのパラメータ
        """
        raise NotImplementedError()

    @abstractmethod
    def define_data_module(
        self, data_module_args: dict[str, Any]
    ) -> LightningDataModule:
        """
        データセットを定義し, 生成します.
        このメソッドに渡される引数は`data_module_args`で定義したパラメータです.
        オーバーライドして使用してください.

        Args:
            data_module_args (dict[str, Any]): データセットのパラメータ. `data_module_args`で設定した値となります.

        Returns:
            LightningDataModule: 生成したデータセット
        """
        raise NotImplementedError()

    def create_data_module(self) -> LightningDataModule:
        """
        定義したデータセットとパラメータでデータセットを生成します.

        Returns:
            LightningDataModule: 生成したデータセット
        """
        return self.define_data_module(**self.data_module_args())

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
        optunaで用いる目標関数を生成します.

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

            trainer = self.create_trainer(trial_dir, monitor)
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

    @override
    def run(
        self,
        tune_monitor: Optional[str] = None,
        trials: int = 20,
        direction: str = "maximize",
        pruner: Optional[optuna.pruners.BasePruner] = optuna.pruners.MedianPruner(),
        study_params: dict[str, Any] = {},
        optimize_params: dict[str, Any] = {},
    ) -> None:
        """
        optunaによりハイパーパラメータチューニングを行います.
        パラメータの定義は`suggestion_parameter`メソッドで定義してください.

        Args:
            tune_monitor (Optional[str], optional): 最適化するメトリクス. `None`の場合, 学習の際に監視するメトリクスと同じになります
            trials (int, optional): 試行回数
            direction (str, optional): 最適化する方向
            pruner (Optional[optuna.pruners.BasePruner], optional): 枝刈りを行う基準
            study_params (dict[str, Any], optional): optunaのstudyに渡すパラメータ
            optimize_params (dict[str, Any], optional): optunaのoptimizeに渡すパラメータ
        """
        study = optuna.create_study(direction=direction, pruner=pruner, **study_params)

        model_type = type(self.create_model())
        data_module = self.create_data_module()

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
