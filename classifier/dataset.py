from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from io import TextIOWrapper
from pathlib import Path
from typing import BinaryIO
from typing import TypeGuard

import pandas as pd
from sklearn.model_selection import train_test_split

from classifier.configuration import Configuration


@dataclass
class DatasetEntry:
    input_text: str
    output_text: str

    @property
    def features(self) -> list[str]:
        return self.input_text.split("; ")

    @property
    def classes(self) -> list[str]:
        return self.output_text.split(", ")

    @property
    def tuple(self):
        return self.input_text, self.output_text


def only_strings(item: str | None) -> TypeGuard[str]:
    return isinstance(item, str)


class ClassificationDataset(list[DatasetEntry]):
    @staticmethod
    def _get_io(data: str | Path | TextIOWrapper):
        if isinstance(data, str):
            return StringIO(data)
        if isinstance(data, Path):
            return open(data)
        if isinstance(data, TextIOWrapper):
            return data
        raise TypeError("Invalid data passed.")

    @classmethod
    def from_csv(
        cls,
        into_io: str | Path | TextIOWrapper,
        config: Configuration,
        delimiter: str = ",",
    ) -> ClassificationDataset:

        data: pd.DataFrame = pd.read_csv(cls._get_io(into_io), delimiter=delimiter)
        return cls._process_entries(data, config)

    @classmethod
    def from_xlsx(
        cls,
        path: Path | BinaryIO,
        config: Configuration,
    ) -> ClassificationDataset:

        if not isinstance(path, Path):
            raise TypeError("Invalid data passed.")
        data: pd.DataFrame = pd.read_excel(path)
        return cls._process_entries(data, config)

    @classmethod
    def load_path(
        cls,
        path: Path,
        config: dict | Configuration,
    ) -> ClassificationDataset:

        if isinstance(config, dict) and not isinstance(config, Configuration):
            config = Configuration(existing_data=config)
        if not isinstance(path, Path):
            raise TypeError(f"Invalid path: {path}.")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}.")
        if path.suffix == ".csv":
            return cls.from_csv(path, config)
        elif path.suffix == ".xlsx":
            return cls.from_xlsx(path, config)
        else:
            raise ValueError(f"Invalid file format: {path.suffix}")

    @classmethod
    def _process_entries(
        cls,
        _data: pd.DataFrame,
        config: Configuration,
    ) -> ClassificationDataset:

        classes_labels: list[str] = config.classes
        (texts, classes) = _data.drop(classes_labels, axis=1), _data[classes_labels]
        labels: list[str] = list(_data.drop(classes_labels, axis=1).columns)
        data: list[tuple[list[str], list[str]]] = list(
            zip(
                map(lambda x: list(x[1:]), texts.itertuples()),
                map(lambda x: list(x[1:]), classes.itertuples()),
            )
        )
        #print(data)
        items = list(
            map(
                lambda entry: DatasetEntry(
                    input_text=" ".join(entry[0])
                    if config.pure_text
                    else (
                        "; ".join(
                            map(lambda x: f"{x[0]}: {x[1]}", zip(labels, entry[0]))
                        )
                    ),
                    output_text=", ".join(
                        list(
                            filter(
                                only_strings,
                                map(
                                    lambda arg: arg[1] if arg[0] else None,
                                    zip(entry[1], classes_labels),
                                ),
                            )
                        )
                    ),
                ),
                data,
            )
        )
        return cls(items)

    def train_test_split(
        self,
        test_size,
        seed,
    ) -> tuple[ClassificationDataset, ClassificationDataset]:

        train, test = train_test_split(self, test_size=test_size, random_state=seed)
        return ClassificationDataset(train), ClassificationDataset(test)

    def tuples(self):
        return list(map(lambda x: x.tuple, self))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ClassificationDataset(list.__getitem__(self, idx))
        else:
            return list.__getitem__(self, idx)
        
        
class RegressionDataset(list[DatasetEntry]):
    @staticmethod
    def _get_io(data: str | Path | TextIOWrapper):
        if isinstance(data, str):
            return StringIO(data)
        if isinstance(data, Path):
            return open(data)
        if isinstance(data, TextIOWrapper):
            return data
        raise TypeError("Invalid data passed.")

    @classmethod
    def from_csv(
        cls,
        into_io: str | Path | TextIOWrapper,
        config: Configuration,
        delimiter: str = ",",
    ) -> RegressionDataset:

        data: pd.DataFrame = pd.read_csv(cls._get_io(into_io), delimiter=delimiter)
        return cls._process_entries(data, config)

    @classmethod
    def from_xlsx(
        cls,
        path: Path | BinaryIO,
        config: Configuration,
    ) -> RegressionDataset:

        if not isinstance(path, Path):
            raise TypeError("Invalid data passed.")
        data: pd.DataFrame = pd.read_excel(path)
        return cls._process_entries(data, config)

    @classmethod
    def load_path(
        cls,
        path: Path,
        config: dict | Configuration,
    ) -> RegressionDataset:

        if isinstance(config, dict) and not isinstance(config, Configuration):
            config = Configuration(existing_data=config)
        if not isinstance(path, Path):
            raise TypeError(f"Invalid path: {path}.")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}.")
        if path.suffix == ".csv":
            return cls.from_csv(path, config)
        elif path.suffix == ".xlsx":
            return cls.from_xlsx(path, config)
        else:
            raise ValueError(f"Invalid file format: {path.suffix}")

    @classmethod
    def _process_entries(
        cls,
        _data: pd.DataFrame,
        config: Configuration,
    ) -> RegressionDataset:

        target_column: list[str] = config.target_column
        (texts, targets) = _data.drop(target_column, axis=1), pd.DataFrame(_data[target_column])
        labels: list[str] = list(_data.drop(target_column, axis=1).columns)
        data: list[tuple[list[str], list[str]]] = list(
            zip(
                map(lambda x: list(x[1:]), texts.itertuples()),
                map(lambda x: list(x[1:]), targets.itertuples()),
            )
        )
        #print(data)
        items = list(
            map(
                lambda entry: DatasetEntry(
                    input_text=" ".join(entry[0])
                    if config.pure_text
                    else (
                        "; ".join(
                            map(lambda x: f"{x[0]}: {x[1]}", zip(labels, entry[0]))
                        )
                    ),
                    output_text=str(entry[1][0])
                    # output_text=", ".join(
                    #     list(
                    #         filter(
                    #             only_strings,
                    #             map(
                    #                 lambda arg: arg[1] if arg[0] else None,
                    #                 zip(entry[1], target_column),
                    #             ),
                    #         )
                    #     )
                    # ),
                ),
                data,
            )
        )
        return cls(items)

    def train_test_split(
        self,
        test_size,
        seed,
    ) -> tuple[RegressionDataset, RegressionDataset]:

        train, test = train_test_split(self, test_size=test_size, random_state=seed)
        return RegressionDataset(train), RegressionDataset(test)

    def tuples(self):
        return list(map(lambda x: x.tuple, self))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return RegressionDataset(list.__getitem__(self, idx))
        else:
            return list.__getitem__(self, idx)
