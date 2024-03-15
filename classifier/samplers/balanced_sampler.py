from __future__ import annotations

import copy
import random

from . import Sampler
from classifier.dataset import DatasetEntry


class BalancedSampler(Sampler):
    name = "balanced"
    __max_length = 5
    __predict = None
    __seed = None
    __classes = None

    def configure(self, config: dict):

        self.__max_length = config.get("n_for_train", self.__max_length)
        self.__predict = config.get("class", self.__predict)
        self.__seed = config.get("seed", self.__seed)
        self.__classes = config.get("classes", self.__classes)

    def sample(
        self,
        train: list[DatasetEntry],
        test: list[DatasetEntry],
    ) -> list[DatasetEntry]:

        items = []
        train = copy.deepcopy(train)
        random.seed(self.__seed)
        random.shuffle(train)
        
        for _ in range(int(self.__max_length / len(self.__classes))):
            items_set = []
            classes_set = []
            for i, entry in enumerate(train):
                if entry.classes not in classes_set:
                    items_set.append(entry)
                    classes_set.append(entry.classes)
                    train.pop(i)
                if len(items_set) >= len(self.__classes):
                    break
            items.extend(items_set)

        return items
