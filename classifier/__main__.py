from __future__ import annotations

import json
import random
from pathlib import Path
from sys import argv

import requests.exceptions
import tqdm
from dotenv import load_dotenv

from .configuration import Configuration
from .configuration import ConfigurationError
from .dataset import ClassificationDataset, RegressionDataset
from .logger import logger
from .providers import CompletionProvider
from .providers import CompletionRequest
from .providers.mistral_provider import MistralCompletionProvider
from .providers.openai_provider import OpenAICompletionProvider
from .utils import create_result_directory
from classifier.samplers.occurrence_sampler import OccurrenceSampler
from classifier.samplers.strict_sampler import StrictSampler
from classifier.samplers.tanimoto_sampler import TanimotoSampler
from classifier.samplers.balanced_sampler import BalancedSampler
from classifier.samplers.random_sampler import RandomSampler


load_dotenv()

config_path = argv[1]
config = Configuration.load(Path(config_path))
random.seed(int(config.seed))

if config.task == "classification":
    dataset = ClassificationDataset.load_path(config.dataset, config)
    splits: tuple[ClassificationDataset, ClassificationDataset] = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    train_dataset, test_dataset = splits
else:
    dataset = RegressionDataset.load_path(config.dataset, config)
    splits: tuple[RegressionDataset, RegressionDataset] = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    train_dataset, test_dataset = splits

results: list[dict] = []
path = create_result_directory(config=config)
total_cost: float = 0

sampler = None
for smp in [
    StrictSampler,
    OccurrenceSampler,
    TanimotoSampler,
    BalancedSampler,
    RandomSampler
]:
    if smp.name == config.get("sampler"):
        sampler = smp()
        sampler.configure(config)

if sampler is None:
    logger.error("No sampler specified.")
    raise ConfigurationError("sampler")

for dry_run in [True, False]:
    for i, item in tqdm.tqdm(enumerate(test_dataset)):
        sampler.configure({"seed": config.get("seed")+i})
        sampler.configure({"request": item.features[0].lstrip('reaction: ')})
        sampler.configure({"class": item.classes[0]})
        samples = sampler.sample(train_dataset + test_dataset, test_dataset)
        # print(item, '\n', samples)
        # break
        provider: CompletionProvider | None = None
        if config.provider == "openai":
            provider = OpenAICompletionProvider()
        elif config.provider == "mistral":
            provider = MistralCompletionProvider()

        if provider is None:
            logger.error("No provider specified.")
            raise ConfigurationError("provider")

        provider.configure(config)

        try:
            if not dry_run:
                logger.debug(f'Trying to get completion for "{item.input_text}"')
            completion = provider.get_completion(
                CompletionRequest(
                    samples=samples, question=item.input_text, engine=config.engine
                ),
                dry_run=dry_run,
            )
            if not dry_run:
                logger.debug(
                    f"Successfully retrieved completion. Classes: {completion.classes}"
                )
            logger.debug(f"Approximate cost of request is {completion.cost:.02f}$")
            if not dry_run:
                results.append(
                    dict(
                        input=item.input_text,
                        target_classes=item.classes,
                        predicted_classes=completion.classes,
                    )
                )
            if completion.cost:
                total_cost += completion.cost
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during completion generation: {e}")
            logger.error(f"Item content: {item.input_text}")
            continue
    if dry_run:
        print(f"Total cost will be approximately {total_cost:.02f}$")
        if input("Do you want to continue? (y/n) ").lower() != "y":
            exit(0)
        total_cost = 0

with open(path / "results.json", "w") as file:
    json.dump(
        list(results),
        file,
    )

logger.info(f"Total cost: {total_cost:.02f}$")
