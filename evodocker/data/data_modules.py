import copy
import itertools
import time
import traceback
from collections import Counter
from functools import partial
import json
import os
import pickle
from typing import Optional, Sequence, Any

import ml_collections as mlc
import lightning as L
import torch
from torch.utils.data import RandomSampler

from evodocker.data.data_pipeline import parse_input_json
from evodocker.data import data_pipeline
from evodocker.utils.tensor_utils import dict_multimap
from evodocker.utils.tensor_utils import (
    tensor_tree_map,
)


class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str,
                 config: mlc.ConfigDict,
                 mode: str = "train",
                 ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                config:
                    A dataset config object. See openfold.config
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir

        self.config = config
        self.mode = mode

        valid_modes = ["train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        self._all_input_files = [i for i in os.listdir(data_dir) if i.endswith(".json")]
        if self.config.data_module.data_loaders.should_verify:
            self._all_input_files = [i for i in self._all_input_files if self._verify_json_input_file(i)]

        self.data_pipeline = data_pipeline.DataPipeline(config, mode)

    def _verify_json_input_file(self, file_name: str) -> bool:
        with open(os.path.join(self.data_dir, file_name), "r") as f:
            try:
                loaded = json.load(f)
                for i in ["input_structure"]:
                    if i not in loaded:
                        return False
                if self.mode != "predict":
                    for i in ["gt_structure", "resolution"]:
                        if i not in loaded:
                            return False
            except json.JSONDecodeError:
                return False
        return True

    def get_metadata_for_idx(self, idx: int) -> dict:
        input_path = os.path.join(self.data_dir, self._all_input_files[idx])
        input_data = json.load(open(input_path, "r"))
        metadata = {
            "resolution": input_data.get("resolution", 99.0),
            "input_path": input_path,
            "input_name": os.path.basename(input_path).split(".json")[0],
        }
        return metadata

    def __getitem__(self, idx):
        return parse_input_json(
            input_path=os.path.join(self.data_dir, self._all_input_files[idx]),
            mode=self.mode,
            config=self.config,
            data_pipeline=self.data_pipeline,
            data_dir=os.path.dirname(self.data_dir),
            idx=idx,
        )

    def __len__(self):
        return len(self._all_input_files)


def resolution_filter(resolution: int, max_resolution: float) -> bool:
    """Check that the resolution is <= max_resolution permitted"""
    return resolution is not None and resolution <= max_resolution


def all_seq_len_filter(seqs: list, minimum_number_of_residues: int) -> bool:
    """Check if the total combined sequence lengths are >= minimum_numer_of_residues"""
    total_len = sum([len(i) for i in seqs])
    return total_len >= minimum_number_of_residues


class OpenFoldDataset(torch.utils.data.Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """

    def __init__(self,
                 datasets: Sequence[OpenFoldSingleDataset],
                 probabilities: Sequence[float],
                 epoch_len: int,
                 generator: torch.Generator = None,
                 _roll_at_init: bool = True,
                 ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator

        self._samples = [self.looped_samples(i) for i in range(len(self.datasets))]
        if _roll_at_init:
            self.reroll()

    @staticmethod
    def deterministic_train_filter(
        cache_entry: Any,
        max_resolution: float = 9.,
        max_single_aa_prop: float = 0.8,
        *args, **kwargs
    ) -> bool:
        # Hard filters
        resolution = cache_entry["resolution"]

        return all([
            resolution_filter(resolution=resolution,
                              max_resolution=max_resolution)
        ])

    @staticmethod
    def get_stochastic_train_filter_prob(
        cache_entry: Any,
        *args, **kwargs
    ) -> float:
        # Stochastic filters
        probabilities = []

        cluster_size = cache_entry.get("cluster_size", None)
        if cluster_size is not None and cluster_size > 0:
            probabilities.append(1 / cluster_size)

        # Risk of underflow here?
        out = 1
        for p in probabilities:
            out *= p

        return out

    def looped_shuffled_dataset_idx(self, dataset_len):
        while True:
            # Uniformly shuffle each dataset's indices
            weights = [1. for _ in range(dataset_len)]
            shuf = torch.multinomial(
                torch.tensor(weights),
                num_samples=dataset_len,
                replacement=False,
                generator=self.generator,
            )
            for idx in shuf:
                yield idx

    def looped_samples(self, dataset_idx):
        max_cache_len = int(self.epoch_len * self.probabilities[dataset_idx])
        dataset = self.datasets[dataset_idx]
        idx_iter = self.looped_shuffled_dataset_idx(len(dataset))
        while True:
            weights = []
            idx = []
            for _ in range(max_cache_len):
                candidate_idx = next(idx_iter)
                # chain_id = dataset.idx_to_chain_id(candidate_idx)
                # chain_data_cache_entry = chain_data_cache[chain_id]
                # data_entry = dataset[candidate_idx.item()]
                entry_metadata_for_filter = dataset.get_metadata_for_idx(candidate_idx.item())
                if not self.deterministic_train_filter(entry_metadata_for_filter):
                    continue

                p = self.get_stochastic_train_filter_prob(
                    entry_metadata_for_filter,
                )
                weights.append([1. - p, p])
                idx.append(candidate_idx)

            samples = torch.multinomial(
                torch.tensor(weights),
                num_samples=1,
                generator=self.generator,
            )
            samples = samples.squeeze()

            cache = [i for i, s in zip(idx, samples) if s]

            for datapoint_idx in cache:
                yield datapoint_idx

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        # TODO bshor: I have removed support for filters (currently done in preprocess) and to weighting clusters
        # now it is much faster, because it doesn't call looped_samples
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )
        self.datapoints = []
        counter_datasets = Counter(dataset_choices.tolist())
        for dataset_idx, num_samples in counter_datasets.items():
            dataset = self.datasets[dataset_idx]
            sample_choices = torch.randint(0, len(dataset), (num_samples,), generator=self.generator)
            for datapoint_idx in sample_choices:
                self.datapoints.append((dataset_idx, datapoint_idx))


class OpenFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)


class OpenFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters

        if stage_cfg.uniform_recycling:
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.

        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs]

        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        # gt_features = batch.pop('gt_features', None)
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1,  # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample,
                device=aatype.device,
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if key == "no_recycling_iters":
                no_recycling = sample

        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)
        # batch['gt_features'] = gt_features

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class OpenFoldDataModule(L.LightningDataModule):
    def __init__(self,
                 config: mlc.ConfigDict,
                 train_data_dir: Optional[str] = None,
                 val_data_dir: Optional[str] = None,
                 predict_data_dir: Optional[str] = None,
                 batch_seed: Optional[int] = None,
                 train_epoch_len: int = 50000,
                 **kwargs
                 ):
        super(OpenFoldDataModule, self).__init__()

        self.config = config
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.predict_data_dir = predict_data_dir
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        if self.train_data_dir is None and self.predict_data_dir is None:
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_data_dir is not None

        # if not self.training_mode and predict_alignment_dir is None:
        #     raise ValueError(
        #         'In inference mode, predict_alignment_dir must be specified'
        #     )
        # elif val_data_dir is not None and val_alignment_dir is None:
        #     raise ValueError(
        #         'If val_data_dir is specified, val_alignment_dir must '
        #         'be specified as well'
        #     )

    def setup(self, stage):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(OpenFoldSingleDataset,
                              config=self.config)

        if self.training_mode:
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                mode="train",
            )

            datasets = [train_dataset]
            probabilities = [1.]

            generator = None
            if self.batch_seed is not None:
                generator = torch.Generator()
                generator = generator.manual_seed(self.batch_seed + 1)

            self.train_dataset = OpenFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=self.train_epoch_len,
                generator=generator,
                _roll_at_init=False,
            )

            if self.val_data_dir is not None:
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    mode="eval",
                )
            else:
                self.eval_dataset = None
        else:
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                mode="predict",
            )

    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        if stage == "train":
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenFoldBatchCollator()

        dl = OpenFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            # num_workers=self.config.data_module.data_loaders.num_workers,
            num_workers=0, # TODO bshor: solve generator pickling issue and then bring back num_workers, or just remove generator
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(L.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)


class DockFormerSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, clusters_json: str, config: mlc.ConfigDict, mode: str = "train"):
        clusters = json.load(open(clusters_json, "r"))
        self.config = config
        self.mode = mode
        self._data_dir = os.path.dirname(clusters_json)
        print("Data dir", self._data_dir)
        self._clusters = clusters
        self._all_input_files = sum(clusters.values(), [])
        self.data_pipeline = data_pipeline.DataPipeline(config, mode)

    def __getitem__(self, idx):
        return parse_input_json(
            input_path=os.path.join(self._data_dir, self._all_input_files[idx]),
            mode=self.mode,
            config=self.config,
            data_pipeline=self.data_pipeline,
            data_dir=self._data_dir,
            idx=idx,
        )

    def __len__(self):
        return len(self._all_input_files)


class DockFormerClusteredDataset(torch.utils.data.Dataset):
    def __init__(self, clusters_json: str, config: mlc.ConfigDict, mode: str = "train", generator=None):
        clusters = json.load(open(clusters_json, "r"))
        self.config = config
        self.mode = mode
        self._data_dir = os.path.dirname(clusters_json)
        self._clusters = list(clusters.values())
        self.data_pipeline = data_pipeline.DataPipeline(config, mode)
        self._generator = generator

    def __getitem__(self, idx):
        try:
            cluster = self._clusters[idx]
            # choose random from cluster
            input_file = cluster[torch.randint(0, len(cluster), (1,), generator=self._generator).item()]

            return parse_input_json(
                input_path=os.path.join(self._data_dir, input_file),
                mode=self.mode,
                config=self.config,
                data_pipeline=self.data_pipeline,
                data_dir=self._data_dir,
                idx=idx,
            )
        except Exception as e:
            print("ERROR in loading", e)
            traceback.print_exc()
            return parse_input_json(
                input_path=os.path.join(self._data_dir, self._clusters[0][0]),
                mode=self.mode,
                config=self.config,
                data_pipeline=self.data_pipeline,
                data_dir=self._data_dir,
                idx=idx,
            )


    def __len__(self):
        return len(self._clusters)


class DockFormerDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        # self.generator = generator

    def _add_batch_properties(self, batch):
        if self.config[self.stage].uniform_recycling:
            aatype = batch["aatype"]
            max_recycling_dim = aatype.shape[-1]

            # num_recycles = torch.randint(0, max_recycling_dim, (1,), generator=self.generator)
            num_recycles = torch.randint(0, max_recycling_dim, (1,)).item()

            resample_recycling = lambda t: t[..., :num_recycles + 1]
            batch = tensor_tree_map(resample_recycling, batch)

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class DockFormerDataModule(L.LightningDataModule):
    def __init__(self,
                 config: mlc.ConfigDict,
                 train_data_file: Optional[str] = None,
                 val_data_file: Optional[str] = None,
                 batch_seed: Optional[int] = None,
                 **kwargs
                 ):
        super(DockFormerDataModule, self).__init__()

        self.config = config
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.batch_seed = batch_seed

        assert self.train_data_file is not None, "train_data_file must be specified"
        assert self.val_data_file is not None, "val_data_file must be specified"

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed + 1)

        self.train_dataset = DockFormerClusteredDataset(
            clusters_json=self.train_data_file,
            config=self.config,
            mode="train",
            generator=generator,
        )

        self.val_dataset = DockFormerSimpleDataset(
            clusters_json=self.val_data_file,
            config=self.config,
            mode="eval",
        )

    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        should_shuffle = stage == "train"
        if stage == "train":
            dataset = self.train_dataset
        elif stage == "eval":
            dataset = self.val_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenFoldBatchCollator()

        dl = DockFormerDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            # generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            # num_workers=self.config.data_module.data_loaders.num_workers,
            num_workers=0, # TODO bshor: solve generator pickling issue and then bring back num_workers, or just remove generator
            collate_fn=batch_collator,
            shuffle=should_shuffle,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        if self.val_dataset is not None:
            return self._gen_dataloader("eval")
        return None
