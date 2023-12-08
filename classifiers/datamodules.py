import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from typing import Tuple, Optional, Callable, List, Union, Iterator, Any
import zarr
import itertools

import sys
from .transforms import ZarrRandomCrop


class DistributedSampler(Sampler):
    """
    Taken from photosynthetic.training.data.samplers.DistributedSampler:

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. important::
        There's a significant difference between this class and `torch.utils.data.DistributedSampler.
        Here, `drop_last=False` doesn't pad the data, just slices the data as given.
        As a consequence, different GPU workers may get different number of batches which may lead to deadlocks
        if not used properly. As a general guide, you should always use `drop_last=True` for training, and ensure
        that there are no distributed operation when `drop_last=False`.

    Args:
        sampler: Sampler used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler do NOT ensure that the data is
            evenly divisible across the replicas and all replicas get some data.
        batch_size (int): per-slot batch size, will be used by lightning
            in order to estimate the number of training steps
    """

    def __init__(
        self,
        sampler: Union[Sampler, BatchSampler],
        drop_last: bool,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Any]:
        if not self.drop_last:
            yield from itertools.islice(self.sampler, self.rank, None, self.num_replicas)
        else:
            for i, index in enumerate(self.sampler):
                if i % self.num_replicas == self.rank:
                    element = index

                if i % self.num_replicas == self.num_replicas - 1:
                    # ensure that all ranks yield the same number of times
                    yield element

    def __len__(self) -> int:
        length = len(self.sampler) // self.num_replicas  # type: ignore[arg-type]
        if not self.drop_last:
            length += self.rank < (len(self.sampler) % self.num_replicas)  # type: ignore[arg-type]
        return length


def load_zarr(path: str) -> zarr.core.Array:
    z = zarr.open_array(path, mode="r")
    return z

class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        pixels /= 255.0
        return pixels

class DomainSampler(BatchSampler):
    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int,
        random_seed: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.sub_batch_size = self.batch_size
        self.metadata = metadata
        self._rng = np.random.RandomState(random_seed)
        # grouping by domain
        self._groups = self.metadata.groupby("batch")
        self._len = 0
        sizes = self._groups.size() // self.sub_batch_size
        remainders = self._groups.size() % self.sub_batch_size
        self._len = sizes.values.sum() + (remainders > 1).sum()


    def __len__(self) -> int:
        return self._len

    ## TODO: ENSURE AT LEAST 2 SAMPLES PER BATCH
        
    def __iter__(self):
        # copy and shuffle (.sample) the groups into a new list
        # we only want the indices of the rows, do not need the whole dataframe
        groups = [group.sample(frac=1, random_state=self._rng).index for _, group in self._groups]
        self._rng.shuffle(groups)
        # prepare the batches by slicing samples out from each group
        sub_batches = []
        group_sizes = []
        for group in groups:
            group_size_ = len(group) // self.sub_batch_size
            remainder = len(group) % self.sub_batch_size
            for i in range(group_size_):
                sub_batches.append(group[i * self.sub_batch_size : (i + 1) * self.sub_batch_size])
            if remainder > 1:
                sub_batches.append(group[-remainder:])
                group_size_ += 1
            group_sizes.append(group_size_)
        # shuffle them so that groups are distributed randomly over the epoch, then yield batches
        self._rng.shuffle(sub_batches)
        for (i, group_size) in enumerate(group_sizes):
            if i == 0:
                start = 0
            else:
                start = sum(group_sizes[:i])
                assert start == end, f"starting group index {start} was not equal to ending group index {end}, which will result in missing at least one batch"
            end = start + group_size
            sub_batches_list = sub_batches[start:end]
            for j in range(group_size):
                yield np.array(
                    sub_batches_list[j]  #https://note.nkmk.me/en/python-list-flatten/#:~:text=be%20discussed%20later.-,Flatten%20list%20with%20sum(),the%20%2B%20operation%20concatenates%20the%20lists.
                ).reshape(-1)
        
class BaseDataModule(LightningDataModule):
    def __init__(self,
        data_path: str,
        batch_size: int = 256
        ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path

        ## subclasses should define self.Dataset a class and self.datasetkwargs a dictionary of kwargs for the class

    def _train_val_split_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        np.random.seed(0)
        df = df.reset_index().sample(frac = 1)
        random = np.random.rand(len(df))

        train_df = df[random < 0.8].reset_index()
        val_df = df[(random > 0.8) & (random < 0.9)].reset_index()
        test_df = df[random > 0.9].reset_index()
        return train_df, val_df, test_df

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = pd.read_parquet(self.data_path) 
        return self._train_val_split_df(self.data)

    def setup(self, stage: Optional[str] = None) -> None:
        # see https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#setup
        # this hook is called on every process when using DDP at the beginning of fit
        (train_df, val_df, test_df) = self.load_data()
        # unpack the prepared metadata
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        self.labels = np.concatenate((self.train_data["guide_id"].values, self.val_data["guide_id"].values, self.test_data["guide_id"].values))

        if stage == "fit":
            self.distributed = self.trainer.world_size > 1
    
    def train_dataloader(self) -> DataLoader:
        train_dataset = self.Dataset(
            self.train_data,
            **self.datasetkwargs
            )
        if self.distributed:
            return DataLoader(train_dataset, batch_sampler = 
                            DistributedSampler(
                                sampler = DomainSampler(self.train_data, batch_size = self.batch_size),
                                drop_last = True,
                                batch_size = self.batch_size
                                ), 
                                num_workers = 8)
        else:
            return DataLoader(train_dataset, batch_sampler = DomainSampler(self.train_data, batch_size = self.batch_size))

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.Dataset(
            self.val_data,
            **self.datasetkwargs
            )
        if self.distributed:
            return DataLoader(val_dataset, batch_sampler = 
                            DistributedSampler(
                                sampler = DomainSampler(self.val_data, batch_size = self.batch_size),
                                drop_last = True,
                                batch_size = self.batch_size
                                ), 
                                num_workers = 8)
        else:
            return DataLoader(val_dataset, batch_sampler = DomainSampler(self.val_data, batch_size = self.batch_size))


    def test_dataloader(self) -> DataLoader:
        test_dataset = self.Dataset(
            self.test_data,
            **self.datasetkwargs
            )
        return DataLoader(test_dataset, batch_sampler = DomainSampler(self.test_data, batch_size = self.batch_size), num_workers = 8)

class ProteomicsDataset(Dataset):
    def __init__(self, metadata):
         super().__init__()
         self.df = metadata

    def __len__(self):
         return len(self.df)
    
    def process_row(self, row):
        label = torch.tensor(row["guide_id"]).long()
        dat = row.filter(regex="^[0-9]").astype("float32")  ## the data columns starts with numeric and metadata is non-numeric
        dat_tensor = torch.from_numpy(dat.values).float() 

        return dat_tensor, label

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        return self.process_row(row)
     

class ZarrDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        zarr_transform: Callable[[zarr.core.Array], torch.Tensor] = ZarrRandomCrop(size = 512),
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, ## augmenter
        keep_cols: List[str] = ["batch", "guide_id"],
        channels_to_keep: List[int] = [0, 1, 2, 3, 4, 5],
    ) -> None:
        """
        Parameters
        ----------
        metadata : pd.DataFrame
            Must contain a string column named "path" that defines the path to a zarr array.
        zarr_transform : Callable[[zarr.core.Array],torch.Tensor]
            A function for loading chuck from zarr.
            Current options in photosynthetic.training.transforms are
                ZarrRandomCrop, ZarrCenterCrop, ZarrLoadEntireArray
        transform : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
            A transformation to apply to the loaded tensor, by default None (no transform)
        """
        super().__init__()
        self.metadata = metadata
        self.zarr_transform = zarr_transform
        self.transform = transform
        self.keep_cols = keep_cols
        self.channels_to_keep = torch.tensor(channels_to_keep)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        """
        Loads an entire zarr as a channels-last Tensor, along with its metadata.
        If `self.transform` is not None, applies the transformation.

        Parameters
        ----------
        index : int
            An index into `self.metadata` from which to load a zarr.

        Returns
        -------
        Dict[str, torch.Tensor]
            The loaded zarr tensor is available under the "pixels" key. Other keys
            come from the corresponding row in `self.metadata`.
        """
        row = self.metadata.iloc[index]
        arr = load_zarr(row["path"])
        # prepare the pixels tensor
        with torch.inference_mode():
            tensor = self.zarr_transform(arr)
            if self.transform is not None:
                tensor = self.transform(tensor)
            else:
                tensor = tensor.index_select(0, index=self.channels_to_keep)
        # build the return dictionary for the batch
        d = {"pixels": tensor}

        for col in self.keep_cols:
            d[col] = row[col]
        return d["pixels"], torch.tensor(d["guide_id"]).long()


class ImageDataModule(BaseDataModule):
    def __init__(self,
        zarr_transform: Callable[[zarr.core.Array], torch.Tensor] = ZarrRandomCrop(size = 512),
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, ## augmenter
        keep_cols: List[str] = ["batch", "guide_id"],
        channels_to_keep: List[int] = [0, 1, 2, 3, 4, 5],
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.Dataset = ZarrDataset
        self.datasetkwargs = {
            "zarr_transform": zarr_transform,
            "transform": transform,
            "keep_cols": keep_cols,
            "channels_to_keep": channels_to_keep
        }

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.Dataset(
        self.train_data,
        **self.datasetkwargs
        )
        return DataLoader(train_dataset, batch_size = self.batch_size, num_workers = 8)

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.Dataset(
        self.val_data,
        **self.datasetkwargs
        )
        return DataLoader(val_dataset, batch_size = self.batch_size, num_workers = 8)

class ProteomicsDataModule(BaseDataModule):
    def __init__(self,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.Dataset = ProteomicsDataset
        self.datasetkwargs = {
        }

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.Dataset(
        self.train_data,
        **self.datasetkwargs
        )
        return DataLoader(train_dataset, batch_size = self.batch_size, num_workers = 8)

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.Dataset(
        self.val_data,
        **self.datasetkwargs
        )
        return DataLoader(val_dataset, batch_size = self.batch_size, num_workers = 8)



if __name__ == "__main__":

    d = ProteomicsDataModule(data_path = "/mnt/ps/home/CORP/johnny.xi/sandbox/proteomics/proteomics_minimap.parquet", batch_size = 3)
    d.setup(stage = "fit")
    dataloader = d.train_dataloader()
    print(next(iter(dataloader)))