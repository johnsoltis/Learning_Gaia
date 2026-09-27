import os
import h5py
import torch
from torch.nn import Sigmoid
import numpy as np
from typing import Dict, Tuple, Literal, Optional, List
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, get_worker_info
import time
from pathlib import Path
from gaia_feature_normalizer_020926 import feature_normer
from astropy.coordinates import SkyCoord
import astropy.units as u
rng = np.random.default_rng(1000)
sigmoid_op = Sigmoid()


class ORIGINAL_H5Reader:
    """
    Read chunks of 'random_index' and selected numeric columns from an HDF5 file.
    Applies only to HDF5 files of non-sharded Gaia data.
    """

    def __init__(self, h5_path: Path, selected_cols: List[str]):

        self.h5_path = h5_path
        self.selected_cols = selected_cols
        
        self.fh = None
        self.nrows = None
        self._init_handle()

    def _init_handle(self):
        self.fh = h5py.File(self.h5_path, "r")
        self.nrows = self.fh['random_index'].shape[0]

    def read_chunk(self, start: int, end: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Read [start:end) rows of random_index, source_id, and selected columns.
        Returns (random_index, source_id, {col: array})
        """
        r = self.fh['random_index'][start:end]
        cols = {c: self.fh[c][start:end] for c in self.selected_cols}
        return r, cols

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None
            
class SHARD_H5Reader:
    """
    Read chunks of 'random_index' and selected numeric columns from an HDF5 file.
    Applies only to HDF5 files of sharded Gaia data.
    """

    def __init__(self, h5_path: Path, selected_cols: List[str]):

        self.h5_path = h5_path
        self.selected_cols = selected_cols

        self.fh = None
        self.nrows = None
        self._init_handle()

    def _init_handle(self):
        self.fh = h5py.File(self.h5_path, "r")
        self.dset = self.fh["data"]
        self.nrows = self.dset.shape[0]

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None

class FormatError(Exception):
    """Exception raised for loading id shards."""
    def __init__(self, message="A id shard loading error occurred."):
        self.message = message
        super().__init__(self.message)

def _ddp_info():
    """Return (rank, world_size) if torch.distributed is initialized, else (0, 1)."""
    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    return rank, world_size
    
class HDF5IterableDataset(IterableDataset):
    """
    Args:
        READ_TYPE: "Original" | "Shard". Determines which file format is being used for the dataset. "Original" means that the original Gaia source catalog hdf5 files are used. "Shard" means that sharded h5 files are being used instead.
        filelist: list of data files.
        split: "train" | "val" | "test" (determines partition).
        set_type: "M" | "A" | "B" (determines which training set you are using).
        columns: Gaia source features you are using.
        shard_columns: the features in the shard files, in the same order as the shard files. Used to guarantee the correct feature selection.
        batch_size: This sets the size of batch_sizes emitted by the dataset. Note that if batch_size is set when READ_TYPE is "Original" then the minimum of batch_size and the post-partition file size will be emitted.
        shuffle_order_each_epoch: Determines whether the order in which id files are read should be shuffled each epoch. For training this should be True. Applies to both READ_TYPEs.
        shuffle_sources_in_file: Shuffles the ordering of the data in the batches. Applies only when READ_TYPE = "Shard" or batch_size < the number of rows post-partition.
        drop_last: This drops the last batch from a file if the batch size is below the standard amount. The benefits of this are unclear.
        normalization: dict {col: {'count':..., 'min':..., 'max':..., 'mean':..., 'std':..., 'est_median':..., 'est_p97_5'..., 'est_p2_5'...}} (columns are gaia features and entries are stats for this sets training data)
        norm_method: "NONE" | "MIN_MAX" | "Z_SCORE" | "SIGMOID_Z_SCORE" | "CUSTOM" | "P_Z_SCORE"
        equalize_batches_across_ranks: If True, the dataset guarantees that every (rank, worker) shard yields exactly the same number of batches per epoch. Required for DDP training with IterableDataset to prevent NCCL all-reduce hangs at epoch end. When True, set_epoch(epoch) MUST be called once per epoch BEFORE iteration begins. Default False preserves the original behavior for single-process inference / sampling code.
        num_workers_hint: Number of DataLoader workers per rank. Required (and only used) when equalize_batches_across_ranks=True, because set_epoch runs in the main process before workers fork and must know the total shard count (world_size * num_workers) up front. Must match the num_workers passed to DataLoader.
    """

    def __init__(
        self,
        READ_TYPE: Literal["Original", "Shard"],
        filelist: List[str],
        split: Literal["train", "val", "test"],
        set_type: Literal["M", "A", "B"],
        columns: List[str],
        shard_columns: List[str],
        batch_size: int,
        shuffle_order_each_epoch: bool,
        shuffle_sources_in_file: bool,
        drop_last: bool,
        normalization: Optional[Dict[str, Dict[str, float]]] = None,
        norm_method: Literal["NONE", "MIN_MAX", "Z_SCORE", "SIGMOID_Z_SCORE", "P_Z_SCORE"] = "NONE",
        equalize_batches_across_ranks: bool = False,
        num_workers_hint: int = 1,
        ):
        
        assert READ_TYPE == "Original", f"Shard read type is currently deprecated. You must use 'Original'. Currently using {READ_TYPE}."
        
        assert len(columns) > 0 and columns is not None, "Must specify features (columns)."
        for col in columns:
            assert col in normalization, f"All features must have normalization entry. Missing {col}."
            
        if READ_TYPE == "Shard":
            assert batch_size is not None and batch_size > 0, "Must enter batch size if using READ_TYPE = Shard"
            self.N_shard = len(filelist)
            assert self.N_shard >= 100, "The number of shard files must be greater than or equal to 100!"
        else:
            assert batch_size is None or batch_size > 0, "Batch size must either be greater than 0 or None."
        super().__init__()
        
        self.READ_TYPE = READ_TYPE
        self.filelist = filelist
        self.split = split
        self.set_type = set_type
        #Handles when proper motions will be converted into galactic coordinates
        if 'pml' in columns or 'pmb' in columns:
            self.PM_GAL = True
            col_list = ['ra', 'dec', 'pmra', 'pmdec']
            for column in columns:
                if column != 'pml' and column != 'pmb':
                    col_list += [column]
            self.load_columns = col_list
            self.desired_columns = list(columns)
            
        else:
            self.PM_GAL = False
            self.load_columns = list(columns)
            self.desired_columns = list(columns)
            
        self.shard_columns = shard_columns
        self.dim = len(self.desired_columns)
        self.batch_size = batch_size
        self.shuffle_order_each_epoch = shuffle_order_each_epoch
        self.shuffle_sources_in_file = shuffle_sources_in_file
        self.drop_last = drop_last
        self.normalization = normalization or {}
        self.method = norm_method.upper() if norm_method else "NONE"
        self.shard_columns = shard_columns
        if 'l' in columns and norm_method == 'CUSTOM':
            assert columns[0] == 'l', f"If you are using the l parameter and the CUSTOM normalization method, then the l parameter must be the first element in the feature names list. Currently it is {np.where(np.array(columns) == 'l')[0]}."
            self.ft_add_factor = 1
        else:
            self.ft_add_factor = 0

        # ---------- DDP batch-count equalization ----------
        # When equalize_batches_across_ranks=True, set_epoch() must be called
        # once per epoch before iteration starts. The dataset will then yield
        # exactly target_batches batches on every (rank, worker) shard, so
        # NCCL collectives stay synchronized at epoch end.
        self.equalize_batches_across_ranks = equalize_batches_across_ranks
        self.num_workers_hint = max(1, int(num_workers_hint))
        # Populated by set_epoch() each epoch:
        self._epoch_shard_files = None      # list[list[str]] of length world_size*num_workers
        self._epoch_target_batches = None   # int, the per-shard batch cap
        self._current_epoch = -1
        # Cached per-file batch counts (computed once, aligned with self.filelist):
        self.file_batch_counts = None
        if self.equalize_batches_across_ranks:
            self._scan_file_batch_counts()
    # ------------------------------------------------
    
    def _pm_gal_transform(self, ra, dec, pmra, pmdec):
        g = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                 pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr).galactic
        return g.pm_l_cosb.value, g.pm_b.value
    
    def _number_loop(self, file_i):
        """
        Reads hdf5 shard files indices. These indices are used for partitioning the data into different sets and partitions.
        Assumes there are <1000 shard files.
        """
        if file_i[-3:] == '.h5' and file_i[-5] == '_':
            return int(file_i[-4])
        elif file_i[-3:] == '.h5' and file_i[-6] == '_':
            return int(file_i[-5:-3])
        elif file_i[-3:] == '.h5' and file_i[-7] == '_':
            return int(file_i[-6:-3])

        raise FormatError(f"Unsupported file name format. The indices at the end of file names should not be greater than three digits. Failed on {file_i}.")
        
    def _partition_selection(self, p_index, p_splits = 100):
        """
        Uses the shard file index or the Gaia source catalog random_index feature to partition the data. Shard files are created with the random_index, so the result is the same.
        Assumes there are >=100 shard files
        """
        shard_factor = p_splits/100
        
        if self.set_type == 'M' and self.split == 'train':
            return (p_index%p_splits < (80 * shard_factor))
        elif self.set_type == 'A' and self.split == 'train':
            return (p_index%p_splits < (32 * shard_factor))
        elif self.set_type == 'B' and self.split == 'train':
            return (p_index%p_splits >= (40 * shard_factor)) & (p_index%p_splits < (72 * shard_factor))
            
        elif self.set_type == 'M' and self.split == 'val':
            return ((p_index%p_splits >= (80 * shard_factor)) & (p_index%p_splits < (90 * shard_factor)))
        elif self.set_type == 'A' and self.split == 'val':
            return ((p_index%p_splits >= (32 * shard_factor)) & (p_index%p_splits < (36 * shard_factor)))
        elif self.set_type == 'B' and self.split == 'val':
            return (p_index%p_splits >= (72 * shard_factor)) & (p_index%p_splits < (76 * shard_factor))
            
        elif self.set_type == 'M' and self.split == 'test':
            return (p_index%p_splits >= (90 * shard_factor))
        elif self.set_type == 'A' and self.split == 'test':
            return ((p_index%p_splits >= (36 * shard_factor)) & (p_index%p_splits < (40 * shard_factor)))
        elif self.set_type == 'B' and self.split == 'test':
            return (p_index%p_splits >= (76 * shard_factor)) & (p_index%p_splits < (80 * shard_factor))

    # ============================================================
    # DDP batch-count equalization machinery
    # ============================================================
    def _n_batches_for_rows(self, n_rows: int) -> int:
        """
        Mirror of the emission logic in _yield_from_hdf5_file /
        _yield_from_shard_file: how many batches a file with n_rows
        post-partition rows will produce.
        """
        if n_rows <= 0:
            # _yield_from_hdf5_file's n_rows<1 branch still yields one
            # (empty) batch. _yield_from_shard_file would yield nothing.
            # We match the Original path (the only supported one); for
            # Shard, the same branch is unreachable because Shard files
            # are either fully in-partition (n_rows = file_nrows) or
            # fully out (handled as count=0 in the scanner).
            return 1 if self.READ_TYPE == "Original" else 0
        if self.batch_size is None or self.batch_size >= n_rows:
            return 1
        if self.drop_last:
            return n_rows // self.batch_size
        return (n_rows + self.batch_size - 1) // self.batch_size

    def _scan_file_batch_counts(self):
        """
        One-time pass over every file in self.filelist: determine the
        post-partition row count, convert to a batch count, and store
        the result in self.file_batch_counts (aligned with self.filelist).

        Reads only random_index (Original) or the data dataset's shape
        attribute (Shard) -- no feature columns -- so the scan is cheap.
        Runs on every rank independently; all ranks produce the same
        counts because the inputs are deterministic.
        """
        filelist = list(self.filelist)
        counts = np.zeros(len(filelist), dtype=np.int64)

        for i, file_i in enumerate(filelist):
            if self.READ_TYPE == "Shard":
                # Shard files are pre-partitioned by filename: the whole
                # file is either in this split or not.
                in_partition = self._partition_selection(
                    self._number_loop(file_i), p_splits=self.N_shard
                )
                if not in_partition:
                    counts[i] = 0
                    continue
                with h5py.File(file_i, "r") as fh:
                    n_rows = int(fh["data"].shape[0])
            else:
                with h5py.File(file_i, "r") as fh:
                    r = fh["random_index"][:]
                set_mask = self._partition_selection(r)
                n_rows = int(np.count_nonzero(set_mask))

            counts[i] = self._n_batches_for_rows(n_rows)

        self.file_batch_counts = counts

    def set_epoch(self, epoch: int):
        """
        Called once per epoch by the training loop, on every rank, BEFORE
        iteration begins. Produces:
          - self._epoch_shard_files: a balanced assignment of files to
            each (rank, worker) shard, identical on every rank because
            the RNG is seeded with the epoch number.
          - self._epoch_target_batches: the per-shard batch cap. Every
            shard will emit exactly this many batches in __iter__.

        No-op when equalize_batches_across_ranks=False (preserves the
        original behavior used by the sampling code).
        """
        self._current_epoch = epoch
        if not self.equalize_batches_across_ranks:
            return
        if self.file_batch_counts is None:
            # Defensive: scan should have run in __init__, but if a
            # subclass skipped it, do it now.
            self._scan_file_batch_counts()

        _, world_size = _ddp_info()
        num_workers = self.num_workers_hint
        total_shards = world_size * num_workers

        filelist = np.asarray(self.filelist)
        counts = self.file_batch_counts.copy()

        # 1. Deterministic shuffle: same on every rank (epoch-seeded RNG).
        if self.shuffle_order_each_epoch:
            perm = np.random.default_rng(int(epoch)).permutation(len(filelist))
            filelist = filelist[perm]
            counts = counts[perm]

        # 2. Greedy LPT (Longest Processing Time) assignment of files to
        #    shards. Walk files in descending batch-count order and place
        #    each on the currently lightest shard. Near-optimal for
        #    minimizing the max-min spread across shards.
        order = np.argsort(-counts, kind='stable')
        shard_files = [[] for _ in range(total_shards)]
        shard_totals = np.zeros(total_shards, dtype=np.int64)
        for idx in order:
            s = int(np.argmin(shard_totals))
            shard_files[s].append(str(filelist[idx]))
            shard_totals[s] += counts[idx]

        # 3. Per-shard cap: minimum total across shards.
        self._epoch_target_batches = int(shard_totals.min())
        self._epoch_shard_files = shard_files

        if _ddp_info()[0] == 0:  # rank 0 only
            print(f"[set_epoch {epoch}] shard batch totals: "
                    f"min={shard_totals.min()}, max={shard_totals.max()}, "
                    f"mean={shard_totals.mean():.1f}, "
                    f"discarded={int(shard_totals.sum() - shard_totals.min()*len(shard_totals))}",
                    flush=True)

    # ============================================================

    def _hdf5_files_for_this_process(self) -> List[int]:
        """
        Partition the file list across (rank, workers) to avoid duplication.
        Also determines which files fall in the desired partition and set if READ_TYPE is Shard.

        When equalize_batches_across_ranks is True, returns the pre-planned
        file list for this (rank, worker) shard as computed by set_epoch().
        """
        # ----- Equalized path: use the plan from set_epoch() -----
        if self.equalize_batches_across_ranks:
            if self._epoch_shard_files is None:
                raise RuntimeError(
                    "HDF5IterableDataset.set_epoch(epoch) must be called once "
                    "per epoch BEFORE iteration begins when "
                    "equalize_batches_across_ranks=True."
                )
            rank, world_size = _ddp_info()
            winfo = get_worker_info()
            num_workers = winfo.num_workers if winfo is not None else 1
            worker_id = winfo.id if winfo is not None else 0
            if num_workers != self.num_workers_hint:
                raise RuntimeError(
                    f"num_workers mismatch: dataset was planned for "
                    f"num_workers_hint={self.num_workers_hint} but DataLoader "
                    f"is using num_workers={num_workers}. Pass the same value "
                    f"to both."
                )
            shard_id = rank * num_workers + worker_id
            return list(self._epoch_shard_files[shard_id])

        # ----- Original (unequalized) path -----
        filelist = np.array(self.filelist)
        
        #The shard files were created using the random_index. The trailing index on the filename can be used to partition the data.
        #The loop below reads the index off each file name, determines if it falls within the desired partition and set. It creates a boolean mask that is then applied to the filelist.
        if self.READ_TYPE == "Shard":
            file_list_mask = np.empty(len(filelist), dtype = bool)
            for i in range(self.N_shard):
                file_i = filelist[i]
                file_list_mask[i] = self._partition_selection(self._number_loop(file_i), p_splits = self.N_shard)
            filelist = filelist[file_list_mask]
            if len(filelist) < 1:
                raise FormatError("Filelist is empty. Masking procedure using file indices failed.")
                
        rank, world_size = _ddp_info()
        winfo = get_worker_info()
        num_workers = winfo.num_workers if winfo is not None else 1
        worker_id = winfo.id if winfo is not None else 0
        
        #Shuffles the order of the file list. It will do this every epoch.
        if self.shuffle_order_each_epoch == True:
            rng.shuffle(filelist)

        # Shard files by combined (rank, worker)
        total_shards = world_size * num_workers
        #print("total_shards", total_shards)
        #n_even = (len(filelist) // total_shards) * total_shards
        #filelist = filelist[:n_even]  # drop the remainder files
        
        shard_id = rank * num_workers + worker_id
        #print("shard_id", shard_id)
        return filelist[shard_id::total_shards]
            
    def _normer_array(self, col, data, TENSOR = False, file_i = ''):
        """
        Handles the normalization of the data.
        Assumes that the data is a single column (i.e., is corresponding to one feature but many sources)
        col is the name of feature, not its index.
        """
        params = self.normalization.get(col, {})
        method = self.method
        shift = 0
        if method == "MIN_MAX":
            mn = params.get('min', None)
            mx = params.get('max', None)
            if mn is None or mx is None:
                raise FormatError(f"No min or max provided for min max normalization of {col}.")
            scale = (mx - mn)
            if scale is None or scale == 0:
                raise FormatError(f"MIN==MAX for {col}. Remove this feature.")
            inv = 1.0 / float(scale)
            shift = -float(mn)
            return (data + shift) * inv

        elif method == "Z_SCORE":
            mean = params.get('mean', None)
            std = params.get('std', None)
            if mean is None:
                raise FormatError(f"No mean provided for z score normalization of {col}.")
            if std is None:
                raise FormatError(f"No standard deviation (std) provided for z score normalization of {col}.")
            elif std == 0 or std == 0.0:
                raise FormatError(f"Standard deviation (std) for {col} is {std}.")
            inv = 1.0 / float(std)
            shift = -float(mean)
            return (data + shift) * inv

        elif method == "SIGMOID_Z_SCORE":
            mean = params.get('mean', None)
            std = params.get('std', None)
            if mean is None:
                raise FormatError(f"No mean provided for sigmoid z score normalization of {col}.")
            if std is None:
                raise FormatError(f"No standard deviation (std) provided for sigmoid z score normalization of {col}.")
            elif std == 0 or std == 0.0:
                raise FormatError(f"Standard deviation (std) for {col} is {std}.")
            inv = 1.0 / float(std)
            shift = -float(mean)
            if TENSOR == True:
                return sigmoid_op(((data + shift) * inv))
            return 1.0 / (1.0 + np.exp(-((data + shift) * inv)))
        
        elif method == "CUSTOM":
            return feature_normer(data, col, self.normalization, debug_file_name = file_i)
            
        elif method == "P_Z_SCORE":
            median = params.get('est_median', None)
            p97_5 = params.get('est_p97_5', None)
            p2_5 = params.get('est_p2_5', None)
            if median is None:
                raise FormatError(f"No median provided for percentile z score normalization of {col}.")
            if p97_5 is None:
                raise FormatError(f"No 97.5 percentile provided for percentile z score normalization of {col}.")
            if p2_5 is None:
                raise FormatError(f"No 2.5 percentile provided for percentile z score normalization of {col}.")
            inv = 1 / (float(p97_5) - float(p2_5))
            shift -= float(median)
            return (data + shift)*inv
            
        elif method == "NONE":
            return data
            
        else:
            raise FormatError(f"Method type {method} not recognized. Must use valid method type: NONE, MIN_MAX, Z_SCORE, SIGMOID_Z_SCORE, or CUSTOM.")
    
    def _yield_from_shard_file(self, filepath):
        """
        Yields batches from shard files. Reads in one shard file and then reads batches out one by one.
        Assumes data is in the format (# of sources, 2, # of features), where [:,0,:] is the data and [:,1,:] is the mask.
        Assumes when mask = 1 the data is visible, when mask = 0 the data is masked out.
        """
        reader = SHARD_H5Reader(filepath, selected_cols = self.desired_columns)
        total = reader.nrows
        
        if self.shuffle_sources_in_file:
            row_index = np.arange(total)
            rng.shuffle(row_index)
        
        # Emit pre-batches directly to reduce Python overhead
        bs = self.batch_size
        
        # Iterate contiguous blocks
        for start in range(0, total, bs):
            end = min(start + bs, total)
            #if the final batch would be less than the desired batch size and drop_last is True, move to the next file
            if end - start < bs and self.drop_last:
                break
                
            if self.shuffle_sources_in_file:
                batch = reader.dset[np.sort(row_index[start:end])]
            else:
                batch = reader.dset[start:end]
            
            batch_length = batch.shape[0]
            if batch.shape[1] < 2:
                raise ValueError(f"Expected data+mask in {filepath}, i.e., [#sources, 2, #features], got {batch.shape} instead")
            
            data = batch[:, 0]         # [B, F]
            mask = torch.zeros((batch_length, self.dim), dtype = torch.bool)
            data = torch.zeros((batch_length, self.dim), dtype = torch.float32)
            
            #The features in the shard file may be different the features used for training. This loop guarantees that the correct features are selected.
            #Note that it explicitly requires that both lists of features be in the same order!
            c_out = 0
            for c in range(len(self.shard_columns)):
                if self.shard_columns[c] == self.desired_columns[c_out]:
                    mask[:, c_out] = torch.from_numpy(batch[:, 1, c].astype(bool))
                    if self.method == "CUSTOM":
                        data[mask[:,c_out], c_out] = torch.from_numpy(self._normer_array(self.shard_columns[c], batch[mask[:,c_out], 0, c]))
                    else:
                        data[mask[:,c_out], c_out] = self._normer_array(self.shard_columns[c], torch.from_numpy(batch[mask[:,c_out], 0, c]), TENSOR = True)
                    c_out += 1
            if c_out <  self.dim:
                raise FormatError(f"Missing desired features from shard file! Missing {self.desired_columns[c_out:]}")
            yield (data, mask)
            
        reader.close()
            
    def _yield_from_hdf5_file(self, filepath):
        """
        Original Gaia source catalog hdf5 files (non-sharded).
        Vectorized NumPy path + one-time Torch conversion.
        """
        reader = ORIGINAL_H5Reader(filepath, selected_cols=self.load_columns)
        n_rows_total = reader.nrows
        r, cols_data = reader.read_chunk(0, n_rows_total)
        reader.close()

        # Compute partition membership from random_index
        set_mask = self._partition_selection(r)  # boolean array
        n_rows = int(np.count_nonzero(set_mask))
        #print(f"File Size: {n_rows} sources")
        # Early exit with empty batch (shape-consistent)
        if n_rows < 1:
            data = torch.zeros((0, self.dim + self.ft_add_factor), dtype=torch.float32)
            mask = torch.zeros((0, self.dim + self.ft_add_factor), dtype=torch.bool)
            yield (data, mask)
            return

        # NumPy preallocations
        mask_np = np.empty((n_rows, self.dim + self.ft_add_factor), dtype=bool)
        data_np = np.zeros((n_rows, self.dim + self.ft_add_factor), dtype=np.float32)

        # Local aliases (minor micro-optimization)
        cols = self.desired_columns
        method = self.method

        pml_raw = None
        pmb_raw = None
        # Per-feature masked normalization in NumPy
        for c, col in enumerate(cols):
            #Because I am now including a transformation of the proper motion into galactic coordinates, data reading because a bit more complicated. If I am not using proper motion in galactic coordinates, default to the old method.
            if col == 'pml' and pml_raw is None:
                ra_vals = cols_data['ra'][set_mask]
                dec_vals = cols_data['dec'][set_mask]
                pmra_vals = cols_data['pmra'][set_mask]
                pmdec_vals = cols_data['pmdec'][set_mask]
                pml, pmb = self._pm_gal_transform(ra_vals,dec_vals,pmra_vals,pmdec_vals)
                pml_raw = pml
                pmb_raw = pmb
                col_vals = pml
                not_nan_PM_GAL = ~np.isnan(ra_vals)&~np.isnan(dec_vals)&~np.isnan(pmra_vals)&~np.isnan(pmdec_vals)
                not_nan = not_nan_PM_GAL
            elif col == 'pml':
                col_vals = pml_raw
                not_nan = not_nan_PM_GAL
            elif col == 'pmb' and pmb_raw is None:
                ra_vals = cols_data['ra'][set_mask]
                dec_vals = cols_data['dec'][set_mask]
                pmra_vals = cols_data['pmra'][set_mask]
                pmdec_vals = cols_data['pmdec'][set_mask]
                pml, pmb = self._pm_gal_transform(ra_vals,dec_vals,pmra_vals,pmdec_vals)
                pml_raw = pml
                pmb_raw = pmb
                col_vals = pmb
                not_nan_PM_GAL = ~np.isnan(ra_vals)&~np.isnan(dec_vals)&~np.isnan(pmra_vals)&~np.isnan(pmdec_vals)
                not_nan = not_nan_PM_GAL
            elif col == 'pmb':
                col_vals = pmb_raw
                not_nan = not_nan_PM_GAL
            else:
                col_vals = cols_data[col][set_mask]                      # 1D NumPy array
                not_nan = ~np.isnan(col_vals)
                
            if col == 'l' and c == 0 and method == "CUSTOM":
                mask_np[:, c] = not_nan
                mask_np[:, c + self.ft_add_factor] = not_nan
            elif col == 'l' and c > 0 and method == "CUSTOM":
                raise FormatError(f"If using l and CUSTOM normalization, one must have l first! l found in position {c} in feature name list.")
            else:
                mask_np[:, c + self.ft_add_factor] = not_nan
                    
            if not_nan.any():
                if method == "CUSTOM":
                    # CUSTOM path returns NumPy; cast once here
                    normalized = self._normer_array(
                        col, col_vals[not_nan], file_i=str(filepath)
                    )
                    if col == 'l' and c == 0:
                        data_np[not_nan, c] = normalized[0].astype(np.float32, copy=False)
                        data_np[not_nan, c + self.ft_add_factor] = normalized[1].astype(np.float32, copy=False)
                    elif col == 'l' and c > 0:
                        raise FormatError(f"If using l and CUSTOM normalization, one must have l first! l found in position {c} in feature name list.")
                    else:
                        data_np[not_nan, c + self.ft_add_factor] = normalized.astype(np.float32, copy=False)
                else:
                    # Use NumPy branch of _normer_array for speed
                    normalized = self._normer_array(
                        col, col_vals[not_nan], TENSOR=False
                    ).astype(np.float32, copy=False)
                    data_np[not_nan, c] = normalized
        
        #batch size is None, or greater than or equal to the number of post-partition rows in the file, simply emit the post-partition file every time.
        if self.batch_size is None or self.batch_size >= n_rows:
            # One-time Torch conversion
            data = torch.from_numpy(data_np)
            mask = torch.from_numpy(mask_np)
            yield (data, mask)
            return
       
        #Only shuffle within the file if the whole post-partition file is not emitted.
        if self.shuffle_sources_in_file:
            perm = rng.permutation(n_rows)
            data_np = data_np[perm]
            mask_np = mask_np[perm]

        chunk_rows = self.batch_size
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            if (end - start) < chunk_rows and self.drop_last:
                break
            # from_numpy shares memory with the slice; the slice itself is a
            # view, so this is O(1) plus the conversion. The downstream
            # consumer should not mutate these tensors (it doesn't).
            chunk_data = torch.from_numpy(data_np[start:end])
            chunk_mask = torch.from_numpy(mask_np[start:end])
            yield (chunk_data, chunk_mask)
                
    def __iter__(self):
        global rng
        winfo = get_worker_info()
        worker_id = winfo.id if winfo is not None else 0
        rank, _ = _ddp_info()
        rng = np.random.default_rng((self._current_epoch, rank, worker_id, 1000))

 
        hdf5_files = self._hdf5_files_for_this_process()

        if len(hdf5_files) < 1:
            # Don't raise — return zero loss so the rank stays alive
            rank, world_size = _ddp_info()
            print("No Files Left, rank:", rank)
            return 0.0, 0.0, 0.0
            #raise FormatError(f"No HDF5 files selected.")

        # Per-shard batch cap when equalizing across ranks; None = unbounded.
        remaining = self._epoch_target_batches if self.equalize_batches_across_ranks else None

        for file_i in hdf5_files:
            if remaining is not None and remaining <= 0:
                break
            if self.READ_TYPE == "Shard":
                file_iter = self._yield_from_shard_file(file_i)
            elif self.READ_TYPE == "Original":
                file_iter = self._yield_from_hdf5_file(Path(file_i))
            else:
                continue
            for batch in file_iter:
                yield batch
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
