import os
import copy
import json
import numpy as np
import awkward as ak
import torch.utils.data
import time

from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor
from src.data.tools import _pad
from src.data.fileio import _read_files
from src.data.preprocess import (
    AutoStandardizer,
    WeightMaker,
)
from src.dataset.functions_graph import create_graph

def _preprocess(table, options):
    indices = np.arange(
        len(table["X_hit"])
    )  
    if options["shuffle"]:
        np.random.shuffle(indices)
    return table, indices


def _load_next(filelist, load_range, options):
    table = _read_files(
        filelist, load_range,
    )
    table, indices = _preprocess(table, options)
    return table, indices


class _SimpleIter(object):
    r"""_SimpleIter
    Iterator object for ``SimpleIterDataset''.
    """

    def __init__(self, **kwargs):
        # inherit all properties from SimpleIterDataset
        self.__dict__.update(**kwargs)
        self.iter_count = 0  # to raise StopIteration when dataset_cap is reached
        if "dataset_cap" in kwargs and kwargs["dataset_cap"] is not None:
            self.dataset_cap = kwargs["dataset_cap"]
            self._sampler_options["shuffle"] = False
            print("!!! Dataset_cap flag set, disabling shuffling")
        else:
            self.dataset_cap = None

        # executor to read files and run preprocessing asynchronously
        self.executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None

        # init: prefetch holds table and indices for the next fetch
        self.prefetch = None
        self.table = None
        self.indices = []
        self.cursor = 0

        self._seed = None
        worker_info = torch.utils.data.get_worker_info()
        file_dict = self._init_file_dict.copy()
        if worker_info is not None:
            # in a worker process
            self._name += "_worker%d" % worker_info.id
            self._seed = worker_info.seed & 0xFFFFFFFF
            np.random.seed(self._seed)
            # split workload by files
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[worker_info.id :: worker_info.num_workers]
                assert len(new_files) > 0
                new_file_dict[name] = new_files
            file_dict = new_file_dict
        self.worker_file_dict = file_dict
        self.worker_filelist = sum(file_dict.values(), [])
        self.worker_info = worker_info
        self.restart()

    def restart(self):
        print("=== Restarting DataIter %s, seed=%s ===" % (self._name, self._seed))
        # re-shuffle filelist and load range if for training
        filelist = self.worker_filelist.copy()
        if self._sampler_options["shuffle"]:
            np.random.shuffle(filelist)
        if self._file_fraction < 1:
            num_files = int(len(filelist) * self._file_fraction)
            filelist = filelist[:num_files]
        self.filelist = filelist

        if self._init_load_range_and_fraction is None:
            self.load_range = (0, 1)
        else:
            (start_pos, end_pos), load_frac = self._init_load_range_and_fraction
            interval = (end_pos - start_pos) * load_frac
            if self._sampler_options["shuffle"]:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.load_range = (offset, offset + interval)
            else:
                self.load_range = (start_pos, start_pos + interval)

        self.ipos = 0 if self._fetch_by_files else self.load_range[0]
        # prefetch the first entry asynchronously
        self._try_get_next(init=True)

    def __next__(self):
        graph_empty = True
        self.iter_count += 1
        if self.dataset_cap is not None and self.iter_count > self.dataset_cap:
            raise StopIteration
        while graph_empty:
            if len(self.filelist) == 0:
                raise StopIteration
            try:
                i = self.indices[self.cursor]
            except IndexError:
                # case 1: first entry, `self.indices` is still empty
                # case 2: running out of entries, `self.indices` is not empty
                while True:
                    if self._in_memory and len(self.indices) > 0:
                        # only need to re-shuffle the indices, if this is not the first entry
                        if self._sampler_options["shuffle"]:
                            np.random.shuffle(self.indices)
                        break
                    if self.prefetch is None:
                        # reaching the end as prefetch got nothing
                        self.table = None
                        if self._async_load:
                            self.executor.shutdown(wait=False)
                        raise StopIteration
                    # get result from prefetch
                    if self._async_load:
                        self.table, self.indices = self.prefetch.result()
                    else:
                        self.table, self.indices = self.prefetch
                    # try to load the next ones asynchronously
                    self._try_get_next()
                    # check if any entries are fetched (i.e., passing selection) -- if not, do another fetch
                    if len(self.indices) > 0:
                        break
                # reset cursor
                self.cursor = 0
                i = self.indices[self.cursor]
            self.cursor += 1
            data, mc_energy, pi0_angle, pi0_distance, graph_empty = self.get_data(i)
        return data, mc_energy, pi0_angle, pi0_distance

    def _try_get_next(self, init=False):
        end_of_list = (
            self.ipos >= len(self.filelist)
            if self._fetch_by_files
            else self.ipos >= self.load_range[1]
        )
        if end_of_list:
            if init:
                raise RuntimeError(
                    "Nothing to load for worker %d" % 0
                    if self.worker_info is None
                    else self.worker_info.id
                )
            if self._infinity_mode and not self._in_memory:
                # infinity mode: re-start
                self.restart()
                return
            else:
                # finite mode: set prefetch to None, exit
                self.prefetch = None
                return
        if self._fetch_by_files:
            filelist = self.filelist[int(self.ipos) : int(self.ipos + self._fetch_step)]
            load_range = self.load_range
        else:
            filelist = self.filelist
            load_range = (
                self.ipos,
                min(self.ipos + self._fetch_step, self.load_range[1]),
            )
        print('Start fetching next batch, len(filelist)=%d, load_range=%s'%(len(filelist), load_range))
        if self._async_load:
            self.prefetch = self.executor.submit(
                _load_next,
                filelist,
                load_range,
                self._sampler_options,
            )
        else:
            self.prefetch = _load_next(
                filelist, load_range, self._sampler_options
            )
        self.ipos += self._fetch_step

    def get_data(self, i):
        # inputs
        # X = {k: self.table["_" + k][i].copy() for k in self._data_config.input_names}
        X = {k: self.table[k][i] for k in self.table.fields}

        [g, features_partnn], mc_energy, pi0_angle, pi0_distance, graph_empty = create_graph(
            X, self.for_training
        )

        return [g, features_partnn], mc_energy, pi0_angle, pi0_distance, graph_empty
        # return X, False


class SimpleIterDataset(torch.utils.data.IterableDataset):
    r"""Base IterableDataset.
    Handles dataloading.
    Arguments:
        file_dict (dict): dictionary of lists of files to be loaded.
        data_config_file (str): YAML file containing data format information.
        for_training (bool): flag indicating whether the dataset is used for training or testing.
            When set to ``True``, will enable shuffling and sampling-based reweighting.
            When set to ``False``, will disable shuffling and reweighting, but will load the observer variables.
        load_range_and_fraction (tuple of tuples, ``((start_pos, end_pos), load_frac)``): fractional range of events to load from each file.
            E.g., setting load_range_and_fraction=((0, 0.8), 0.5) will randomly load 50% out of the first 80% events from each file (so load 50%*80% = 40% of the file).
        fetch_by_files (bool): flag to control how events are retrieved each time we fetch data from disk.
            When set to ``True``, will read only a small number (set by ``fetch_step``) of files each time, but load all the events in these files.
            When set to ``False``, will read from all input files, but load only a small fraction (set by ``fetch_step``) of events each time.
            Default is ``False``, which results in a more uniform sample distribution but reduces the data loading speed.
        fetch_step (float or int): fraction of events (when ``fetch_by_files=False``) or number of files (when ``fetch_by_files=True``) to load each time we fetch data from disk.
            Event shuffling and reweighting (sampling) is performed each time after we fetch data.
            So set this to a large enough value to avoid getting an imbalanced minibatch (due to reweighting/sampling), especially when ``fetch_by_files`` set to ``True``.
            Will load all events (files) at once if set to non-positive value.
        file_fraction (float): fraction of files to load.
    """

    def __init__(
        self,
        file_dict,
        data_config_file,
        for_training=True,
        load_range_and_fraction=None,
        extra_selection=None,
        fetch_by_files=False,
        fetch_step=0.01,
        file_fraction=1,
        remake_weights=False,
        up_sample=True,
        weight_scale=1,
        max_resample=10,
        async_load=True,
        infinity_mode=False,
        in_memory=False,
        name="",
        laplace=False,
        edges=False,
        diffs=False,
        dataset_cap=None,
    ):
        self._iters = {} if infinity_mode or in_memory else None
        _init_args = set(self.__dict__.keys())
        self._init_file_dict = file_dict
        self._init_load_range_and_fraction = load_range_and_fraction
        self._fetch_by_files = fetch_by_files
        self._fetch_step = fetch_step
        self._file_fraction = file_fraction
        self._async_load = async_load
        self._infinity_mode = infinity_mode
        self._in_memory = in_memory
        self._name = name
        self.dataset_cap = dataset_cap  # used to cap the dataset to some fixed number of events - used for debugging purposes
        self.for_training = for_training
        # ==== sampling parameters ====
        self._sampler_options = {
            "up_sample": up_sample,
            "weight_scale": weight_scale,
            "max_resample": max_resample,
        }

        if for_training:
            self._sampler_options.update(training=True, shuffle=True, reweight=True)
        else:
            self._sampler_options.update(training=False, shuffle=False, reweight=False)
        self._init_args = set(self.__dict__.keys()) - _init_args


    def __iter__(self):
        if self._iters is None:
            kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
            return _SimpleIter(**kwargs)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            try:
                return self._iters[worker_id]
            except KeyError:
                kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
                self._iters[worker_id] = _SimpleIter(**kwargs)
                return self._iters[worker_id]
