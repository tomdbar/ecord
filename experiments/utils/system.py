import gc
import os

import torch

from ecord.utils import mk_dir

import pickle
import bz2
import _pickle as cPickle
# import pickle as cPickle

import snappy

def _to_pkl_fname(fname):
    if os.path.splitext(fname)[-1] != ".pkl":
        fname += ".pkl"
    return fname

def _to_pbz2_fname(fname):
    if os.path.splitext(fname)[-1] != ".pbz2":
        fname += ".pbz2"
    return fname

def _to_snappy_fname(fname):
    if os.path.splitext(fname)[-1] != ".snappy":
        fname += ".snappy"
    return fname

def _to_compressed_fname(fname, use_snappy=True):
    if use_snappy:
        return _to_snappy_fname(fname)
    else:
        return _to_pbz2_fname(fname)

def save_pickle(fname, obj, verbose=False):
    if verbose:
        print(f"Saving to {fname}", end="...")
    fname = _to_pkl_fname(fname)
    with open(fname, 'wb') as file:
        pickle.dump(obj, file)
    if verbose:
        print(f"done.")

def load_pickle(fname, verbose=False):
    fname = _to_pkl_fname(fname)
    if verbose:
        print(f"Loading from {fname}", end="...")
    with open(fname, 'rb') as file:
        obj= pickle.load(file)
    if verbose:
        print(f"done.")
    return obj

def save_compressed_pickle_as_pbz2(fname, obj, verbose=False):
    fname = _to_pbz2_fname(fname)
    if verbose:
        print(f"Saving to {fname}", end="...")
    with bz2.BZ2File(fname, 'w') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"done.")

def load_compressed_pickle_from_pbz2(fname, verbose=False):
    if verbose:
        print(f"Loading from {fname}", end="...")
    fname = _to_pbz2_fname(fname)
    with bz2.BZ2File(fname, 'rb') as file:
        obj= pickle.load(file)
    if verbose:
        print(f"done.")
    return obj

def save_compressed_pickle_as_snappy(fname, obj, verbose=False):
    fname = _to_snappy_fname(fname)
    if verbose:
        print(f"Saving to {fname}", end="...")
    with open(fname, 'wb') as file:
        file.write(snappy.compress(pickle.dumps(obj)))
    if verbose:
        print(f"done.")

def load_compressed_pickle_from_snappy(fname, verbose=False):
    if verbose:
        print(f"Loading from {fname}", end="...")
    fname = _to_snappy_fname(fname)
    with open(fname, 'rb') as file:
        obj = pickle.loads(snappy.uncompress(file.read()))
    if verbose:
        print(f"done.")
    return obj

def save_to_disk(fname, obj, compressed=True, verbose=False):
    if compressed:
        return save_compressed_pickle_as_snappy(fname, obj, verbose)
    else:
        return save_pickle(fname, obj, verbose)

def load_from_disk(fname, compressed=True, verbose=False):
    if compressed:
        return load_compressed_pickle_from_snappy(fname, verbose)
    else:
        return load_pickle(fname, verbose)

def export_script(script_fname, target_dir):
    mk_dir(target_dir, quiet=True)
    target_fname = os.path.join(target_dir, os.path.split(script_fname)[-1])
    with open(script_fname, 'r') as f:
        with open(target_fname, 'w') as out:
            for line in f.readlines():
                out.write(line)

def export_summary(fname, content):
    mk_dir(os.path.dirname(fname), quiet=True)
    with open(fname, "w") as f:
        if type(content) is list:
            f.write('\n'.join(content))
        else:
            f.write(content)

def print_tensors_on_gpu(num_devices=None):
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data')):
                if obj.is_cuda:
                    print(f"type : {type(obj)}, size : {obj.size()}, dtype : {obj.dtype}, device : {obj.device}, has_grads : {obj.grad is not None}")
        except:
            pass
    for i in range(num_devices):
        try:
            print( torch.cuda.memory_summary(device=i, abbreviated=False) )
        except:
            pass