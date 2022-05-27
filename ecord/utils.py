import os
import random

import torch

import numpy as np
import scipy as sp


def set_global_seed(seed=-1):
    if seed < 0:
        seed = random.randint(0, 2 ** 32)
    print("\n------------------------------------------")
    print(f"\tSetting global seed using {seed}.")
    print("------------------------------------------\n")
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** 32))
    sp.random.seed(random.randint(0, 2 ** 32))
    torch.manual_seed(random.randint(0, 2 ** 32))
    torch.cuda.manual_seed(random.randint(0, 2 ** 32))
    torch.cuda.manual_seed_all(random.randint(0, 2 ** 32))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mk_dir(dir, quiet=False):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            if not quiet:
                print('created directory: ', dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
        except Exception:
            pass
    else:
        if not quiet:
            print('directory already exists: ', dir)