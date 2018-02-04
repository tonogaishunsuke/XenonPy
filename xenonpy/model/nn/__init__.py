# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import warnings as warnings

with warnings.catch_warnings():
    # warnings.simplefilter('default')
    try:
        import torch
    except ImportError:
        warnings.warn("Can't fing pytorch, will not load neural network modules.", RuntimeWarning)
    else:
        from .base import *
        from .random_model import *
        from .layer1 import *
