# fast_pyROCAUC

6.5 times faster than sklearn implementation, 2.5 times faster than numba jit implementation.

Derive c/c++ implementation from lightgbm.

### usage

```
make
```

run makefile and get **metric.dll**

```
from roc_auc import roc_auc_score
```

