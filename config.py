import joblib
import numpy as np


class DefaultConfig(object):
    random_state = 42

    # 多进程参数设置
    n_jobs = int(joblib.cpu_count() * 0.9)

    # 模型参数设置
    test_size = 264  # "2019-10-21"

    steps = np.array([1, 2, 3])  # 1 -> "step 1", 2 -> "step 2", 3 -> "step 3"
    window_length = 12

    strategy = "direct"

    # station = "S.W. PIER MI"
    station = "ROCK CUT MI"
