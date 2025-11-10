import numpy as np

# 1. 3个波长点（覆盖5、6号谱线核心位置，单位：埃）
wave_data = np.array([5247.05, 5248.63, 5250.21], dtype=np.float32)
# 对应说明：5247.05（Fe I 5号谱线）、5248.63（谱线中间点）、5250.21（Fe I 6号谱线）

# 2. 保存文件
np.save('test_wav.npy', wave_data)