import numpy as np

ny, nx, nlogtau = 50, 50, 50
model = np.zeros((ny, nx, nlogtau, 11), dtype=np.float32)

# 按 SIR 标准参数顺序填充（关键！顺序不能错）
model[:, :, :, 0] = 5777.0  # 0: 温度（K）
model[:, :, :, 1] = 0.0     # 1: 视向速度（km/s）
model[:, :, :, 2] = 1500.0  # 2: 磁场强度（G）
model[:, :, :, 3] = 1.0e5   # 3: 微湍流速度（cm/s）
model[:, :, :, 4] = 0.0     # 4: 宏湍流速度（km/s）
model[:, :, :, 5] = np.pi/4 # 5: 磁场倾角（rad）
model[:, :, :, 6] = 0.0     # 6: 磁场方位角（rad）
model[:, :, :, 7] = 0.0     # 7: 压力（默认0）
model[:, :, :, 8] = 0.0     # 8: 电子密度（默认0）
model[:, :, :, 9] = 1.0     # 9: 元素丰度
model[:, :, :, 10] = 0.0    # 10: 备用参数

np.save('test_field.npy', model)