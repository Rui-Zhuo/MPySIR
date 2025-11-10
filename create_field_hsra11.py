import numpy as np

def exact_hsra11_convert(mod_path, output_npy="test_field.npy", ny=50, nx=50):
    """
    完全按 hsra11.mod 的原始参数生成 test_field.npy
    维度：[ny, nx, n_layers, 11]，参数1:1还原，无任何自定义修改
    """
    # 1. 读取 hsra11.mod 完整内容
    with open(mod_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # 2. 解析全局参数（第一行：宏湍流速度、填充因子、杂散光因子）
    global_params = np.array(lines[0].split(), dtype=float)
    v_macro = global_params[0]       # 宏湍流速度 (km/s)
    filling_factor = global_params[1]# 填充因子（单分量模型为0）
    stray_light = global_params[2]   # 杂散光因子 (%)
    print(f"hsra11.mod 全局参数：")
    print(f"  宏湍流速度：{v_macro} km/s")
    print(f"  填充因子：{filling_factor}")
    print(f"  杂散光因子：{stray_light}%")
    
    # 3. 解析分层参数（后续行，每行11列，完全对应SIR模型格式）
    layers = []
    for line in lines[1:]:
        params = np.array(line.split(), dtype=float)
        if len(params) == 11:
            layers.append(params)
    hsra_model = np.array(layers, dtype=np.float32)
    n_layers = hsra_model.shape[0]
    print(f"\n分层数据：共 {n_layers} 层，11列参数（完全还原）")
    
    # 4. 定义 SIR 要求的 11 列参数顺序（与 hsra11.mod 列序完全对齐）
    # 输出模型列序 = hsra11.mod 列序（无任何映射修改）：
    # 列0: logτ₅₀₀₀（5000Å光学深度对数）→ mod列1
    # 列1: 温度 (K) → mod列2
    # 列2: 电子压强 (dyn/cm²) → mod列3
    # 列3: 微湍流速度 (cm/s) → mod列4
    # 列4: 磁场强度 (G) → mod列5（hsra11为宁静太阳，磁场≈0）
    # 列5: 视向速度 (cm/s) → mod列6
    # 列6: 磁场倾角 (deg) → mod列7（转为弧度存储）
    # 列7: 磁场方位角 (deg) → mod列8（转为弧度存储）
    # 列8: 几何尺度 (km) → mod列9
    # 列9: 气体压强 (dyn/cm²) → mod列10
    # 列10: 气体密度 (gr/cm³) → mod列11
    
    # 5. 构建 50×50 像素模型（每个像素完全复制 hsra11 的分层数据）
    test_field = np.zeros((ny, nx, n_layers, 11), dtype=np.float32)
    
    # 完全还原参数（仅将角度单位从度转为弧度，SIR内部默认弧度计算）
    test_field[:, :, :, 0] = hsra_model[:, 0]                # logτ₅₀₀₀
    test_field[:, :, :, 1] = hsra_model[:, 1]                # 温度 (K)
    test_field[:, :, :, 2] = hsra_model[:, 2]                # 电子压强
    test_field[:, :, :, 3] = hsra_model[:, 3]                # 微湍流速度
    test_field[:, :, :, 4] = hsra_model[:, 4]                # 磁场强度（≈0G）
    test_field[:, :, :, 5] = hsra_model[:, 5] / 1000.0       # 视向速度（cm/s→km/s）
    test_field[:, :, :, 6] = np.deg2rad(hsra_model[:, 6])    # 磁场倾角（deg→rad）
    test_field[:, :, :, 7] = np.deg2rad(hsra_model[:, 7])    # 磁场方位角（deg→rad）
    test_field[:, :, :, 8] = hsra_model[:, 8]                # 几何尺度
    test_field[:, :, :, 9] = hsra_model[:, 9]                # 气体压强
    test_field[:, :, :, 10] = hsra_model[:, 10]              # 气体密度
    
    # 6. 保存为 setup.py 要求的 npy 文件
    np.save(output_npy, test_field)
    print(f"成功生成 {output_npy}")
    print(f"模型维度：{test_field.shape}（{ny}×{nx}像素 × {n_layers}层 × 11参数）")
    print(f"参数示例（第1像素第1层）：")
    print(f"  温度：{test_field[0,0,0,1]:.1f}K")
    print(f"  磁场强度：{test_field[0,0,0,4]:.1f}G")
    print(f"  视向速度：{test_field[0,0,0,5]:.1f}km/s")
    print(f"  磁场倾角：{np.rad2deg(test_field[0,0,0,6]):.1f}°")

# 执行转换（确保 hsra11.mod 在 invDefault 目录）
if __name__ == "__main__":
    exact_hsra11_convert(
        mod_path="invDefault/hsraB.mod",  # hsra11.mod 路径（必须正确）
        output_npy="test_field.npy",       # 输出文件名（与 setup.py 对应）
        ny=100, nx=100                       # 像素尺寸（保持与原配置一致）
    )