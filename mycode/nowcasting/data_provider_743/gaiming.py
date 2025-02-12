import os

# 目标目录路径
path = r"d:\Desktop\python\NowcastNet\data\dataset\yangben_augmented"

# 遍历目录下所有文件
for filename in os.listdir(path):
    # 只处理 npz 文件
    if filename.endswith(".npz"):
        # 将 'sample_' 替换为 'sampleaugmented_'
        new_filename = filename.replace("sample_", "sampleaugmented_")
        # 如果有替换发生且新旧文件名不同，才进行重命名
        if new_filename != filename:
            old_path = os.path.join(path, filename)
            new_path = os.path.join(path, new_filename)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")
