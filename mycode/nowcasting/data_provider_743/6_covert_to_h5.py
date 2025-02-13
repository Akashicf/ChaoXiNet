import os
import glob
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(npz_file, dst_folder_full, dst_folder_solo, do_full, do_solo):
    """
    处理单个 npz 文件，将数据转换为 h5 格式：
      - 如果 do_full 为 True，则保存完整数据（键 'input' 与 'output'）到 dst_folder_full；
      - 如果 do_solo 为 True，则提取 data['input'][:, 1:2, 0] 和 data['output'][:, 1:2, 0]
        保存到 dst_folder_solo。
    """
    try:
        # 获取不含扩展名的基本文件名
        base_name = os.path.splitext(os.path.basename(npz_file))[0]

        # 加载 npz 文件
        data = np.load(npz_file)
        full_input = data['input']  # 原始形状: (10, 3, 3, 256, 256)
        full_output = data['output']

        # 如果需要转换完整数据，则写入 h5 文件（不启用压缩）
        if do_full:
            h5_full_path = os.path.join(dst_folder_full, base_name + '.h5')
            with h5py.File(h5_full_path, 'w') as f:
                f.create_dataset('input', data=full_input)
                f.create_dataset('output', data=full_output)

        # 如果需要转换 solo 数据，则提取并写入 h5 文件
        if do_solo:
            # 取 data['input'][:, 1:2, 0] 和 data['output'][:, 1:2, 0]
            solo_input = full_input[:, 1:2, 0]  # 形状：(10, 1, 256, 256)
            solo_output = full_output[:, 1:2, 0]  # 形状：(10, 1, 256, 256)
            h5_solo_path = os.path.join(dst_folder_solo, base_name + '.h5')
            with h5py.File(h5_solo_path, 'w') as f:
                f.create_dataset('input', data=solo_input)
                f.create_dataset('output', data=solo_output)

        return f"Processed: {npz_file}"
    except Exception as e:
        return f"Error processing {npz_file}: {str(e)}"


def main():
    # 源数据文件夹（所有 npz 文件）
    src_folder = r'd:\Desktop\python\NowcastNet\data\dataset\yangben'
    # 目标文件夹：完整数据转换后的 h5 文件存放位置
    dst_folder_full = r'd:\Desktop\python\NowcastNet\data\dataset\yangben_h5'
    # 目标文件夹：solo 数据转换后的 h5 文件存放位置
    dst_folder_solo = r'd:\Desktop\python\NowcastNet\data\dataset\yangben_solo_h5'

    # 设置选项：是否执行对应的转换任务
    do_full = True  # 是否转换完整数据（data['input'] 和 data['output']）
    do_solo = True  # 是否转换 solo 数据（data['input'][:, 1:2, 0] 和 data['output'][:, 1:2, 0]）

    # 确保目标文件夹存在
    os.makedirs(dst_folder_full, exist_ok=True)
    os.makedirs(dst_folder_solo, exist_ok=True)

    # 获取所有 npz 文件
    npz_files = glob.glob(os.path.join(src_folder, '*.npz'))
    if not npz_files:
        print("未找到任何 npz 文件！")
        return

    # 使用多进程并行转换
    with ProcessPoolExecutor() as executor:
        futures = []
        for npz_file in npz_files:
            future = executor.submit(
                process_file,
                npz_file,
                dst_folder_full,
                dst_folder_solo,
                do_full,
                do_solo
            )
            futures.append(future)

        # 输出转换结果
        for future in as_completed(futures):
            result = future.result()
            print(result)


if __name__ == '__main__':
    main()
