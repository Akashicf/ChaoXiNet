import os
import glob
import re

# 文件目录（请根据实际情况修改）
data_dir = r'../data/dataset/yangben'
# 查找所有 .npz 文件
files = glob.glob(os.path.join(data_dir, '*.npz'))

# 正则表达式匹配文件名，提取两个数字
# 文件格式形如：data_dir_000_sample_0.npz
pattern = re.compile(r"data_dir_(\d+)_sample_(\d+)\.npz")

# 用字典按第一个数字进行分组，值为[(second, full_path), ...]
groups = {}
for file in files:
    basename = os.path.basename(file)
    m = pattern.match(basename)
    if m:
        first_num = int(m.group(1))  # 第一个数字
        second_num = int(m.group(2))  # 第二个数字
        groups.setdefault(first_num, []).append((second_num, file))
    else:
        print("无法匹配文件名格式：", file)

# 对每个组按第二个数字进行排序（自然排序，确保8在10之前）
for first, lst in groups.items():
    lst.sort(key=lambda x: x[0])
    # 仅保留文件路径列表，排序后的顺序与样本顺序相同
    groups[first] = [f for _, f in lst]

# 定义删除指令，格式为： {first_number: instruction_string}
# instruction_string 可以是单个数字，如 "0" 或一个范围 "2-18"
deletion_instructions = {
    5: "0",
    7: "2-18",
    38: "0-11",
    65: "0-3",
    78: "0-11",
    163: "0-6",
    179: "0-13",
    223: "0-3"
}

# 根据删除指令进行删除操作
for first, instr in deletion_instructions.items():
    # 如果该组不存在，则跳过
    if first not in groups:
        print(f"文件组 {first} 不存在。")
        continue

    file_list = groups[first]
    # 解析指令：如果包含 '-'，则认为是范围，否则仅删除一个文件
    if '-' in instr:
        start_str, end_str = instr.split('-')
        start = int(start_str)
        end = int(end_str)
        indices_to_delete = list(range(start, end + 1))
    else:
        indices_to_delete = [int(instr)]

    # 检查索引是否在范围内
    indices_to_delete = [i for i in indices_to_delete if i < len(file_list)]
    if not indices_to_delete:
        print(f"对于文件组 {first}，指令 {instr} 无效（索引超出范围）。")
        continue

    # 为避免因删除文件导致索引错乱，从后往前删除
    for idx in sorted(indices_to_delete, reverse=True):
        file_to_delete = file_list[idx]
        print(f"删除文件：{file_to_delete}")
        try:
            os.remove(file_to_delete)
        except Exception as e:
            print(f"删除 {file_to_delete} 失败：{e}")
