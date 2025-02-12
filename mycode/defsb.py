import os
import shutil

# 指定需要检查的根目录
root_dir = r"d:\Desktop\python\NowcastNet\data\dataset\mrms\us_eval.split\MRMS_Final_Test_Patch"

# 遍历根目录下的所有子项
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    # 如果该项是一个目录，则进入检查
    if os.path.isdir(subfolder_path):
        delete_folder = False

        # 遍历该子文件夹内的所有文件（递归遍历子目录）
        for dirpath, dirnames, filenames in os.walk(subfolder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    # 判断文件大小是否为 0 字节
                    if os.path.getsize(file_path) == 0:
                        delete_folder = True
                        # 找到空文件后即可退出当前子文件夹的检查
                        break
                except OSError as e:
                    print(f"Error accessing file {file_path}: {e}")
            if delete_folder:
                break

        # 如果在该子文件夹内检测到空文件，则删除整个子文件夹
        if delete_folder:
            print(f"Deleting folder: {subfolder_path}")
            shutil.rmtree(subfolder_path)
