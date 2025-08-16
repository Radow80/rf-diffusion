import os
import shutil
import random
from pathlib import Path

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def copy_random_files(src_dir, dst_dir, num_files):
    """从源目录中随机复制指定数量的文件到目标目录"""
    # 获取所有.mat文件
    all_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.mat'):
                all_files.append(os.path.join(root, file))
    
    # 随机选择文件
    selected_files = random.sample(all_files, min(num_files, len(all_files)))
    
    # 确保目标目录存在
    ensure_dir(dst_dir)
    
    # 复制文件
    for src_file in selected_files:
        dst_file = os.path.join(dst_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dst_file)
    
    return selected_files

def copy_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # 创建必要的目录
    train_dir = os.path.join(data_dir, 'train')
    test_fall_dir = os.path.join(data_dir, 'test_fall')
    test_unfall_dir = os.path.join(data_dir, 'test_unfall')
    
    # 原始数据目录
    original_train_dir = '/srv/dataset/fall_det_12_12/train'
    original_test_fall_dir = '/srv/dataset/fall_det_12_12/test/fall'
    original_test_unfall_dir = '/srv/dataset/fall_det_12_12/test/unfall'
    
    # 复制训练数据（从各个子目录中均匀采样）
    print("准备训练集...")
    train_subdirs = ['walk_silence', 'walk_sit', 'squat', 'dance', 'Open_Close_door']
    files_per_subdir = 20  # 每个子目录20个文件，总共100个
    for subdir in train_subdirs:
        src_subdir = os.path.join(original_train_dir, subdir)
        dst_subdir = os.path.join(train_dir, subdir)
        ensure_dir(dst_subdir)
        copy_random_files(src_subdir, dst_subdir, files_per_subdir)
    
    # 复制测试数据
    print("准备非跌倒测试集...")
    copy_random_files(original_test_unfall_dir, test_unfall_dir, 50)
    
    print("准备跌倒测试集...")
    copy_random_files(original_test_fall_dir, test_fall_dir, 50)
    
    print("数据集准备完成！")
    print(f"训练集：{train_dir}")
    print(f"非跌倒测试集：{test_unfall_dir}")
    print(f"跌倒测试集：{test_fall_dir}")
    
    # 创建数据列表文件
    def create_file_list(data_dir, output_file):
        with open(output_file, 'w') as f:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.mat'):
                        f.write(os.path.join(root, file) + '\n')
    
    create_file_list(train_dir, os.path.join(data_dir, 'train/train.txt'))
    create_file_list(test_unfall_dir, os.path.join(data_dir, 'test_unfall/test.txt'))
    create_file_list(test_fall_dir, os.path.join(data_dir, 'test_fall/test.txt'))

if __name__ == '__main__':
    copy_dataset() 