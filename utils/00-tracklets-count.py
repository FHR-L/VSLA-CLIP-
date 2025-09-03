import matplotlib.pyplot as plt
from collections import defaultdict
from brokenaxes import brokenaxes

# 定义文件路径
file_paths = [
    '/Users/luowenlong/Documents/datasets/G2A-Plus/train_name.txt',
    '/Users/luowenlong/Documents/datasets/G2A-Plus/test_name.txt']

# 初始化字典，用于统计id < 10000和id >= 10000的序列长度
sequence_lengths_lt_10000 = defaultdict(int)  # id < 10000
sequence_lengths_ge_10000 = defaultdict(int)  # id >= 10000

# 读取文件并统计序列长度
for file_path in file_paths:
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉换行符
            filename = line.strip()
            # 解析id和序列id
            id_part = filename[:5]  # 前5位是id
            sequence_part = filename[9:13]  # T0001中的0001是序列id
            # 组合id和序列id作为唯一标识
            sequence_key = f"{id_part}_{sequence_part}"
            # 判断id是否小于10000
            if int(id_part) < 10000:
                sequence_lengths_lt_10000[sequence_key] += 1
            else:
                sequence_lengths_ge_10000[sequence_key] += 1

# 提取序列长度
lengths_lt_10000 = list(sequence_lengths_lt_10000.values())  # id < 10000的序列长度
lengths_ge_10000 = list(sequence_lengths_ge_10000.values())  # id >= 10000的序列长度

# 创建断轴图
bax = brokenaxes(xlims=((0, 401),), ylims=((0, 600), (2600, 2640)), hspace=0.1)

# 绘制id < 10000的序列长度分布
bax.hist(lengths_lt_10000, bins=range(0, 402, 20), alpha=0.5, label='ID < 10000', color='blue', edgecolor='black')
# 绘制id >= 10000的序列长度分布
bax.hist(lengths_ge_10000, bins=range(0, 402, 20), alpha=0.5, label='ID >= 10000', color='orange', edgecolor='black')

# 设置标签和标题
bax.set_xlabel('Sequence Length')
bax.set_ylabel('Frequency')
bax.set_title('Distribution of Sequence Lengths (ID < 10000 vs ID >= 10000)')
bax.legend(loc='upper right')

# 显示图形
plt.show()