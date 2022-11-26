# 导入pandas、numpy库
import numpy as np
import pandas as pd
import time


# 根据配置文件构造泛化树
class Tree:
    def __init__(self, confile):
        self.confile = confile
        self.nodes = {}
        self.level = 0
        self.build_tree()

    def build_tree(self):
        tree_df = pd.read_csv(self.confile, header=None)
        self.level = tree_df.shape[1] - 1
        for row in tree_df.itertuples():
            pre = None
            h = len(row) - 2
            for col in row[-1:0:-1]:
                self.nodes[str(col)] = (pre, h)
                pre = col
                h -= 1


# 数据集中每列元素名
names = (
    'age',
    'workclass',
    'fnlwgt',  # 一个人代表的权重 指一个人可以代表多少人
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
)

# 准标识符
qi_names = (
    'age',
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'race',
    'sex',
    'native-country',
)


# 找到最多不同值的准标识符
def max_qi(df):
    return df[list(qi_names)].nunique().idxmax()


# 统计 未K匿名的数据条数 和 数据重复频率
def freq(df, k):
    group_df = df[list(qi_names)].groupby(list(qi_names), as_index=False).size()
    return group_df[group_df['size'] < k]['size'].sum(), pd.merge(df, group_df, on=list(qi_names))


# 根据泛化树获得泛化后值
def generalize(value, qi):
    return gen_tree[qi].nodes[str(value)][0]


def cal_information_loss(gen_level, tree, qi_names):
    """
    根据泛化层级计算信息损失度
    :param gen_level: 当前泛化等级
    :param tree: 泛化树
    :param qi_names: 准标识符
    :return: 信息损失度
    """
    return sum([gen_level[i]/tree[i].level for i in qi_names])/len(qi_names)


# 读取数据集
df = pd.read_csv("./adult.csv", names=names)
df.insert(0, 'index', range(len(df)), allow_duplicates=False)
# 丢弃所有带？的行数据
df = df.replace('?', np.NAN).dropna()

# 为每个准标识符生成泛化树
gen_tree = {}
# 泛化级别
gen_level = {}
conf_prefix = './conf/'
conf_suffix = '_hierarchy.csv'
for qi_name in qi_names:
    gen_tree[qi_name] = Tree(conf_prefix + qi_name + conf_suffix)
    gen_level[qi_name] = 0

start = time.time()
rounds = 0
# 设置 k 值
k = 5

while True:
    count, freq_df = freq(df, k)
    if count > k and rounds < 7:
        mqi = max_qi(df)
        df[mqi] = df[mqi].apply(generalize, qi=mqi)
        gen_level[mqi] += 1
        rounds += 1
    else:
        df = freq_df[freq_df['size'] >= k]
        info_loss = cal_information_loss(gen_level, gen_tree, qi_names)
        # 显示泛化层级
        print("------泛化层级------")
        for qi_name in qi_names:
            print(f"{qi_name}:\t{gen_level[qi_name]}/{gen_tree[qi_name].level}")
        print("-------------------")
        print(f"耗时{time.time() - start}s")
        print(f"泛化{rounds}轮 未K匿名数据总数(丢弃数据): {count}, 信息损失率: {info_loss}")
        break

# 保存数据至excel
df.to_excel('./result.xlsx', index=False)
