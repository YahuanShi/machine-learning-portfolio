import scorecardpy as sc
import pandas as pd

# 示例数据
dat = sc.germancredit()
print(dat.head())

# 自动分箱 + WOE 计算
bins = sc.woebin(dat, y='creditability')
woe_data = sc.woebin_ply(dat, bins)

# 查看一个变量的分箱结果
bins['age in years']

