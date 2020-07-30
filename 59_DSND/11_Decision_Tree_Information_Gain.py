from math import log
import pandas as pd

bugs = pd.read_csv('Datasets/Information-Gain-ml-bugs.csv')
print(bugs.head())

#    Species  Color  Length (mm)
# 0   Mobug  Brown         11.6
# 1   Mobug   Blue         16.3
# 2   Lobug   Blue         15.1
# 3   Lobug  Green         23.7
# 4   Lobug   Blue         18.4


def entropy(bugs):
    lob = bugs[bugs['Species'] == 'Lobug'].shape[0]
    mob = bugs[bugs['Species'] == 'Mobug'].shape[0]
    total = bugs['Species'].shape[0]

    entropy = (-1 * (lob/total) * log((lob/total), 2)) + \
        (-1 * (mob/total) * log((mob/total), 2))
    return entropy


def information_gain(series):
    sub_01 = bugs[series].copy()
    sub_02 = bugs[~series].copy()

    parent_entropy = entropy(bugs)

    weight_01 = sub_01.shape[0] / bugs.shape[0]
    weight_02 = sub_02.shape[0] / bugs.shape[0]

    return parent_entropy - (weight_01 * entropy(sub_01) + weight_02 * entropy(sub_02))


for series, string in [(bugs['Color'] == 'Brown', "bugs['Color'] == 'Brown'"),
                       (bugs['Color'] == 'Blue', "bugs['Color'] == 'Blue'"),
                       (bugs['Color'] == 'Green', "bugs['Color'] == 'Green'"),
                       (bugs['Length (mm)'] < 17, "bugs['Length (mm)'] < 17"),
                       (bugs['Length (mm)'] < 20, "bugs['Length (mm)'] < 20")]:
    info_gain = information_gain(series)
    print(f"{string} -> {info_gain:1.4f}")

# bugs['Color'] == 'Brown' -> 0.0616
# bugs['Color'] == 'Blue' -> 0.0006
# bugs['Color'] == 'Green' -> 0.0428
# bugs['Length (mm)'] < 17 -> 0.1126 <- Most Information Gain
# bugs['Length (mm)'] < 20 -> 0.1007    