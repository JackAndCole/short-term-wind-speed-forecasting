import os

import pandas as pd


def load_data(station="S.W. PIER MI"):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    filename = os.path.join(root, "%s.csv" % station)

    wind_speed = pd.read_csv(filename, index_col=[0], parse_dates=[0], squeeze=True)
    # 将时间索引转换成数字索引（sktime只支持数字索引）
    wind_speed.index = pd.RangeIndex(start=0, stop=len(wind_speed), step=1)

    return wind_speed


if __name__ == "__main__":
    ws = load_data(station="ROCK CUT MI")
    print(ws.head())
