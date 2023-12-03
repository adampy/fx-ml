CONTRACTS = 10000.0
COMMISSION = 0
TRAIN_TICK_LENGTH = 40 # Per training sample, we have 40 ticks
PREDICT_TICK_LENGTH = 20 # Per prediction, we want to predict 7 ticks

# 40, 7 has profit factor of 1.9
# 40, 10 has profit factor of 2.0
# 40, 20 has profit factor of 2.0

import pandas as pd
import numpy as np

def pandas_data_loader(file_name):
    df = pd.read_csv(file_name, sep=";", header=None)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d %H%M%S")
    df.set_index("date", inplace=True)

    # drop everything apart from close
    df.drop(["open", "high", "low", "volume"], axis=1, inplace=True)

    # add 40 ticks in future as features
    for d in range(1, TRAIN_TICK_LENGTH):
        col = "%dd" % d
        df[col] = df['close'].shift(-1 * d)

    # now remove last 40 since they do not have future ticks
    df.dropna(inplace=True)

    return df

# load data
df = pandas_data_loader("eurusd/DAT_ASCII_EURUSD_M1_2022/DAT_ASCII_EURUSD_M1_2022.csv")
df.tail()

from sklearn.model_selection import train_test_split

# we want to predict 7 ticks ahead
X = df.iloc[:, :TRAIN_TICK_LENGTH - PREDICT_TICK_LENGTH]
y = df.iloc[:, TRAIN_TICK_LENGTH - PREDICT_TICK_LENGTH:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
y_test.head()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=True,
    n_jobs=16,
    random_state=None,
    verbose=1,
    warm_start=False)

regressor.fit(X_train, y_train)

import matplotlib.pyplot as plt

def evaluate_model(regressor, X_test, y_test):
    y_predicted = regressor.predict(X_test)

    LAST_PRED_COL = "p" + str(TRAIN_TICK_LENGTH - 1) + "d"                    # e.g. p39d
    ACTUAL_COL = str(TRAIN_TICK_LENGTH - 1) + "d"                             # e.g. 39d
    LAST_KNOWN_COL = str(TRAIN_TICK_LENGTH - PREDICT_TICK_LENGTH - 1) + "d"   # e.g. 32d
    
    # load in y_predicted into new pandas df
    y_predicted_df = pd.DataFrame(y_predicted, index=y_test.index, columns=y_test.columns)
    y_predicted_df.head()
    # change columns to begin with p
    y_predicted_df.columns = ["p" + str(col) for col in y_predicted_df.columns]

    # reduce y_predicted to just last columm
    y_predicted_df = y_predicted_df[[LAST_PRED_COL]]
    # reduce y_test to just last column
    y_test = y_test[[ACTUAL_COL]]
    # reduce X_test to just last column
    X_test = X_test[[LAST_KNOWN_COL]]

    # now merge x_test and y_predicted_df
    merged_df = pd.merge(X_test, y_predicted_df, left_index=True, right_index=True)

    # also merge with y_test
    merged_df = pd.merge(merged_df, y_test, left_index=True, right_index=True)

    # add col for buy or sell and pnl
    merged_df["buy"] = merged_df[LAST_PRED_COL] > merged_df[LAST_KNOWN_COL]
    merged_df["pnl"] = np.where(
        merged_df["buy"],
        (merged_df[ACTUAL_COL] - merged_df[LAST_KNOWN_COL])*CONTRACTS - COMMISSION,
        (merged_df[LAST_PRED_COL] - merged_df[ACTUAL_COL])*CONTRACTS - COMMISSION
    )

    merged_df['equity'] = merged_df['pnl'].cumsum()

    # print first 5 rows
    # merged_df[[last_known_col, last_pred_col, actual_col, "buy", "pnl", "equity"]].head()

    n_win_trades = float(merged_df[merged_df['pnl']>0.0]['pnl'].count())
    n_los_trades = float(merged_df[merged_df['pnl']<0.0]['pnl'].count())
    #print("Net Profit            : $%.2f" % merged_df.tail(1)['equity'])
    print("Number Winning Trades : %d" % n_win_trades)
    print("Number Losing Trades  : %d" % n_los_trades)
    print("Percent Profitable    : %.2f%%" % (100*n_win_trades/(n_win_trades + n_los_trades)))
    print("Avg Win Trade         : $%.3f" % merged_df[merged_df['pnl']>0.0]['pnl'].mean())
    print("Avg Los Trade         : $%.3f" % merged_df[merged_df['pnl']<0.0]['pnl'].mean())
    print("Largest Win Trade     : $%.3f" % merged_df[merged_df['pnl']>0.0]['pnl'].max())
    print("Largest Los Trade     : $%.3f" % merged_df[merged_df['pnl']<0.0]['pnl'].min())
    print("Profit Factor         : %.2f" % abs(merged_df[merged_df['pnl']>0.0]['pnl'].sum()/merged_df[merged_df['pnl']<0.0]['pnl'].sum()))

    
    # fig, ax = plt.subplots()
    # ax.hist(merged_df["pnl"], bins=20)
    # ax.set_xlabel("PnL ($)")
    # ax.set_ylabel("Frequency")
    # ax.set_title("Distribution of PnL")

    # fig, ax = plt.subplots()
    # ax.plot(merged_df["equity"])
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Equity ($)")
    # ax.set_title("Equity Curve")
    # plt.show()

evaluate_model(regressor, X_test, y_test)