import pandas
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

new_valid = ['4-3', '5-7', '3-3', '2-3', '9-3', '10-5', '3-2', '3-7']
new_train = ['9-1', '2-2', '4-7', '9-7', '1-8', '4-6', '2-7', '8-4', '7-2', '10-3', '2-4', '7-4', '3-4',
             '5-4', '8-7', '7-7', '4-4', '1-3', '7-1', '5-2', '6-4', '9-8', '9-5', '6-3', '10-8', '1-6', '3-5',
             '2-6', '3-8', '3-6', '4-8', '7-8', '5-1', '2-8', '8-2', '1-5', '7-3', '10-2', '5-5', '9-2', '5-6', '1-7',
             '8-3', '4-1', '4-2', '1-4', '6-5', ]
new_test = ['9-6', '4-5', '1-2', '10-7', '1-1', '6-1', '6-6', '9-4', '10-4', '8-5', '5-3', '10-6',
            '2-5', '6-2', '3-1', '8-8', '8-1', '8-6', '7-6', '6-8', '7-5', '10-1']


def extract_features(data):
    q_discharge = []
    charge_time = []
    # discharge_timeidx = 0
    for cellname, cellitem in data.items():
        for cycleidx, cycledata in cellitem['data'].items():
            q_discharge.append([])
            discharge_started = False
            for timeidx in range(len(cycledata)):
                if 'discharge' in cycledata['Status'][timeidx]:
                    q_discharge[-1].append(cycledata['Capacity (mAh)'][timeidx])
                    if not discharge_started:
                        charge_time.append(cycledata['Time (s)'][timeidx])
                        discharge_started = True
                # else:
                #     print(cycledata['Status'][timeidx])
        q_discharge_10 = np.array(q_discharge[9])  #
        q_discharge_100 = np.array(q_discharge[99])
        feature_num = min(len(q_discharge_10), len(q_discharge_100))
        delta_Q = q_discharge_100[:feature_num] - q_discharge_10[:feature_num]
        variance_dQ_100_10 = np.log(np.abs(np.var(delta_Q)))
        minimum_dQ_100_10 = np.log10(np.abs(np.min(delta_Q)))
        skewness_dQ_100_10 = np.log(np.abs(skew(delta_Q)))
        kurtosis_dQ_100_10 = np.log(np.abs(kurtosis(delta_Q)))

        discharge_capacity_2 = cellitem['dq'][2]  # this index begins from 1

        q = np.array([cellitem['dq'][i] for i in range(2, 101)]).reshape(-1,
                                                                         1)  # discharge capacities; q.shape = (99, 1);
        # print(q)
        X = np.array([i for i in range(1, 100)]).reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)
        interception_lin_fit_2_100 = linear_regressor_2_100.intercept_.item()
        slope_lin_fit_2_100 = linear_regressor_2_100.coef_[0].item()
        diff_discharge_capacity_max_2 = np.max(q) - discharge_capacity_2

        mean_charge_time_2_6 = np.mean(charge_time[1:6])
        return [minimum_dQ_100_10, variance_dQ_100_10, skewness_dQ_100_10, kurtosis_dQ_100_10,
                slope_lin_fit_2_100, interception_lin_fit_2_100, discharge_capacity_2,
                diff_discharge_capacity_max_2, mean_charge_time_2_6, cellitem['rul'][1]]


def build_features(cell_idxs, save_csv_path):
    features_df = {
        "cell_key": [],
        "minimum_dQ_100_10": [],
        "variance_dQ_100_10": [],
        "skewness_dQ_100_10": [],
        "kurtosis_dQ_100_10": [],
        "slope_lin_fit_2_100": [],
        "intercept_lin_fit_2_100": [],
        "discharge_capacity_2": [],
        "diff_discharge_capacity_max_2": [],
        "mean_charge_time_2_6": [],
        "minimum_IR_2_100": [],
        "diff_IR_100_2": [],
        "minimum_dQ_5_4": [],
        "variance_dQ_5_4": [],
        "cycle_life": [],
        "cycle_550_clf": []
    }
    for cellidx in cell_idxs:
        print('extracting features from' + cellidx)
        datapth = './data/our_data/' + cellidx + '.pkl'
        data = pickle.load(open(datapth, 'rb'))
        features = extract_features(data)
        features_df['cell_key'].append(cellidx)
        features_df['minimum_dQ_100_10'].append(features[0])
        features_df['variance_dQ_100_10'].append(features[1])
        features_df['skewness_dQ_100_10'].append(features[2])
        features_df['kurtosis_dQ_100_10'].append(features[3])
        features_df['slope_lin_fit_2_100'].append(features[4])
        features_df['intercept_lin_fit_2_100'].append(features[5])
        features_df['discharge_capacity_2'].append(features[6])
        features_df['diff_discharge_capacity_max_2'].append(features[7])
        features_df['mean_charge_time_2_6'].append(features[8])
        features_df['minimum_IR_2_100'].append(9999)
        features_df['diff_IR_100_2'].append(9999)
        features_df['minimum_dQ_5_4'].append(9999)
        features_df['variance_dQ_5_4'].append(9999)
        features_df['cycle_550_clf'].append(9999)
        features_df['cycle_life'].append(features[9])
    features_df = pandas.DataFrame(features_df)
    features_df.to_csv(save_csv_path, index=False)


build_features(['4-3'], 'try.csv')
build_features(new_valid + new_train, 'train.csv')
build_features(new_test, 'test.csv')

