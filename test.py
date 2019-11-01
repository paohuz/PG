from helper import Helper
from helper import DataType
from log import Logger

import pandas as pd
import numpy as np

import time
from environment import Environment


class Test:
    def __init__(self):
        self.config = Helper.get_config('config')
        self.config_csv = self.config['csv']
        self.config_model = self.config['model']

        self.buffer = []

    def samp_data(self, data):
        """sampling data for whole period

        Arguments:
            data {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        data = self.format_data(data)
        date_range = pd.date_range(
            self.config_csv['datestart'], self.config_csv['dateend'], freq=self.config_csv['freq'])
        code_dict = dict()
        for code in self.config_csv['codes']:
            code_info = data[data.code == code].reindex(
                date_range).sort_index()
            code_info = code_info.resample(self.config_csv['freq']).mean()
            code_info = code_info.fillna(method='pad')
            baseprice = code_info.ix[-1, self.config_csv['baseprice']]

            code_dict[str(code)] = code_info
            code_dict[str(code)][self.config_csv['features']] = code_dict[str(
                code)][self.config_csv['features']]/baseprice
            code_dict[str(code)] = code_dict[str(
                code)].fillna(method='pad')

        return date_range, code_dict

    def prep_state(self, data):
        states = []
        return_hist = []

        date_range, code_dict = self.samp_data(data)
        # Helper.plot_features(
        #     code_dict, self.config_csv['features'], self.config_csv['codes'], DataType.DICT)

        asset_cnt = len(self.config_csv['codes']) + 1

        money_st = np.ones(self.config_model['window'])

        t = self.config_model['window'] + 1

        while t < len(date_range) - 1:
            state = []
            y = np.ones(1)
            st_dict = dict()

            state.append(
                np.ones([self.config_model['window'], len(self.config_csv['features'])]))

            for code in self.config_csv['codes']:

                y = np.vstack((y, code_dict[str(
                    code)].ix[t, self.config_csv['baseprice']]/code_dict[str(code)].ix[t-1, self.config_csv['baseprice']]))

                val = code_dict[str(
                    code)].ix[t - self.config_model['window'] - 1:t - 1, self.config_csv['features']].values
                state.append(val)
                # np.vstack((state, val))
                # state = code_dict[str(
                #     code)].ix[t - self.config_model['window'] - 1:t - 1, self.config_csv['features']]
                # states.append(code_dict[str(
                #     code)].ix[t - self.config_model['window'] - 1:t - 1, self.config_csv['features']].values)
                # print(val)
            states.append([state])
            # print(state)
            # print('=======')
            # for feature in self.config_csv['features']:

            #     if feature not in st_dict:

            #         st_dict[feature] = []
            #         st_dict[feature].append(np.vstack((money_st, code_dict[str(
            #             code)].ix[t - self.config_model['window'] - 1:t - 1, feature])))

            #     else:
            #         st_dict[feature][0] = np.vstack((st_dict[feature][0], code_dict[str( code)].ix[t - self.config_model['window'] - 1:t - 1, feature]))

            # for feature in self.config_csv['features']:
            #     state.append(st_dict[feature])

            # state = np.stack(state, axis=2)
            # state = np.swapaxes(state, 2, 3)
            # states.append(state)
            return_hist.append(y)
            t = t + 1
        # print(states)
        # Helper.log_msg(states, Logger.test)
        return states, return_hist

    def get_data(self):
        """Get data from CSV

        Returns:
            dataframe -- data read from CSV
        """
        def dateparse(x): return pd.datetime.strptime(
            x, self.config_csv['dateformat'])
        return pd.read_csv(f"{self.config_csv['path']}{self.config_csv['filename']}", index_col=0, parse_dates=True, date_parser=dateparse, dtype=object).sort_index()

    def format_data(self, data):
        """Format features to float

        Arguments:
            data {dataframe} --

        Returns:
            data -- formatted data
        """
        # data = self.get_data()
        data = data[self.config_csv['datestart']:self.config_csv['dateend']]
        data[self.config_csv['features']] = data[self.config_csv['features']].astype(float)
        return data

    def prep_data(self):
        data = self.get_data()
        data_format = self.format_data(data)
        return self.prep_state(data_format)


def main():

    start_time = time.time()
    # env = Environment()
    # env.prep_data()

    test = Test()
    test.prep_data()


#     Helper.log_msg(env.states, Logger.feature_norm)

#     Helper.log_msg(
#         f'{time.time() - start_time}\n===================', Logger.feature_norm)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
