from helper import Helper
from helper import DataType
from log import Logger

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Environment:
    def __init__(self, mode):
        self.config = Helper.get_config('config')[mode]
        self.config_mode = self.config['mode']
        self.config_csv = self.config['csv']
        self.config_model = self.config['model']

        self.buffer = []

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
        data[self.config_csv['features']
             ] = data[self.config_csv['features']].astype(float)
        return data

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

            scaler = MinMaxScaler(feature_range=(
                self.config_model['norm']['mn'], self.config_model['norm']['mx']))
            scaler = scaler.fit(code_info[self.config_csv['features']])
            # print('-----------------trans---------------------')
            # print(scaler.transform(code_info[self.config_csv['features']]))
            # print('-----------------features---------------------')
            # print(code_dict[str(code)][self.config_csv['features']])
            code_info[self.config_csv['features']] = scaler.transform(
                code_info[self.config_csv['features']])

            # print(type(baseprice))
            # print(type(code_info.ix[-1, self.config_csv['baseprice']]))
            # print(type(code_info['close'].values))
            # print(code_info[self.config_csv['features']].values)

            # mx = max(code_info[self.config_csv['features']].values)
            # mn = min(code_info[self.config_csv['features']].values)

            # print(f'max : {mx}')

            # print(f'-----------code info type : {type(code_info)}')

            # code_info[self.config_csv['features']] = code_info[self.config_csv['features']].applymap(
            #     lambda x: (((x-mn)/(mx-mn)))*(10-1) + 1)

            # code_info[self.config_csv['features']] = code_info[self.config_csv['features']].applymap(
            #     lambda x: 2*x-mn)

            # print(code_info)

            # code_info = list(map(lambda x: (((x-mn)/(mx-mn)))*(10-1) + 1, code_info))

            code_dict[str(code)] = code_info
            # code_dict[str(code)][self.config_csv['features']] = code_dict[str(
            #     code)][self.config_csv['features']]/baseprice * 1e2
            code_dict[str(code)] = code_dict[str(
                code)].fillna(method='pad')

        return date_range, code_dict

    def prep_state(self, data):
        states = []
        return_hist = []

        date_range, code_dict = self.samp_data(data)

        # Helper.log_dict(code_dict,Logger.feature_norm)

        # Helper.plot_features(
        #     code_dict, self.config_csv['features'], self.config_csv['codes'], DataType.DICT)

        asset_cnt = len(self.config_csv['codes']) + 1

        money_st = np.ones(self.config_model['window'])

        t = self.config_model['window'] + 1

        # print(f'code dict:\n{code_dict}')

        while t < len(date_range) - 1:
            state = []

            y = np.ones(1)
            st_dict = dict()

            for code in self.config_csv['codes']:
                # print(f"base price: {code_dict[str(code)].ix[t-1, self.config_csv['baseprice']]}")
                y = np.vstack((y, code_dict[str(
                    code)].ix[t, self.config_csv['baseprice']]/code_dict[str(code)].ix[t-1, self.config_csv['baseprice']]))
                # print(f'st_dict: {st_dict}')
                # print(f'============code: {code}==============')
                # print(code_dict[str(code)])
                for feature in self.config_csv['features']:
                    # print(f'feature: {feature}')
                    if feature not in st_dict:
                        # print(f'not in feature: {feature}')
                        st_dict[feature] = []
                        st_dict[feature].append(np.vstack((money_st, code_dict[str(
                            code)].ix[t - self.config_model['window'] - 1:t - 1, feature])))
                        # print(code_dict[str(
                        #     code)].ix[t - self.config_model['window'] - 1:t - 1, feature])
                        # print(st_dict)
                    else:
                        # print(f'in feature: {feature}')
                        st_dict[feature][0] = np.vstack((st_dict[feature][0], code_dict[str(
                            code)].ix[t - self.config_model['window'] - 1:t - 1, feature]))
                        # print(st_dict[feature])

                # print('=================================')

            for feature in self.config_csv['features']:
                state.append(st_dict[feature])

            # print(state)

            state = np.stack(state, axis=2)
            # print(state)
            # print(f'shape: {state.shape}')
            state = np.swapaxes(state, 2, 3)
            # print(state)
            # print(f'shape: {state.shape}')
            # state = state.reshape(1, asset_cnt, self.config_model['window'], len(
            #     self.config_csv['features']))
            # print(state)
            # print(f'shape: {state.shape}')
            states.append(state)
            return_hist.append(y)
            t = t + 1
            # print('===========================================')
            # print(state)
            # print('-------------------------------------------')
            # print(np.swapaxes(state, 0, 1))
            # print('===========================================')

        # print(np.shape(return_hist))
        # sw = np.swapaxes(return_hist, 0, 1)
        # sw = np.swapaxes(sw, 1, 2)
        # print(np.shape(sw))
        # for code in range(np.shape(sw)[0]):
        #     print(f'-----------------{code}')
        #     print(sw[code][0][:])
        #     mx = max(sw[code][0][:])
        #     mn = min(sw[code][0][:])
        #     data_new = []
        #     for i in sw[code][:][0]:
        #         data_new.append(self.norm(i, mx, mn))
        # data_new = np.swapaxes(data_new, 1, 2)
        # data_new = np.swapaxes(data_new, 0, 1)
        return states, return_hist

    def norm(self, x, mx, mn):
        return (((x-mn)/(mx-mn)))*(10-1) + 1
        # if x >= 1:
        #     ret = ((x-1)/(mx-1))*(1.5-1) + 1
        # else:
        #     ret = ((x-mn)/(0.99999999-mn))*(0.99999999-0.5) + 0.5
        # return ret

    def prep_state_mod(self, data):
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
                    code)].ix[t - self.config_model['window'] - 1: t - 1, self.config_csv['features']].values
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

    def prep_data(self):
        data = self.get_data()
        data_format = self.format_data(data)
        # Helper.plot_features(
        #     data_format, self.config_csv['features'], self.config_csv['codes'], DataType.DF)
        self.states, self.return_hist = self.prep_state(data_format)
    # def prep_data(self):
    #     test = Test()
    #     self.states, self.return_hist = test.prep_data()

    def reset_buffer(self):
        self.buffer = []
