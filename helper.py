from log import Log
from log import Logger

import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from enum import Enum


class DataType(Enum):
    DF = 1,
    DICT = 2


class Helper:

    log = Log()
    dict_logger = {
        Logger.test: log.test,
        Logger.feature_norm: log.feature_norm,
        Logger.kernel: log.kernel
    }
    pd.options.display.max_colwidth = 100
    pd.set_option("display.colheader_justify", "left")

    @staticmethod
    def get_config(config_name):
        """Read json config

        Arguments:
            config_name {string} -- json config name

        Returns:
            dict -- json config
        """
        with open(f'{config_name}.json') as f:
            return json.load(f)

    @staticmethod
    def plot_features(data, features, codes, data_type):
        plt.autoscale(False)
        fig, ax = plt.subplots(len(codes), 1)
        if len(codes) == 1:
            ax = [ax]
        for code in range(len(codes)):
            for feature in features:
                if data_type == DataType.DF:
                    filename = 'features.png'
                    ax[code].plot(
                        data.loc[data['code'] == codes[code], feature], label=f'{feature}')
                elif data_type == DataType.DICT:
                    filename = 'features_norm.png'
                    ax[code].plot(data[str(codes[code])]
                                  [feature], label=f'{feature}')
            ax[code].set_title(codes[code])
            ax[code].legend()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(filename)
        plt.show()

    @staticmethod
    def plot_state(state):
        state = np.swapaxes(state, 2, 3)
        code_qty = state.shape[1]
        feature_qty = state.shape[2]

        fig, ax = plt.subplots(code_qty, 1)
        for code in range(code_qty):
            for feature in range(feature_qty):
                #     print(state[0][code][feature])
                #     print('-------------')
                # print("=============================")
                ax[code].plot(state[0][code][feature],
                              label=f'feature - {feature}')
            ax[code].set_title(f'code - {code}')
            ax[code].legend()

        # # print(state.shape)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    @staticmethod
    def plot_prediction(state, action):
        state = np.swapaxes(state, 2, 3)
        code_qty = state.shape[1]
        feature_qty = state.shape[2]

        fig, ax = plt.subplots(code_qty + 1, 1)
        for code in range(code_qty):
            for feature in range(feature_qty):
                #     print(state[0][code][feature])
                #     print('-------------')
                # print("=============================")
                ax[code].plot(state[0][code][feature],
                              label=f'feature - {feature}')
            ax[code].set_title(f'code - {code}')
            ax[code].legend()

        print(action)
        ax[code_qty].plot(action[0], label=f'action')
        ax[code_qty].set_title(f'code - {code}')
        ax[code_qty].legend()
        # # print(state.shape)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    @staticmethod
    def plot_action(actions):
        # plot feature and actions
        pass

    @staticmethod
    def log_dict(dict, logger):
        msg = ''
        for key in dict.keys():
            msg += f'{key}\n{dict[key]}\n'

        if logger == Logger.feature_norm:
            Helper.log.feature_norm.info(msg)

    @staticmethod
    def log_msg(msg, logger):
        Helper.dict_logger[logger].info(msg)

    @staticmethod
    def log_list(items, logger):
        # msg = ''
        # msg = ''.join(f"name: {item.name}\t\tshape: {item.shape}\tdtype: {''.join(f'{dtype} ' for dtype in item.dtype)}\n" for item in items)
        dfObj = pd.DataFrame(columns=['Name', 'Shape', 'Dtype', 'Trainable'])
        # listOfSeries = [pd.Series(['Raju', 21, 'Bangalore', 'India'], index=dfObj.columns )]
        # dfObj.loc[i] = ['name' + str(i)] + list(randint(10, size=2))
        # pd.DataFrame({"a":[1, 2, 3],
        #             "b":[5, 6, 7],
        #             "c":[1, 5, 4]})
        dfObj = dfObj.append([pd.DataFrame({'Name': [item.name], 'Shape':[str(item.shape)], 'Dtype':[
                             str(item.dtype)], 'Trainable':[str(item.trainable)]}) for item in items], ignore_index=True)
        # msg = ''.join(
        #     f"name: {item.name}\t\tshape: {item.shape}\tdtype: {item.dtype}\ttrainable: {item.trainable}\n" for item in items)
        # msg = [f'{item.shape}\n' for item in items]
        # msg = pd.DataFrame(items)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(dfObj)
        # print(pd.options.display.max_colwidth)
        # print(dfObj.to_string())
        # dfObj = dfObj.stack().str.lstrip().unstack()
        # print(pd.describe_option())
        # dfObj.style.set_properties(**{'text-align': 'left'})
        # print(dfObj)
        Helper.dict_logger[logger].info(dfObj)

    @staticmethod
    def w_csv(content, name):
        workbook1 = load_workbook('history.xlsx')
        writer = pd.ExcelWriter('history.xlsx', engine='openpyxl')
        writer.book = workbook1
        pd.DataFrame(content).to_excel(writer, sheet_name=name)
        writer.save()
        writer.close()
        # with pd.ExcelWriter('history.xlsx') as writer:
        #     pd.DataFrame({'weight': w}).to_excel(
        #         writer, sheet_name=str(profit))
        #     pd.DataFrame({'weight': w}).to_excel(
        #         writer, sheet_name=str(profit+1))
        #     writer.save()
#             pd.Series(w).to_csv()
# df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
# ...                    'mask': ['red', 'purple'],
# ...                    'weapon': ['sai', 'bo staff']})
# ...     df1.to_excel(writer, sheet_name='Sheet_name_1')
