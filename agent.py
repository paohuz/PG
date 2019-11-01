import tensorflow as tf
import tflearn
import numpy as np
import pprint
import os

from helper import Helper
from log import Logger


class Agent:
    def __init__(self, config):
        self.count = 0

        self.config = config
        self.config_csv = self.config['csv']
        self.config_model = self.config['model']

        self.asset_cnt = len(self.config_csv['codes']) + 1

        self.session = tf.Session()  # tf.InteractiveSession()  # tf.Session()
        self.global_step = tf.Variable(0, trainable=False)
        self.build_model()

        self.future_price = tf.placeholder(tf.float32, [None]+[self.asset_cnt])
        self.pv_vector = tf.reduce_sum(
            self.predicted_action*self.future_price, reduction_indices=[1])*self.pc()
        self.profit = tf.reduce_prod(self.pv_vector, name='p1')
        self.loss = -tf.reduce_mean(tf.log(self.pv_vector))
        self.optimize = tf.train.AdamOptimizer(self.config_model['learning_rate'], epsilon=1.0, name='a2').minimize(
            self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.profit_var = tf.Variable(0., name='profittrain')
        self.eval = tf.Variable(0., name='profiteval')
        self.writer = tf.summary.FileWriter('./summary', self.session.graph)

        tf.summary.scalar("profit train", self.profit_var)
        tf.summary.scalar("profit eval", self.eval)
        # self.summary_vars = [self.profit_var,self.eval]
        self.write_op = tf.summary.merge_all()

        self.checkpoint = tf.train.get_checkpoint_state(
            self.config_model['checkpoint'])

        if self.config_model["continue"]:
            if os.path.isfile('./model/checkpoint'):
                tf.reset_default_graph()
                self.saver.restore(
                    self.session, self.checkpoint.model_checkpoint_path)
            else:
                print("Not found checkpoint")
                self.session.run(tf.global_variables_initializer())
        else:
            self.session.run(tf.global_variables_initializer())

        # self.saver.restore(self.session, self.checkpoint.model_checkpoint_path)

    def pc(self):
        # print(tf.abs(self.predicted_action[:,1:]-self.previous_action[:,1:]))
        return 1-tf.reduce_sum(tf.abs(self.predicted_action[:, 1:]-self.previous_action[:, 1:]), axis=1)*self.config_model['cost']

    def build_model(self):

        # with tf.name_scope()
        self.state = tf.placeholder(tf.float32, shape=[
                                    None, self.asset_cnt, self.config_model['window'], len(self.config_csv['features'])], name='input_st')
        # print("state shape: " + str(self.state.get_shape()))
        self.network = tf.layers.conv2d(
            inputs=self.state,
            filters=2,
            kernel_size=[1, 2],
            strides=[1, 1],
            padding='valid',
            activation='relu', name='conv1')
        # self.network = tflearn.layers.conv_2d(
        #     self.state, 2, [1, 2], [1, 1], 'valid', 'relu', name='conv1')
        # self.network = tflearn.layers.conv_2d(
        #     self.state, 2, [1, 2], [1, 1, 1, 1], 'valid', 'relu', name='conv1')
        # self.network = tflearn.layers.conv_2d(
        #     self.state, 3, [1, 3], [1, 1, 1, 1], 'valid', 'relu', name='conv1')
        # self.network = tf.layers.flatten(self.network)
        # self.w_init = tf.random_uniform_initializer(-0.005, 0.005)
        # self.predicted_action = tf.layers.dense(
        #     self.network, self.asset_cnt, activation=tf.nn.softmax, kernel_initializer=self.w_init)

        # self.state = tf.placeholder(tf.float32, shape=[
        #                             None]+[self.asset_cnt]+[self.config_model['window']] + [len(self.config_csv['features'])])

        # print("state shape: " + str(self.state.get_shape()))
        # self.network = tflearn.layers.conv_2d(
        #     self.state, 2, [1, 2], [1, 1, 1, 1], 'valid', 'relu')
        # width = self.network.get_shape()[2]

        # self.network = tf.layers.conv2d(
        #     inputs=self.network,
        #     filters=48,
        #     kernel_size=[1, 10],
        #     strides=[1, 1],
        #     padding='valid',
        #     activation='relu', name='conv1_5')

        regularizer = tf.contrib.layers.l2_regularizer(scale=5e-9)

        self.network = tf.layers.conv2d(
            inputs=self.network,
            filters=48,
            kernel_size=[1, self.network.get_shape()[2]],
            strides=[1, 1],
            padding='valid',
            activation='relu',
            activity_regularizer=regularizer, name='conv2')

        # self.network = tflearn.layers.conv_2d(self.network, 48,
        #                                       [1, self.network.get_shape()[2]],
        #                                       [1, 1],
        #                                       "valid",
        #                                       'relu',
        #                                       regularizer="L2",
        #                                       weight_decay=5e-9)
        # print("network shape: " + str(self.network.get_shape()))

        self.previous_action = tf.placeholder(
            tf.float32, shape=[None, self.asset_cnt], name='prev_act')
        self.network = tf.concat([self.network, tf.reshape(
            self.previous_action, [-1, self.asset_cnt, 1, 1])], axis=3, name='concat')

        # print(f'd0: {self.network.get_shape()[0]}')
        # print(f'd1: {self.network.get_shape()[1]}')
        # print(f'd2: {self.network.get_shape()[2]}')
        # print(f'd3: {self.network.get_shape()[3]}')

        regularizer1 = tf.contrib.layers.l2_regularizer(
            scale=5e-9)

        self.network = tf.layers.conv2d(
            inputs=self.network,
            filters=1,
            kernel_size=[1, self.network.get_shape()[2]],
            strides=[1, 1],
            padding='valid',
            activation='relu',
            activity_regularizer=regularizer1, name='conv3')

        # self.network = tflearn.layers.conv_2d(self.network, 1,
        #                                       [1, self.network.get_shape()[2]],
        #                                       [1, 1],
        #                                       "valid",
        #                                       'relu',
        #                                       regularizer="L2",
        #                                       weight_decay=5e-9)

        # print(f'd0: {self.network.get_shape()[0]}')
        # print(f'd1: {self.network.get_shape()[1]}')
        # print(f'd2: {self.network.get_shape()[2]}')
        # print(f'd3: {self.network.get_shape()[3]}')
        self.network = tf.layers.flatten(self.network, name='flatten')
        # self.w_init = tf.random_uniform_initializer(-0.005, 0.005)

        self.w_init = tf.random_uniform_initializer(-0.05, 0.05)

        self.predicted_action = tf.layers.dense(
            self.network, self.asset_cnt, activation=tf.nn.softmax, kernel_initializer=self.w_init, name='pred_act')
        # with tf.variable_scope('a1', reuse=True):
        #     w = tf.get_variable('kernel')
        #     b = tf.get_variable('bias')
        #     weights = tf.get_default_graph().get_tensor_by_name('a1/kernel:0')
        #     print(f'*****************{w}')
        #     print(f'*****************{b}')

    def predict(self, state, action, price):
        aa = self.session.run(self.predicted_action, feed_dict={
                              self.state: state, self.previous_action: action})

        # if self.config_model["plot"]:
        #     Helper.plot_prediction(state, aa)
        # else:
        #     if self.count <= 2:
        #         Helper.plot_prediction(state, aa)
        #     self.count += 1

        if self.count <= 2:
            print(action)
            self.count += 1

        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1'):
        msg = ''
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            msg += f'{i.name} -\nshape - {i.eval(self.session).shape}\n{i.eval(self.session)}\n======================='
        # print(f'{i.name} -\n{i.eval(self.session)}')

        # Helper.log_list(tf.trainable_variables(), Logger.kernel)
        # Helper.log_list(tf.all_variables(), Logger.kernel)
        # Helper.log_msg(msg, Logger.kernel)
        # node = ''.join(f'{n.name}\t{type(n)}\n' for n in tf.get_default_graph().as_graph_def().node)
        # node = ''.join(
        #     f"{n.name}\t{[int(a) for a in n.attr['strides'].list.i]}\n" for n in tf.get_default_graph().as_graph_def().node)
        # # node = [f'{n.name}\n' for n in tf.get_default_graph().as_graph_def().node]
        # print(node)
        # var = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]
        # var = [v for v in tf.trainable_variables()]
        # print(var)

        # [print(v) for v in tf.trainable_variables()]

        # with tf.variable_scope('a1', reuse=True):
        #     w = tf.get_variable('kernel')
        #     b = tf.get_variable('bias')

        #     print(f'*****************{w.eval(self.session)}')
        #     print(f'*****************{b.eval(self.session)}')

        # with tf.variable_scope('a1', reuse=True):
        #     w = tf.get_variable('kernel')
        #     b = tf.get_variable('bias')
        #     weights = tf.get_default_graph().get_tensor_by_name('a1/kernel:0')
        #     print(f'*****************{w.eval(self.session)}')
        #     print(f'*****************{b.eval(self.session)}')

        # price = self.price_history[self.t]
        # price = price+np.stack(np.random.normal(0, 0.00002,
        #                                         (1, len(price))), axis=1)
        # aa = aa / (np.dot(aa, price) + 10e-8)

        return aa

    def extract_buffer(self, buffer):
        state = [data[0][0] for data in buffer]
        price = [data[1] for data in buffer]
        action = [data[2] for data in buffer]
        previous_action = [data[3] for data in buffer]
        return state, price, action, previous_action

    def write_summary(self, tprofit, eprofit):
        # summary_str = self.sesson.run(self.summary_ops, feed_dict={
        #     self.profit_var: tprofit,
        #     self.eval: eprofit
        # })
        # self.summary_writer.add_summary(summary_str, self.sesson.run(self.global_step))
        summary = self.session.run(
            self.write_op, {self.profit_var: tprofit, self.eval: eprofit})
        self.writer.add_summary(summary, self.session.run(self.global_step))
        self.saver.save(self.session, './model/'+"mymodel",
                        global_step=self.global_step)

    def train(self, buffer, num):
        state, price, action, previous_action = self.extract_buffer(buffer)
        # print(f'sec opt: {self.optimize}')
        # print(np.reshape(action, (-1, self.asset_cnt)))

        profit, p = self.session.run([self.profit, self.optimize], feed_dict={self.state: state, self.predicted_action: np.reshape(
            action, (-1, self.asset_cnt)), self.future_price: np.reshape(price, (-1, self.asset_cnt)), self.previous_action: np.reshape(previous_action, (-1, self.asset_cnt))})
        # with tf.variable_scope('a1', reuse=True):
        #     w = tf.get_variable()
        # msg = ''
        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     msg += f'{i.name} -\nshape - {i.eval(self.session).shape}\n{i.eval(self.session)}\n======================='
        # print(msg)
        # print(f'thrid opt: {p}')
        # print(f'thrid opt: {self.optimize}')

        print(profit)
        self.write_summary(profit, np.nan)
        # self.summary_writer.add_summary(summary_str, self.sesson.run(self.global_step))
        # self.writer.flush()
        self.saver.save(self.session, './model/'+"mymodel",
                        global_step=self.global_step)

        # print(p)
        if self.config_model["save_hist"]:
            # print(np.shape(state))
            # dict_csv = {'weight': action}
            dict_csv = dict()
            [dict_csv.update({f'action-{code}': np.array(action)[:, 0, code]})
             for code in range(self.asset_cnt)]
            [dict_csv.update({f'price-{code}': np.array(price)[:, code, 0]})
             for code in range(self.asset_cnt)]

            [[dict_csv.update({f'code-{code} feature-{feature}': [str(np.array(state)[i, code, :, feature]) for i in range(len(state))]}) for feature in range(len(self.config_csv['features']))]
             for code in range(self.asset_cnt)]

            Helper.w_csv(dict_csv, str(f'tr-{profit}'))
            # Helper.w_csv({'weight': action, 'price': price,
            #               'state': state}, str(profit))

        # self.count = 0

    def test(self, buffer):
        state, price, action, previous_action = self.extract_buffer(buffer)
        # print(f'sec opt: {self.optimize}')
        # print(np.reshape(action, (-1, self.asset_cnt)))

        profit = self.session.run(self.profit, feed_dict={self.state: state, self.predicted_action: np.reshape(
            action, (-1, self.asset_cnt)), self.future_price: np.reshape(price, (-1, self.asset_cnt)), self.previous_action: np.reshape(previous_action, (-1, self.asset_cnt))})

        print(profit)
        self.write_summary(np.nan, profit)

        if self.config_model["save_hist"]:

            dict_csv = dict()
            [dict_csv.update({f'action-{code}': np.array(action)[:, 0, code]})
                for code in range(self.asset_cnt)]
            [dict_csv.update({f'price-{code}': np.array(price)[:, code, 0]})
                for code in range(self.asset_cnt)]

            [[dict_csv.update({f'code-{code} feature-{feature}': [str(np.array(state)[i, code, :, feature]) for i in range(len(state))]}) for feature in range(len(self.config_csv['features']))]
                for code in range(self.asset_cnt)]

            Helper.w_csv(dict_csv, str(f'te-{profit}'))

    def close_sess(self):
        tf.reset_default_graph()
        tflearn.config.init_training_mode()
        self.session.close()
