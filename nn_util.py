import util
import numpy as np
# Import all the required layers
from keras.layers import Dense, Conv1D, Dropout, Input, Reshape, GlobalMaxPool1D, LSTM
# Import the concatenate layer
from keras.layers.merge import concatenate
import keras
import logging
import os
from util import load_pickle, save_pickle
from keras.utils import Sequence

MAX_WORDS = 40  # Number of words
WORD_VEC_FILE = 'Data/word_vec.pkl'


class loader(Sequence):

    def __init__(self, fname, batch_size, num_class):
        from keras.utils.np_utils import to_categorical
        self.dat = load_pickle(fname)
        self.class_map = load_pickle('Data/class_map.pkl')
        self.word_vec = load_pickle(WORD_VEC_FILE)
        self.keys = list(self.dat.keys())
        self.batch_size = batch_size
        self.max_words = MAX_WORDS
        logging.info('Pickle file loaded: {}'.format(fname))

        # Populating array
        self.dat_arr, self.cat_labels = [], []
        for label, val in self.dat.items():
            self.dat_arr += self.dat[label]
            self.cat_labels += [self.class_map[label]] * len(self.dat[label])
        self.dat_arr = np.array(self.dat_arr)
        self.lab_arr = [np.expand_dims(to_categorical(ele, num_class), 0) for ele in self.cat_labels]
        self.cat_labels = np.array(self.cat_labels)
        # Initialize array for input  data (BS*MAX_NUM_WORDS*300)
        self.input_dat_batch = np.zeros((self.batch_size, self.max_words, 300))
        self.lab_arr = np.array(self.lab_arr)
        self.shuffle_data()

    def __len__(self):
        return len(self.dat_arr) // self.batch_size

    def shuffle_data(self):
        idx_num = np.arange(len(self.dat_arr))
        np.random.shuffle(idx_num)
        self.dat_arr = self.dat_arr[idx_num]
        self.lab_arr = self.lab_arr[idx_num]
        self.cat_labels = self.cat_labels[idx_num]

    def on_epoch_end(self):
        self.shuffle_data()

    def __getitem__(self, index):
        # Create shallow copy of the input_dat
        x = self.input_dat_batch.copy()

        def produce_vector():
            for idx, text in enumerate(text_arr):
                if idx >= self.max_words:
                    return
                if text in self.word_vec:
                    x[offset, idx, :] = self.word_vec[text]

        offset = 0
        for sent_idx in range(index * self.batch_size, (index + 1) * self.batch_size):
            text_arr = self.dat_arr[sent_idx]
            produce_vector()
            offset += 1
        # Concatenate all the label arrays
        y = np.concatenate(self.lab_arr[index * self.batch_size:(index + 1) * self.batch_size])
        return x, y


def cnn_network():
    # Definng input sentence
    input_sent = Input(shape=(MAX_WORDS, 300), name='Input_Sent')

    # Defining convolutions
    c3 = Conv1D(filters=80, kernel_size=3, padding='same', name='C3')(input_sent)
    c4 = Conv1D(filters=80, kernel_size=4, padding='same', name='C4')(input_sent)
    c5 = Conv1D(filters=80, kernel_size=5, padding='same', name='C5')(input_sent)
    c6 = Conv1D(filters=80, kernel_size=6, padding='same', name='C6')(input_sent)
    c10 = Conv1D(filters=40, kernel_size=10, padding='same', name='C10')(input_sent)
    # Doing 1D max pooling
    m3 = GlobalMaxPool1D()(c3)
    m4 = GlobalMaxPool1D()(c4)
    m5 = GlobalMaxPool1D()(c5)
    m6 = GlobalMaxPool1D()(c6)
    m10 = GlobalMaxPool1D()(c10)
    # Doing concatenation
    combined_vec = concatenate([m3, m4, m5, m6, m10], name='Combining_Layers')
    # Dense layer
    fcl = Dense(units=100,activation='relu')(combined_vec)
    final_layer = Dense(units=7, activation="softmax")(fcl)

    model = keras.Model(inputs=input_sent, output=final_layer)
    logging.info('Summary of CNN model: {}'.format(model.summary()))
    return model


def lstm_network():
    # Defining input sentence
    input_sent = Input(shape=(MAX_WORDS, 300), name='Input_Sent')
    # LSTM
    out_vec = LSTM(units=200)(input_sent)
    # Dense layer
    fcl = Dense(units=100)(out_vec)
    final_layer = Dense(units=7, activation="softmax")(fcl)

    model = keras.Model(inputs=input_sent, output=final_layer)
    logging.info('Summary of LSTM model: {}'.format(model.summary()))
    return model


# Callback for printing out accuracy + losses
class custom_callback(keras.callbacks.Callback):
    def __init__(self, file_name, val_thresh, out_folder, name):
        self.fname = file_name
        self.val_thresh = val_thresh
        self.out_folder = out_folder
        self.name = name
        super(custom_callback, self).__init__()
        logging.info(
            "Instantiated custom_callback with parameters:\nWriting records:{} \n"
            "Validation threshold to save parameters:{}\nOutput folder for saving parameters:{}".format(
                file_name, val_thresh, out_folder))

    def on_epoch_end(self, epoch, logs=None):

        # Enter heading if first time
        if not os.path.exists(self.fname):
            with open(self.fname, 'w') as fid:
                fid.writelines('Epoch,Accuracy,Loss,Validation_Accuracy,Validation_Loss\n')

        # Print the data to file
        with open(self.fname, 'a') as fid:
            fid.writelines(
                '{},{},{},{},{}\n'.format(epoch, logs['acc'], logs['loss'], logs['val_acc'], logs['val_loss']))

        # Save model if greater than validation accuracy
        if logs['val_acc'] > self.val_thresh:
            fname = '{}/{}_{}_{}.pkl'.format(self.out_folder, self.name, epoch, round(logs['val_acc'], 4))
            logging.info('Saving model parameters: {}'.format(fname))
            self.model.save_weights(fname)


class nn_driver:

    def __init__(self, nn, val_thresh, track_progress, out_dir, num_class, name):
        self.nn = nn
        self.loss_fn, self.optimizer = None, None
        self.track_progress = track_progress
        self.val_thresh = val_thresh
        self.out_dir = out_dir
        self.num_class = num_class
        self.name = name

    def train(self, train_file, val_file, batch_size=32, init_epoch=0, tot_epochs=10, prev_train=None, stop_delta=0.01,
              stop_patience=5):
        logging.info("Batch Size: {}\n Initial Epoch: {}\n Total Epochs: {}".format(
            batch_size, init_epoch, tot_epochs))
        callback = custom_callback(self.track_progress, val_thresh=self.val_thresh,
                                   out_folder=self.out_dir, name=self.name)

        # Load parameters
        if prev_train:
            logging.info("Loading weights from:{}".format(prev_train))
            self.nn.load_weights(util.load_pickle(prev_train))
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=stop_delta, patience=stop_patience,
                                                      mode='max')
        self.nn.fit_generator(loader(num_class=self.num_class, fname=train_file, batch_size=batch_size),
                              validation_data=loader(num_class=self.num_class, fname=val_file, batch_size=batch_size),
                              initial_epoch=init_epoch, callbacks=[callback, early_stopper], epochs=tot_epochs)

    def testing(self, test_file, test_acc_file, param=None):
        from time import time
        from sklearn.metrics import f1_score
        from datetime import datetime
        if param is not None:
            self.nn.load_weights(param)
        test_dat = loader(batch_size=32, fname=test_file, num_class=self.num_class)
        pred_out = self.nn.predict_generator(test_dat)
        pred_label = np.argmax(pred_out, axis=1)
        ac_label = test_dat.cat_labels[:len(pred_label)]
        # Getting accuracy
        acc = [0] * self.num_class
        for idx in range(self.num_class):
            idx_arr = [ac_label == idx]
            pred, ac = (pred_label[idx_arr] == idx).astype(np.int0), (ac_label[idx_arr] == idx).astype(np.int0)
            acc[idx] = str(f1_score(ac, pred))
            # str(100 * (ac_label[idx_arr] == pred_label[idx_arr]).astype(np.int0).mean())

        if param is None:
            param = ""
        # Printing to file
        out_str = '{},{},{},{}\n'.format(self.name, param,
                                         datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'), ','.join(acc))
        with open(test_acc_file, 'a') as fid:
            fid.writelines(out_str)

        logging.info('Testing completed result added to: {}\nResult:{}'.format(test_acc_file, out_str))
