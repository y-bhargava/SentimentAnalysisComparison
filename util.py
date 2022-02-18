### This file will contain the utility functions: ###
import matplotlib.pyplot as plt
import ujson as json
import numpy as np
import logging
from glob import glob
from tqdm import tqdm
import pickle
from collections import Counter
from random import shuffle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

ALLCLASS = {'AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork',
            'SearchScreeningEvent'}


def load_pickle(fname):
    if not os.path.exists(fname):
        logging.warning("Pickle file not present" + fname)
        exit()
    with open(fname, mode='rb') as fid:
        return pickle.load(fid)


def save_pickle(fname, dat):
    with open(fname, mode='wb') as fid:
        pickle.dump(dat, fid)


def get_file_array(dir_name, matcher):
    file_arr = glob(pathname="{}/{}".format(dir_name, matcher))
    return file_arr


def load_json(fname):
    if not os.path.exists(fname):
        logging.warning("Json file not present" + fname)
        exit()
    with open(fname, mode='rb') as fid:
        return json.load(fid)


def save_json(fname, dat):
    with open(fname, mode='wb') as fid:
        json.dump(dat, fid)


def get_all_word(fname, main_counter):
    """
    Gets all the words present in the json
    :param fname: JSON file to open
    :param main_counter: Counter object to increment
    """
    curr_dict = load_json(fname=fname)
    key_list = list(curr_dict.keys())
    assert (len(key_list) == 1)
    label = key_list[0]
    for dat in curr_dict[label]:
        curr_sample = dat["data"]
        for i in curr_sample:
            for word in i["text"].split():
                main_counter[word.lower()] += 1


def make_word_vec_pkl():
    """
    Makes pickle which stores all the word vectors by using spacy
    :return:
    """
    import spacy
    logging.info('Loading Spacy....')
    nlp = spacy.load('en_core_web_lg')
    logging.info('Loading spacy completed')
    main_counter = Counter()

    for file in get_file_array(dir_name="Data", matcher="*.json"):
        get_all_word(fname=file, main_counter=main_counter)
    unique_words = [word for word in main_counter.keys()]
    vector_dict = {}
    for word_idx in tqdm(range(len(unique_words))):
        word = unique_words[word_idx]
        try:
            word_obj = nlp(word)
            if word_obj.has_vector:
                vector_dict[word] = word_obj.vector
        except UnicodeEncodeError:
            logging.warning('Unicode encoding error encountered for: {}'.format(word))
    save_pickle('word_vec.pkl', vector_dict)


def make_combined_data(dir_name, regex_exp):
    master_dict = {}
    f_arr = get_file_array(dir_name=dir_name, matcher=regex_exp)
    logging.info('Total number of files: {}'.format(len(f_arr)))
    for fname in f_arr:
        curr_dict = load_json(fname)
        key_list = list(curr_dict.keys())
        assert (len(key_list) == 1)
        label = key_list[0]
        if label not in master_dict:
            master_dict[label] = []
        logging.info('Using file: {}\nLabel: {}\nTotal Samples: {}\n'.format(fname, label, len(curr_dict[label])))
        for dat in curr_dict[label]:
            # Processing for each sample
            curr_sample, words_arr, idx = dat["data"], [], 0
            # Processing each sample
            for sec in curr_sample:
                # Processing each word
                for word in sec['text'].split():
                    words_arr.append(word)
                    idx += 1
                    if idx > 40:
                        logging.warning('Encountered case with more than 40 words')
                        break
            master_dict[label].append(words_arr)
    save_pickle(dat=master_dict, fname='Data/combined_dat.pkl')


def get_sentvector(fname):
    """
    Makes sentence vector
    :param fname:
    :return:
    """
    # Loading the required pickles
    class_map, ent_map = load_pickle(fname='class_map.pkl'), load_pickle(fname='ent_map.pkl')
    vector_dict = load_pickle(fname='word_vec.pkl')

    master_arr = []
    curr_dict = load_json(fname)
    key_list = list(curr_dict.keys())
    zero_arr = np.zeros((40, 300))
    assert (len(key_list) == 1)
    label = key_list[0]
    label_idx = class_map[label]
    logging.info('Using file: {}\nLabel: {}\nTotal Samples: {}\n'.format(fname, label, len(curr_dict[label])))
    for dat in curr_dict[label]:
        curr_sample = dat["data"]
        words_vec, idx = zero_arr.copy(), 0
        for sec in curr_sample:
            for word in sec['text'].split():
                if word in vector_dict:
                    words_vec[idx, :] = vector_dict[word]
                    idx += 1
                    if idx > 40:
                        logging.warning('Encountered case with more than 40 words')
                        break
        master_arr.append({'words_vec': words_vec})
    save_pickle(fname=fname[:-5] + '.pkl', dat={'data': master_arr, 'label': label_idx})


def combine_data(folder, train=False):
    """
    Combines all the .pkl files in the folder run get_sent_vector before
    If train is False saved as combined else as train_dat and val_dat (dependent on the ratio of split (train))
    :param folder: Folder with .pkl files
    :param train: Specify split if preparing train and validation split
    """
    from keras.utils.np_utils import to_categorical

    # Get all data in centralized array
    words_vec_arr, ent_arr, lab_arr = [], [], []
    for f_name in get_file_array(folder, '*.pkl'):
        curr_dict = load_pickle(f_name)
        label = curr_dict['label']
        for dat in curr_dict['data']:
            words_vec_arr.append(dat['words_vec'])
            ent_arr.append(dat['entity'])
            lab_arr.append(label)

    # Randomly shuffle the data
    words_vec_arr, ent_arr, lab_arr = np.array(words_vec_arr), np.array(ent_arr), np.array(lab_arr)

    # One hot encoding for the labels required for categorical cross entropy loss
    lab_arr = to_categorical(lab_arr, num_classes=7)

    # Shuffle all the data
    shuffle_arr = np.arange(len(words_vec_arr))
    np.random.shuffle(shuffle_arr)
    words_vec_arr, ent_arr, lab_arr = words_vec_arr[shuffle_arr], ent_arr[shuffle_arr], lab_arr[shuffle_arr]

    # Split data if required
    if train:
        train_len = int(train * len(words_vec_arr))
        logging.info('Total training samples: {}\n Validation: {}'.format(train_len, len(words_vec_arr) - train_len))
        save_pickle(fname=folder + '/train_dat.pkl', dat={'words_vec': words_vec_arr[:train_len],
                                                          'ent_arr': ent_arr[:train_len],
                                                          'lab_arr': lab_arr[:train_len]})
        save_pickle(fname=folder + '/val_dat.pkl', dat={'words_vec': words_vec_arr[train_len:],
                                                        'ent_arr': ent_arr[train_len:],
                                                        'lab_arr': lab_arr[train_len:]})
    else:
        save_pickle(fname=folder + '/combined.pkl', dat={'words_vec': words_vec_arr,
                                                         'ent_arr': ent_arr,
                                                         'lab_arr': lab_arr})


def generate_plots(file_name, out_name=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(file_name)

    fig, ax1 = plt.subplots()
    l1 = ax1.plot(df["Epoch"], df["Accuracy"], 'b-', label="Training Accuracy")
    l2 = ax1.plot(df["Epoch"], df["Validation_Accuracy"], 'b--', label="Validation Accuracy")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params('y', colors='b')

    # Make the y-axis label, ticks and tick labels match the line color.
    ax2 = ax1.twinx()
    ax2.tick_params('y', colors='b')
    l3 = ax2.plot(df["Epoch"], df["Loss"], 'r-', label="Training Loss")
    l4 = ax2.plot(df["Epoch"], df["Validation_Loss"], 'r--', label="Validation Loss")
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params('y', colors='r')

    # Printing Legend
    lns = l1 + l2 + l3 + l4
    labs = [curr_line.get_label() for curr_line in lns]
    ax1.legend(lns, labs, loc='center')
    if out_name:
        plt.savefig(out_name, dpi=300)
        return
    plt.show()


def get_data(fields, tr_split, val_split, hard_cutoff, base_folder):
    # Load the combined data
    main_dat = load_pickle('Data/combined_dat.pkl')
    logging.info('Classwise split is performed')
    tr_dict, val_dict, test_dict = {}, {}, {}
    if not hard_cutoff and (tr_split + val_split > 1):
        logging.warning('Total split has to be less than 1 \n Exiting...'), exit()
    for key in main_dat.keys():
        if key not in fields:
            continue
        # Randomly shuffle the array
        shuffle(main_dat[key])
        # Get indices
        if hard_cutoff:
            val_st, val_end = int(tr_split), int(tr_split + val_split)
        else:
            val_st = int(tr_split * len(main_dat[key]))
            val_end = val_st + int(val_split * len(main_dat[key]))
        logging.info('For field:{} Train:{} Validation:{} Test:{}'.format(
            key, val_st, val_end - val_st, len(main_dat[key]) - val_end))
        if len(main_dat[key]) - val_end <= 0:
            print(len(main_dat[key]) - val_end)
            logging.warning('No test samples for field:{} \n Exiting...'.format(key))
        # Split dataset and store it
        tr_dict[key] = main_dat[key][:val_st]
        val_dict[key] = main_dat[key][val_st:val_end]
        test_dict[key] = main_dat[key][val_end:]
    save_pickle(dat=tr_dict, fname=base_folder + '/tr.pkl'), save_pickle(dat=val_dict, fname=base_folder + '/val.pkl')
    save_pickle(dat=test_dict, fname=base_folder + '/test.pkl')
    logging.info('Saved data after the split (test, tr, val).pkl')


def check_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='NN DRIVER')
    parser.add_argument("-tr_split", default=2000, type=float, help="Number of samples to use train set")
    parser.add_argument("-val_split", default=400, type=float, help="Number of samples to use val set")
    parser.add_argument("-base_folder", default="", required=True, type=str, help="Base folder")
    parser.add_argument("--hard_cutoff", action="store_true", help="If yes splits are hard cutoff")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level='INFO')
    get_data(ALLCLASS, args.tr_split, args.val_split, args.hard_cutoff, args.base_folder)
