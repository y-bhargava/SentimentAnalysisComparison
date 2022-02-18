import os
from util import check_dir
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)-15s: %(message)s', level='INFO')
BASE_DIR = "Data"
MASTER_NAME = "CNN"

for i in tqdm(range(20, 120, 20)):
    new_base = '{}/{}_{}'.format(BASE_DIR, MASTER_NAME, i)
    check_dir(new_base)
    for j in range(5):
        final_base = '{}/{}'.format(new_base, j)
        name_nn = '{}_{}_{}'.format(MASTER_NAME, i, j)
        data_ops = 'python3 util.py -tr_split={} -val_split=200 ' \
                   '--hard_cutoff -base_folder={}'.format(i * 15, final_base)
        logging.warning('Data operation being executed: {}'.format(data_ops))
        os.system(data_ops)

        nn_dir = '{}/nn_models'.format(final_base)
        check_dir(nn_dir)

        nn_ops = "python3 nn_driver.py -nn=1 -track_progress={}.txt -name={} --train --test" \
                 " -test_acc=\'test_acc.csv\' -test_file={}/test.pkl -train_file={}/tr.pkl" \
                 " -val_file={}/val.pkl -track_progress={}/track.txt".format(name_nn, name_nn,
                                    final_base, final_base, final_base,final_base)
        logging.warning('NN command being executed: {}'.format(nn_ops))
        os.system(nn_ops)
