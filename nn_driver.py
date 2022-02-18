import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# NN_DICT
NN_DICT = {1: "CNN model", 2: "LSTM model"}


def main(args):
    import nn_util
    from keras.optimizers import adam
    # Neural network architecture choosing
    if args.nn == 1:
        nn = nn_util.cnn_network()
    elif args.nn == 2:
        nn = nn_util.lstm_network()
    else:
        logging.info('Invalid neural network architecture selected \nExiting....'), exit()

    nn.compile(optimizer=adam(lr=args.lr), loss=args.loss_fn, metrics=['accuracy'])
    logging.info('Adam optimizer with learning rate: {}\nLoss Function: {}'.format(args.lr, args.loss_fn))
    nn_obj = nn_util.nn_driver(nn=nn, val_thresh=args.val_thresh, track_progress=args.track_progress,
                               out_dir=args.out_dir, num_class=args.num_class, name=args.name)
    if args.train:
        nn_obj.train(train_file=args.train_file, val_file=args.val_file, batch_size=args.bs,
                     init_epoch=args.init_epoch, prev_train=args.prev_train, tot_epochs=args.tot_epoch,
                     stop_delta=args.stop_delta, stop_patience=args.stop_patience)
    if args.test:
        if args.test_acc is None:
            logging.warning('Please enter test_acc file\nExiting...'), exit()
        nn_obj.testing(test_file=args.test_file, param=args.test_param, test_acc_file=args.test_acc)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='NN DRIVER')
    # parser.add_argument("-optimizer", default="adam", type="str", help="Optimizer to use")
    parser.add_argument("-lr", default=0.0001, type=float, help="Learning rate of adam optimizer")
    parser.add_argument("-train_file", default="Data/tr.pkl", type=str, help="Training pickle file to use")
    parser.add_argument("-val_file", default="Data/val.pkl", type=str, help="Validation pickle file to use")
    parser.add_argument("-test_file", default="Data/test.pkl", type=str, help="Test data pickle file to use")
    parser.add_argument("-test_param", default=None, type=str, help="Parameters to load for testing")
    parser.add_argument("-val_thresh", default=0.6, type=float,
                        help="If validation accuracy above this store the parameters")
    parser.add_argument("-stop_delta", default=0.01, type=float,
                        help="If validation accuracy does not increase by more than 1% stop")
    parser.add_argument("-stop_patience", default=3, type=int, help="Early stopping patience")
    parser.add_argument("-bs", default=32, type=int, help="Batch Size")
    parser.add_argument("-init_epoch", default=0, type=int, help="Starting epoch number")
    parser.add_argument("-tot_epoch", default=50, type=int, help="Total epochs")
    parser.add_argument("-track_progress", default=None, required=True, help=".txt file to track accuracies")
    parser.add_argument("-loss_fn", default="categorical_crossentropy", type=str, help="Loss Function to use")
    parser.add_argument("-nn", default=1, required=True, type=int, help="NN to use:\n1: CNN Model \n2: LSTM Model")
    parser.add_argument("--train", action="store_true", help="Train")
    parser.add_argument("-out_dir", default="nn_models", type=str,
                        help="Directory where the model parameter should be saved")
    parser.add_argument("-log", default="INFO", type=str, help="Logging level to use")
    parser.add_argument("-prev_train", default=None, type=str, help="Parameters to load a train model")
    parser.add_argument("--test", action="store_true", help="Test")
    parser.add_argument("-num_class", default=7, type=int, help="Number of classes")
    parser.add_argument("-name", default=None, type=str, help="Name of the nn", required=True)
    parser.add_argument("-test_acc", default=None, type=str, help="Name of the file with test accuracy")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', level=args.log)
    main(args)
