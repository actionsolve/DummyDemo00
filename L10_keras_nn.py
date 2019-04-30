import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os,sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time

import unittest


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#from pandas import read_csv
#from pandas import DataFrame
#from pandas import concat
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import mean_squared_error


#from tensorflow.python import keras as keras
#from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.layers import Input
#from tensorflow.python.keras import Sequential
#import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler

import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.regularizers import Regularizer
#from keras import regularizers
#from keras.layers import Dropout
#from keras.layers import LSTM
#from keras.utils import to_categorical
#from keras import optimizers
#from keras import regularizers
from keras import backend as K

import tensorflow as tf


#sys.path.append('../..')
#import support.channel_tools as channel_tools
#import support.plotting_tools as plot_tools
#import support.dataframe_tools as dataframe_tools
#import support.time_tools as time_tools


'''
Simple Keras NN learner

TODO
  * layer arch
  * activations
  * batch size: training, prediction
  * epochs
  * provide validation data to 'train'.  Can then plot 'val_acc' too
'''

# -------------------------------------------------------------------
class L10_keras_nn():

    # Class members
    __name_short    = 'L10'
    __name_long     = 'L10: Simple NN (Keras Sequential)'
    __name_confidence_short = 'L10c'


    #-------------------------------------------------------------------
    def __init__(self, optimisation_mode, verbose):

        # Initialise instance members
        self.__optimisation_mode = optimisation_mode

        self.__enable_optimise_arch = True
        self.__num_layer_1 = 0
        self.__num_layer_2 = 0

        self.__enable_optimise_training = False
        self.__num_epochs  = 0           # 1000 OK for simple 3ch XOR
        self.__batch_size  = 0           # Lower batch size reduces error, but takes longer.  10% num epochs works
        self.__num_training_frames = 0   # TODO optimise __num_training_frames

        self.__num_output_channels = 0
        self.__num_input_channels  = 0

        self.__est_chan_names = ['?']

        self.__expected_na_frames = 0

        self.__model = None

        print("    %s.init()    optimisation_mode: %s" % (self.__class__.__name__, optimisation_mode) )

        if(verbose):
            print("      Keras  : " + keras.__version__)
            print("      Backend: " + keras.backend.backend())
            print("      TF Vers: " + tf.__version__)
            device_name = tf.test.gpu_device_name()  # import tensorflow as tf
            print('      Found GPU at: [%s]' % device_name)
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            #if device_name != '/device:GPU:0':
            #    raise SystemError('GPU device not found')

        # TF cache clear (previous models?)  Slows down if not done regularly
        #K.clear_session()


    #-------------------------------------------------------------------
    def _init_model(self, num_input_channels, num_output_channels
                 , num_layer_1, num_layer_2
                 , num_training_frames, num_epochs, batch_size, verbose ):

        # TF cache clear (previous models?)  Slows down if not done regularly
        K.clear_session()

        self.__num_output_channels = num_output_channels
        self.__num_input_channels  = num_input_channels

        self.__num_layer_1 = num_layer_1
        self.__num_layer_2 = num_layer_2

        self.__num_training_frames = num_training_frames
        self.__num_epochs  = num_epochs
        self.__batch_size  = batch_size

        self.__est_chan_names = ["y" + str(i) + '^' for i in range(self.__num_output_channels)]
        self.__expected_na_frames = self.__num_training_frames

        print("\n    %s.init_model() -   In: %d -> %d -> %d ->  Out: %d (%s),  Epochs: %d,  Batch: %d "
              % (self.__class__.__name__, self.__num_input_channels
                , self.__num_layer_1, self.__num_layer_2, self.__num_output_channels, self.__est_chan_names
                , self.__num_epochs, self.__batch_size))

        # Stack of layers
        # Dense(units=16, input_dim=num_x_channels, bias_initializer = 'uniform', activation='relu')
        #   bias_initializer = 'uniform', zeros, ones
        #   activation='relu', 'linear', 'sigmoid', 'softmax'
        self.__model = Sequential()
        self.__model.add(Dense(units=self.__num_layer_1, input_dim=self.__num_input_channels
                    , bias_initializer = 'uniform' , activation='relu'
                    #, activity_regularizer=regularizers.l1(0.0001)
                    #, kernel_regularizer=regularizers.l2(0.0001)
                    ))
        #self.__model.add(Dropout(0.1))

        if(self.__num_layer_2 > 0):
            #model.add(Dense(units=8, init = 'uniform', activation='linear'))
            self.__model.add(Dense(units=self.__num_layer_2, bias_initializer ='uniform', activation='relu'))
            #self.__model.add(Dropout(0.05))
            #model.add(Dense(units=num_y_channels, activation='softmax'))
            #model.add(Dense(units=num_y_channels, activation='relu'))  # much better than softmax

        # Ensure last layer activation is 'sigmoid'
        self.__model.add(Dense(units=num_output_channels, bias_initializer ='uniform', activation ='sigmoid'))

        # Custom optimiser
        # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        #opt_sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #opt_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #opt_rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        # Compile model
        # loss: 'categorical_crossentropy', 'mse', 'binary_crossentropy', keras.losses.*
        # opt: 'sgd', 'rmsprop', 'adam', keras.optimizers.*     * JB: 'adam' works well
        # metrics: ['mae', 'acc'], ['accuracy', 'mae', 'acc']
        # TSD: self._model.compile(loss="mse", optimizer="rmsprop", metrics= ['accuracy', 'mae'])
        #self._model.compile(loss="mse", optimizer="rmsprop", metrics= ['accuracy', 'mae'])
        # JB: model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.__model.compile(loss="binary_crossentropy", optimizer="adam", metrics= ['accuracy', 'mae'])



    #-------------------------------------------------------------------
    def get_name(self, verbose = False):
        if verbose:
            return(self.__name_long)
        else:
            return(self.__name_short)


    #-------------------------------------------------------------------
    def get_param_bounds(self):

        # Preserve order:  'exp_wu', 'exp_wl'
        bounds = []

        # Arch
        if(self.__enable_optimise_arch):
            bounds.append((30, 60))    # num_layer_1, self.__num_layer_1
            bounds.append((10, 30 ))   # num_layer_2, self.__num_layer_2

        # Training
        if(self.__enable_optimise_training):
            bounds.append((50, 150))    # __num_training_frames
            bounds.append((400, 1000))  # __num_epochs
            bounds.append((50, 100))    # __batch_size

        return bounds


    #-------------------------------------------------------------------
    def set_param_values(self, param_values, verbose):

        assert(len(self.get_param_keys()) == len(param_values))      # Check lengths same
        #assert(len(self.get_param_bounds()) == len(param_values))      # Check lengths same

        # Preserve order:  'exp_wu', 'exp_wl'
        index = 0

        # Arch
        if (self.__enable_optimise_arch):
            self.__num_layer_1 = int(param_values[index]) ; index += 1
            self.__num_layer_2 = int(param_values[index]) ; index += 1

        # Training
        if(self.__enable_optimise_training):
            self.__num_training_frames = int(param_values[index]) ; index += 1
            self.__num_epochs          = int(param_values[index]) ; index += 1
            self.__batch_size          = int(param_values[index]) ; index += 1

        if (verbose): print("\n    %s   set_param_values(len: %d: %s)"
                            % (self.get_name(), len(param_values), param_values))


    #-------------------------------------------------------------------
    def get_param_keys(self):

        # Preserve order:  'macd0_win_short', 'macd0_win_long'
        keys = []

        # Arch
        if (self.__enable_optimise_arch):
            keys.append('num_layer_1')
            keys.append('num_layer_2')

        # Training
        if(self.__enable_optimise_training):
            keys.append('num_training_frames')
            keys.append('num_epochs')
            keys.append('batch_size')

        return list(keys)

    #-------------------------------------------------------------------
    def get_chan_names(self):

        return self.__est_chan_names


    # -------------------------------------------------------------------
    def get_expected_na_frames(self):

        return self.__expected_na_frames


    # -------------------------------------------------------------------
    def train(self, df_train_x, df_train_y, df_train_M1
              , df_valid_x=None, df_valid_y=None
              , verbose=True):

        if(verbose):
            print("    Training - df_train_x (%d x %d), df_train_y (%d x %d), df_train_M1 (%d x %d)"
                    % (len(df_train_x), len(df_train_x.columns)
                    , len(df_train_y), len(df_train_y.columns)
                    , len(df_train_M1), len(df_train_M1.columns)) )
        assert self.__model is not None, "Model NOT initialised"

        # Trim df_train_x, df_train_y if needed
        assert len(df_train_x) >= self.__num_training_frames, "Too few training frames %d < %d" % (len(df_train_x), self.__num_training_frames)
        assert len(df_train_x) == len(df_train_y), "Mismatched training x,y frames %d <> %d" % (len(df_train_x), len(df_train_y))
        if(len(df_train_x) > self.__num_training_frames):
            df_train_x = df_train_x.tail(self.__num_training_frames)
            df_train_y = df_train_y.tail(self.__num_training_frames)
            print("    * Trimmed training data to %d frames *******************" % len(df_train_x))

        # Train model
        # Lower batch size reduces error, but takes longer
        # x_train and y_train are Numpy arrays, just like in the Scikit-Learn API.
        data_train_x = df_train_x.values  ; data_train_y = df_train_y.values
        if(df_valid_x is not None and df_valid_y is not None):
            data_valid_x = df_valid_x.values  ; data_valid_y = df_valid_y.values
            history = self.__model.fit(data_train_x, data_train_y
                           , validation_data=(data_valid_x, data_valid_y)
                           #, epochs=self._epochs, batch_size=self._batch_size, validation_split=0.33, verbose=False)
                           , epochs=self.__num_epochs, batch_size=self.__batch_size, verbose=False)
        else:
            history = self.__model.fit(data_train_x, data_train_y
                           #, validation_data=(data_valid_x, data_valid_y)
                           #, epochs=self._epochs, batch_size=self._batch_size, validation_split=0.33, verbose=False)
                           , epochs=self.__num_epochs, batch_size=self.__batch_size, verbose=False)


        return history


    # -------------------------------------------------------------------
    def evaluate_training(self, df_test_x, df_test_y, verbose):

        print("    Evaluation - df_test_x (%d x %d: %s), df_test_y (%d x %d)"
                % (len(df_test_x), len(df_test_x.columns)
                , dataframe_tools.get_date_range_as_str(df_test_x, False)
                , len(df_test_y), len(df_test_y.columns) ) )
        assert self.__model is not None, "Model NOT initialised"

        # Evaluate
        # loss_and_metrics = model.evaluate(data_test_x.values, data_test_y.values
        # metrics=['mae', 'acc', metrics.mae, metrics.categorical_accuracy]
        (loss, accuracy, mae) = self.__model.evaluate(df_test_x.values, df_test_y.values
                , batch_size=10, verbose=False)
        print("      Metric names: %s" % self.__model.model.metrics_names)
        print("      -> loss: %.4f,  acc: %.4f%%,  mae: %.4f" % (loss, accuracy * 100, mae))


    # -------------------------------------------------------------------
    def predict(self, df_test_x, verbose):

        if(verbose): print("    Prediction - df_test_x (%d x %d)"
              % (len(df_test_x), len(df_test_x.columns) ) )
        assert self.__model is not None, "Model NOT initialised"

        # Predict
        batch_size = self.__batch_size                  # ;int(self._batch_size/2)  # batch_size=50
        data_pred_y = self.__model.predict(df_test_x.values, batch_size=batch_size)

        # Repackage as df
        df_pred_y = pd.DataFrame(data_pred_y, index=df_test_x.index, columns=self.__est_chan_names)
        #print(data_pred_y)

        if(verbose): print("      ->  df_pred_y (%d x %d)" % (len(df_pred_y), len(df_pred_y.columns) ) )

        return df_pred_y


    # -------------------------------------------------------------------
    def add_feature_channel(self, df_Mn, df_M1, column_names_target, column_name_target, verbose):
        ''' Used with learner optimisation only.  Requires calls to set_param_values( ...)
        :param df_Mn:
        :param df_M1:
        :param verbose:
        :return:
        '''

        if (verbose): print("      %s.add_feature_channel(%d frames)" % (self.get_name(), len(df_Mn)))
        assert df_M1 is None, "df_M1 should be unused in learner (%d rows)" % len(df_M1)

        num_layer_1 = 45; num_layer_2 = 15; num_training = 150 ; num_epochs = 1000; batch_size = 60 # TODO REMOVE

        if(self.__optimisation_mode):

            if (verbose): print("        Adding feature channel, overriding default params")

            # Add feature channel, using self.* values
            # Arch
            if (self.__enable_optimise_arch):
                num_layer_1 = self.__num_layer_1
                num_layer_2 = self.__num_layer_2

            # Training
            if (self.__enable_optimise_training):
                num_training = self.__num_training_frames
                num_epochs   = self.__num_epochs
                batch_size   = self.__batch_size

            # TODO  Correct data set, with accurate num I/P channels
            # TODO  num_training, epochs, batch size
            # TODO  correct num_epochs

        else:

            if (verbose): print("        Adding feature channel, using previously-optimised params")
            #num_layer_1 = 30; num_layer_2 = 20; num_training = 200 ; num_epochs = 1000; batch_size = 100 # TODO REMOVE
            #num_test = 100 # len(df_Mn) - num_training  # no validation set
            # Add optimised/hard-coded version of params: Default values for NN hyper params
            #num_layer_1 = 30; num_layer_2 = 20; num_epochs = 1000; batch_size = 100     # XXX
            #print('*** Badly hacked num_epochs *************************************************************')
            #num_layer_1 = 30; num_layer_2 = 20

            #proportion_for_training = 0.7  # XXX
            #num_training = int(proportion_for_training * len(df_Mn))
            #num_epochs = 1000; batch_size = 50

        # Should we 'FIX' num_test ?
        num_test = len(df_Mn) - num_training  # no validation set

        # Store scaling for target channel: column_name_target_M_O.   # XXX Assumes (0,1) best for learner
        scaler_target = MinMaxScaler(feature_range=(0,1)) ; scaler_target.fit(df_Mn[[column_name_target]])

        # Scale training and test data (0, 1)
        df_scaled = dataframe_tools.scale_df(df_Mn, (0, 1), False) #; df_Mn_y = dataframe_tools.scale_df(df_Mn_y, (0, 1), False)

        # Split df into training, test
        print("    Training on %d, of %d,  testing on %d" % (num_training, len(df_scaled), num_test))
        assert num_training + num_test <= len(df_scaled)

        # Split df into x, y
        df_train_x, df_train_y = dataframe_tools.split_df_by_chan_names(df_scaled, column_names_target
                    , single_chan_in_second=column_name_target)
        #print("      Train x %d chans: %s" % (len(df_train_x.columns.values), df_train_x.columns.values))
        print("      Train x : %s" % (dataframe_tools.get_rc_as_str(df_train_x, verbose)))
        print("      Train y : %s" % (dataframe_tools.get_info_as_str(df_train_y, column_name_target)))

        # Train and forecast - single shot
        mean_abs_error, max_abs_error, learner, df_comparison = train_and_predict(self
                , num_layer_1=num_layer_1, num_layer_2=num_layer_2
                , num_epochs=num_epochs, batch_size=int(batch_size)
                , column_name_target=column_name_target
                , df_train_x=df_train_x, df_train_y=df_train_y
                , num_training=num_training, num_test=num_test
                , use_training_data_for_test=False, verbose=False)

        # Recover y predictions, revert scaling
        chan_name_int_est        = learner.__est_chan_names[0]
        chan_name_ext_est        = "E_%s_%s" %  (learner.get_name(), column_name_target)
        df_comparison[chan_name_ext_est] = scaler_target.inverse_transform(df_comparison[[chan_name_int_est]])

        # Copy across to df
        df_Mn[chan_name_ext_est] = df_comparison[chan_name_ext_est]
        self.__est_chan_names    = [chan_name_ext_est]

        if (verbose): print("      %s.add_feature_channel(), added [%s]" % (self.get_name(), chan_name_ext_est))

        return df_Mn


# -----------------------------------------------------------------
def plot_history0(history):

    print("  Plot training history")
    print("    history.history.keys() -> [%s]" % history.history.keys())

    # History for accuracy
    plt.subplot(2, 1, 1)  # rows, columns, and which subplot you're on. So 1, 2, 1 -> "1-row, 2-column"
    # metrics=['mae', 'acc']
    plt.plot(history.history['acc'])
    plt.plot(history.history['mean_absolute_error'])  # ; plt.plot(history.history['val_acc'])
    plt.title('model accuracy')  ;  plt.ylabel('accuracy')  ; plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()

    # History for loss
    plt.subplot(2, 1, 2)  # rows, columns, and which subplot you're on. So 1, 2, 1 -> "1-row, 2-column"
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')  ; plt.ylabel('loss')  ;  plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Consistent precision on axes
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.03g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.03g'))
    plt.show()
    #plt.show(block=True)


#-----------------------------------------------------------------------------------
def plot_history(history):

    # Get relevant keys in history
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # Calc num epochs  (needed?)
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    plt.title('Loss')  ;   plt.xlabel('Epochs')  ;  plt.ylabel('Loss')  ;     plt.legend()

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    plt.title('Accuracy')  ;   plt.xlabel('Epochs')  ;  plt.ylabel('Accuracy') ;     plt.legend()

    plt.show()


#-----------------------------------------------------------------------
def train_and_predict(learner, num_layer_1, num_layer_2, num_epochs, batch_size
        , column_name_target, df_train_x, df_train_y
        , num_training, num_test
        , use_training_data_for_test, verbose):

    print('--------------------------------------------------')
    # Split into training and test (validation)
    if(use_training_data_for_test):
        print('  XXX Using training data for test')
        df_test_x = df_train_x.copy(); df_test_y = df_train_y.copy()
    else:
        df_train_x, df_test_x, _ = dataframe_tools.split_by_num(df_train_x, num_training, num_test)
        df_train_y, df_test_y, _ = dataframe_tools.split_by_num(df_train_y, num_training, num_test)
        #df_train_x, df_test_x, _ = dataframe_tools.split_by_ratio(df_train_x, proportion_for_training, 1-proportion_for_training)
        #df_train_y, df_test_y, _ = dataframe_tools.split_by_ratio(df_train_y, proportion_for_training, 1-proportion_for_training)

    # Setup learner
    if(learner is None):
        learner = L10_keras_nn(optimisation_mode=False, verbose=True)
    num_input_channels = len(df_train_x.columns)
    num_output_channels = len(df_train_y.columns)
    num_training_frames = len(df_train_x)
    # num_layer_1 = 30;  num_layer_2 = 20;    num_epochs = 1000;    batch_size = 100
    learner._init_model(num_input_channels, num_output_channels, num_layer_1, num_layer_2
            , num_training_frames, num_epochs, batch_size, True)

    # Evaluate training
    if (verbose): learner.evaluate_training(df_test_x, df_test_y, True)

    # Train
    history = learner.train(df_train_x, df_train_y, df_train_x, True)

    # Plot training history
    if (verbose):
        plot_history(history) # ; plot_history0(history)

    # Evaluate training
    if (verbose): learner.evaluate_training(df_test_x, df_test_y, True)

    # Forecast
    df_y = learner.predict(df_test_x, verbose)
    # print("  Predictions")  ;     print(df_y.head(10))

    # Form df_comparison: [ T_M_O   y0^   err ]
    col_name_est = 'y0^'
    if (verbose): print("  Analysis")  # ;     print(df_y.head(10))
    # df_comparison  = pd.concat([df_test_x, df_test_y, df_y], axis=1)
    df_comparison = pd.concat([df_test_y, df_y], axis=1)
    df_comparison['err0'] = df_comparison[column_name_target] - df_comparison[col_name_est]
    if (verbose):
        with pd.option_context('precision', 3, 'float_format', '{:.3f}'.format):
            print(df_comparison.head(5));   print('...........');   print(df_comparison.tail(5))
        plot_tools.plot_residuals(df_comparison, column_name_target, col_name_est)

    # Calc mean, max error
    error_abs_array = np.abs(df_comparison['err0'])  # ; print(error_abs_array)
    mean_abs_error = error_abs_array.mean()  ;  max_abs_error = error_abs_array.max()
    if (verbose): print('      -> mean abs error: %.4f' % mean_abs_error)
    return mean_abs_error, max_abs_error, learner, df_comparison


#-------------------------------------------------------------------
class MyTest(unittest.TestCase):

    def setUp(self):
        print('setUp()')


    def tearDown(self):
        print('tearDown()')


    def test_train_forecast_xor_3ch(self):

        '''
        Simple train and predict of XOR function + 3rd noisy input
        :return:
        '''
        print('test_train_forecast_xor_3ch()')
        start_time_run = time.time()  # Start the clock

        # Setup demo data
        filename_src = '../Data.test/Data.XOR.3ch.csv'
        df = pd.read_csv(filename_src)
        column_names_target = ["XOR"]
        df_train_x, df_train_y = dataframe_tools.split_df_by_chan_names(df, column_names_target)
        print("      Train x %d: %s" % (len(df_train_x.columns.values), df_train_x.columns.values))
        print("      Train y %d: %s" % (len(df_train_y.columns.values), df_train_y.columns.values))
        # Rename
        df_train_y = df_train_y.rename(columns={"XOR": "y"})

        # Setup learner
        num_input_channels = len(df_train_x.columns);  num_output_channels = len(df_train_y.columns)
        num_layer_1 = 30  ; num_layer_2 = 10  ;
        num_training_frames = len(df_train_x); num_epochs = 100 ;  batch_size = 10
        learner = L10_keras_nn(optimisation_mode=False, verbose=True)
        learner._init_model(num_input_channels, num_output_channels, num_layer_1, num_layer_2
                , num_training_frames, num_epochs, batch_size, True)

        # Train
        history = learner.train(df_train_x, df_train_y, df_train_x
                    , df_valid_x=df_train_x, df_valid_y=df_train_y, verbose=True)
        # Plot training history
        plot_history(history)

        # Get test data
        # XXX VERY BAD practice to reuse training data
        print('  XXX Using training data for test')
        df_test_x = df_train_x.copy()  ; df_test_y = df_train_y.copy()

        # Evaluate training
        learner.evaluate_training(df_test_x, df_test_y, True)

        # Forecast
        df_y = learner.predict(df_test_x, True)
        # print("  Predictions")  ;     print(df_y)

        # Analyse
        df_comparison  = pd.concat([df_test_x, df_test_y, df_y], axis=1)
        df_comparison['err0'] = df_comparison['y'] - df_comparison['y0^']
        with pd.option_context('precision', 3, 'float_format', '{:.3f}'.format):
            print(df_comparison)
        max_abs_error = max(np.abs(df_comparison['err0'])) #; print(max_abs_error)
        self.assertTrue(max_abs_error < 0.5, 'Unexpected large prediction error: %f' % max_abs_error)

        elapsed_sec = time.time() - start_time_run
        print('\n  Elapsed: %.3f  ' % (elapsed_sec))




    def xtest_train_forecast_30ch(self):

        '''
        Test train and predict of timeseries data, unscaled, ~30 channels
        Target channel 'column_name_target' can be linear combo of inputs or product...
        Option to optimise hyperparameters: see 'optimise_params' loop
        :return:
        '''
        print('test_train_forecast_30ch()')
        print('                                            XXX more channels needed')

        filename_src = '../Data.test/Data.L_VOD.L_M20_100x.csv'
        column_name_datetime           = 'Datetime'
        column_names_target            = ['T_M_O', 'T_C_C']
        column_name_target             = column_names_target[0]

        # Setup demo data
        df = pd.read_csv(filename_src)
        print("  Loaded [%s] -> % 6d rows  x  %d columns:\n    %s"
              % (filename_src, len(df), len(df.columns), df.columns.values))    # ; print(df_full.info())
        df[column_name_datetime] = pd.to_datetime(df[column_name_datetime])
        df = df.set_index(column_name_datetime);
        df_train_x, df_train_y = dataframe_tools.split_df_by_chan_names(df, column_names_target, single_chan_in_second=column_name_target)
        print("      Train x %d: %s" % (len(df_train_x.columns.values), df_train_x.columns.values))
        print("      Train y %d: %s" % (len(df_train_y.columns.values), df_train_y.columns.values))

        # Limit chans in df_train_x
        #df_train_x = df_train_x[[ 'M2MD', 'M_O', 'Range' ]]

        # XXX Add predictable target channel: 'T_M_O'
        # df_future = df_train_x.shift(-1)    # Future
        df_train_y[column_name_target] = df_train_x['Median'] * df_train_x['Range']   # Multiplication
        # Just linear sum, similar magnitudes
        #df_train_y[column_name_target] = df_train_x['M2MD'] + df_train_x['M_O'] + df_train_x['Range']
        #print(df_train_y.head(10))

        # Scaling
        df_train_x = dataframe_tools.scale_df(df_train_x, (0, 1), False)
        df_train_y = dataframe_tools.scale_df(df_train_y, (0, 1), False)

        # Default values for NN hyper params
        num_layer_1 = 30; num_layer_2 = 0; num_epochs = 1000; batch_size = 100
        proportion_for_training = 0.7
        num_training = int(proportion_for_training * len(df_train_x))
        num_test     = int( (1.0-proportion_for_training) * len(df_train_x))

        # Iterate through various combinations of NN hyper params to find best learning
        optimise_params = False
        if(optimise_params):

            verbose = False
            df_opt_results = pd.DataFrame(columns=['num_layer_2', 'mae'])

            # 0: start:  num_layer_1 = 30; num_layer_2 = 20; num_epochs = 1000; batch_size = 100
            #for num_layer_1 in range(10, 41, 5):  #    -> any, so choose 10
            #for num_epochs in range(100, 2000, 300):   -> any, so choose 1000
            #for batch_size in range(10, int(num_epochs/2), 100): smaller more accurate (eg 10) but time consuming
            for num_layer_2 in range(0, 10, 2):

                print("\n  Trying  [num_layer_2=%d]" % num_layer_2)

                num_runs = 8
                mean_abs_error_arr = []
                for run_num in range(num_runs):

                    #if(run_num == (num_runs-1)): verbose = True

                    mean_abs_error, max_abs_error, learner, df_comparison = train_and_predict(None
                            , num_layer_1 = num_layer_1, num_layer_2 = num_layer_2
                            , num_epochs = num_epochs, batch_size = int(batch_size)
                            , column_name_target=column_name_target
                            , df_train_x=df_train_x, df_train_y=df_train_y
                            , num_training=num_training, num_test=num_test
                            , use_training_data_for_test=False, verbose=verbose)

                    # Store results
                    mean_abs_error_arr.append(mean_abs_error)
                    print('      -> mean_abs_error: %.4f' % mean_abs_error)
                    row = {'num_layer_2': num_layer_2, 'mae': mean_abs_error}
                    df_opt_results = df_opt_results.append(row, ignore_index=True)

                print("\n  Tried  num_layer_2: %d" % num_layer_2)
                print('    Errors: %s' % (['{:.4f}'.format(i) for i in mean_abs_error_arr]))
                print('      -> mean: %.4f' % np.mean(mean_abs_error_arr))

                print("--------------------------------------")

            df_opt_results['num_layer_2'] = df_opt_results['num_layer_2'].astype(int)
            print(df_opt_results)
            plt.scatter(df_opt_results['num_layer_2'], df_opt_results['mae'], label='df_opt_results')
            plt.xlabel('num_layer_2');            plt.ylabel('mae')
            plt.tight_layout();     _ = plt.legend()  #;  plt.grid()
            plt.show(block=True)

        else:
            # Single shot mode
            verbose = True
            mean_abs_error, max_abs_error, learner, df_comparison = train_and_predict(None
                    , num_layer_1 = num_layer_1, num_layer_2 = num_layer_2
                    , num_epochs = num_epochs, batch_size = int(batch_size)
                    , column_name_target=column_name_target
                    , df_train_x=df_train_x, df_train_y=df_train_y
                    , num_training=num_training, num_test=num_test
                    , use_training_data_for_test=False, verbose=verbose)

        # Unit test check
        self.assertTrue(max_abs_error < 0.5, 'Unexpected large max prediction error: %f' % max_abs_error)
        self.assertTrue(mean_abs_error < 0.3, 'Unexpected large mean prediction error: %f' % mean_abs_error)



#-------------------------------------------------------------------
if __name__ == "__main__":

    print("\n%s Started" % datetime.now().strftime('%H:%M:%S'))
    start_time = time.time()  # Start the clock



    print('Running unit tests')
    print("  Now is %s" % (datetime.now().strftime("YYYYMMDD HH:mm:ss (%Y%m%d %H:%M:%S)")))

    unittest.main(verbosity=2)
    #demo_train_forecast_xor_3ch()



    elapsed_sec = time.time() - start_time
    print('%s Finished    (Elapsed: %.3f)'
            % ( datetime.now().strftime('%H:%M:%S'), elapsed_sec))
    exit(0)
