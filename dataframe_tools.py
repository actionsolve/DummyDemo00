from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import unittest
import time as time



#-------------------------------------------------------------------
def load_ts_data(file_name, column_name_datetime, min_frames_needed=0, verbose=True):

    ''' Load timeseries data, and do simple data integrity checks
       * Enough frames
       * Timestamp monotonically increasing    '''

    # Load data, and check sizes
    df = pd.read_csv(file_name)
    if(verbose): print("    Loaded data [%s] -> % 6d rows  x  %d columns: \n      %s"
            % (file_name, len(df), len(df.columns), df.columns.values))    # ; print(df.info())
    if(min_frames_needed > 0):
        assert len(df) >= min_frames_needed, "Not enough frames: %d < %d" % (len(df), min_frames_needed)

    # Set datetime col as index
    df[column_name_datetime] = pd.to_datetime(df[column_name_datetime])
    df = df.set_index(column_name_datetime);                            # ; print(df_full.info())
    if(verbose): print("      Date range:  %s" % ( get_date_range_as_str(df, False)))

    # Check timestamp increasing
    assert df.index.is_monotonic_increasing, "Datetime index is NOT incr over time - check order"

    return df


#-------------------------------------------------------------------
def split_by_date(df, dt):
    
    #print("      split_by_date(df(%d frames), @ dt=%s)" % (len(df), dt))
    
    '''
    Returns df0, df1
    First df is < dt,  second df is >= dt
    '''
    
    # Check index is datetime
    #assert type(df.index) is datetime, "df.index is not an datetime" 
    assert isinstance(df.index, pd.DatetimeIndex), "df.index is not datetime" 
    assert len(df) > 0, "df is empty"

    # Get date extrema, and check split date in range
    date_first, date_last, num_days_in_df = get_date_range(df)
    assert dt >= date_first, 'Date (%s) is outside range: %s .. %s' % (dt, date_first, date_last)
    assert dt <= date_last,  'Date (%s) is outside range: %s .. %s' % (dt, date_first, date_last)
    #print("WARN: %s is < %s in df" % (1,2)) if  dt > date_first

    # Split
    df0 = df[:dt-timedelta(milliseconds=1)]
    df1 = df[dt:]
    
    return df0, df1
    
'''        
        df.set_index('Timestamp', inplace=True)
        df[d1:d0]
        df.loc['2014-01-01':'2014-02-01']
        df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]
        
        df['20160101':'20160301']
        
        from datetime import date

        import pandas as pd
        
        value_to_check = pd.Timestamp(date.today().year, 1, 1)
        filter_mask = df['date_column'] < value_to_check
        filtered_df = df[filter_mask]
        
        import datetime 
        df.loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2014,month=2,day=1)]
'''


def split_by_num(df, num_training, num_test, num_validation=0, verbose=False):
    if (verbose):
        print("    split_by_num(%d frames into %.2f : %.2f : %.2f)"
              % (len(df), ratio_training, ratio_test, ratio_validation))
    # ok_sum = ((ratio_training + ratio_test + ratio_validation) == 1.0)
    assert ( (num_training + num_test + num_validation) <= len(df)) \
        , "Unusable split ratios: %.2f, %.2f, %.2f" % (ratio_training, ratio_test, ratio_validation)

    # Training data
    # logging.info("    num_training %d" % (num_training))
    df_training = pd.DataFrame(df.head(num_training))
    # print(data_training.describe());print(data_training)
    if (verbose):
        print("    Train: %d frames  x %d channels"
              % (len(df_training), len(df_training.columns)))
    # logging.info("    Columns: %s" % (data_training.columns))

    # Test Data
    # logging.info("    num_test %d" % (num_test))
    df_test = pd.DataFrame(df.head(num_training + num_test).tail(num_test))
    # print(data_test.describe());print(data_test)
    if (verbose):
        print("    Test : %d frames  x %d channels"
              % (len(df_test), len(df_test.columns)))

    # Validation Data
    # logging.info("    num_validation %d" % (num_validation))
    df_validation = pd.DataFrame(df.tail(num_validation))
    # print(data_test.describe());print(data_validation)
    if (verbose):
        print("    Valid: %d frames  x %d channels"
              % (len(df_validation), len(df_validation.columns)))

    return df_training, df_test, df_validation



#-------------------------------------------------------------------
def split_by_ratio(df, ratio_training, ratio_test, ratio_validation=0.0, verbose=False):

    if(verbose):
        print("    split_by_ratio(%d frames into %.2f : %.2f : %.2f)"
                 % (len(df), ratio_training, ratio_test, ratio_validation))
    #ok_sum = ((ratio_training + ratio_test + ratio_validation) == 1.0)
    assert( (ratio_training + ratio_test + ratio_validation) <= 1.0) \
           , "Unusable split ratios: %.2f, %.2f, %.2f" % (ratio_training, ratio_test, ratio_validation)
    
    # Calc numbers for each split
    num_training = int(len(df) * ratio_training )
    num_validation = int(len(df) * ratio_validation )
    # num_test = int(len(df) * ratio_test )  # May lose frame, due to int rounding
    num_test = len(df) - (num_training + num_validation)

    df_training, df_test, df_validation = split_by_num(df, num_training, num_test, num_validation
                        , verbose=verbose)

    '''
    # Training data
    #logging.info("    num_training %d" % (num_training))
    df_training = pd.DataFrame(df.head(num_training))
    #print(data_training.describe());print(data_training)
    if(verbose):
        print("    Train: %d frames  x %d channels"
                 % (len(df_training), len(df_training.columns)))
    #logging.info("    Columns: %s" % (data_training.columns))
  
    # Test Data
    #logging.info("    num_test %d" % (num_test))
    df_test = pd.DataFrame(df.head(num_training+num_test).tail(num_test))
    #print(data_test.describe());print(data_test)
    if(verbose):
        print("    Test : %d frames  x %d channels"
                 % (len(df_test), len(df_test.columns)))
    
    # Validation Data
    #logging.info("    num_validation %d" % (num_validation))
    df_validation = pd.DataFrame(df.tail(num_validation))
    #print(data_test.describe());print(data_validation)
    if(verbose):
        print("    Valid: %d frames  x %d channels"
                 % (len(df_validation), len(df_validation.columns)))
    '''
    return df_training, df_test, df_validation


def get_slice_before_date(df, dt_end, num_frames):
    '''
    Get df slice containing given num frames, and ending at 'dt_end'
    '''

    # Check index is datetime
    #assert type(df.index) is datetime, "df.index is not an datetime"
    assert isinstance(df.index, pd.DatetimeIndex), "df.index is not datetime"
    assert len(df) > 0, "df is empty"

    df_slice = df[df.index < dt_end]
    df_slice = df_slice.tail(num_frames)

    return df_slice


#-------------------------------------------------------------------
def get_date_range(df):

    # Check index is datetime
    #assert type(df.index) is datetime, "df.index is not an datetime" 
    assert isinstance(df.index, pd.DatetimeIndex), "df.index is not datetime" 
    assert len(df) > 0, "df is empty"

    date_first = df.head(1).index[0]
    date_last = df.tail(1).index[0]
    num_days_in_df = (date_last - date_first).days
    #print("      get_date_range(%d frames):   %s  ..  %s" % (len(df), date_first, date_last))

    return (date_first, date_last, num_days_in_df)


#-------------------------------------------------------------------
def get_date_range_as_str(df, short_format = True):

    is_dt_index = isinstance(df.index, pd.DatetimeIndex)
    if(is_dt_index):

        date_first, date_last, num_days_in_df = get_date_range(df)
        if(short_format):
            msg = ("% 8d rows,% 4d days:   %s  ..  %s"
                        % (len(df), num_days_in_df, date_first.date(), date_last.date()))
        else:
            msg = ("% 8d rows,% 4d days:   %s  ..  %s"
                        % (len(df), num_days_in_df, date_first, date_last))

    else:
        msg = ("% 8d rows (non-datetime index)"    % (len(df)))
    return msg


#-------------------------------------------------------------------
#def get_rc_as_str(df, show_chan_names=false) -> str:
def get_rc_as_str(df, show_chan_names=False):

    if df is None or len(df)==0:
        return "EMPTY DF"

    num_na_worst_case = df.isna().sum().max()  # max across all columns -> worst case
    if(show_chan_names):
        str = "%d rows  (%d OK  +  %d NAN), %d cols: %s" \
            % (len(df), len(df) - num_na_worst_case, num_na_worst_case, len(df.columns)
            , df.columns.values)
    else:
        str = "%d rows  (%d OK  +  %d NAN), %d cols" \
              % (len(df),  len(df) - num_na_worst_case, num_na_worst_case, len(df.columns))
    return str


#-------------------------------------------------------------------
def get_info_as_str(df, chan_name):

    series = df[chan_name]
    #num_na_count = df.isna().sum().max()  # max across all columns -> worst case
    num_na = series.isna().sum()
    str = "[%s]  %d rows:   Range: %.3f .. %.3f,   Mean: %.3f,   Std: %.3f    NaN: %d" \
          % (chan_name, len(series), series.min(),  series.max(),  series.mean(),  series.std(), num_na)

    return str



# ---------------------------------------------------------------------
def change_timeframe(df, minutes_per_frame, verbose=False):
    '''
    Change timeframe, 
        minutes_per_frame=1440 -> 1 frame for every day
    '''
    ohlc_dict = { 'Open' : 'first',
                  'High' : 'max',
                  'Low'  : 'min',
                  'Close': 'last' #,'Volume':'sum'
                }
    # Resample df
    num_frames_start = len(df)
    resample_str = str(minutes_per_frame) + 'T'
    #print('      Resampling(%s) @ %d frames orig frames -> 1 new frame' 
    #        % (resample_str, minutes_per_frame))
    df2 = df.resample(resample_str, closed='left', label='left').apply(ohlc_dict)
    #print('            %d -> %d frames' % (num_frames_start, len(df2)))
    print('      Resampling(%s) @ %d frames orig frames -> 1 new frame    (%d -> %d frames)' 
            % (resample_str, minutes_per_frame, num_frames_start, len(df2)))
    
    # Preserve column ordering
    df2 = df2[df.columns.values]

    # Return df
    return df2


# ---------------------------------------------------------------------
def get_trimmed_on_midnight(df):

    date_first, date_last, num_days_in_df = get_date_range(df)
    #data1M.to_csv( 'tmp.2018_data1H.csv', float_format='%.5f')

    # Ensure midnight boundaries -> top and tail df to fit
    date_first = (date_first + timedelta(days=1)).replace(hour=0, minute=0, second=0)
    date_last  = (date_last - timedelta(days=1)).replace(hour=0, minute=0, second=0)
    _, df = split_by_date(df, date_first) 
    df, _ = split_by_date(df, date_last) 
    #num_days_in_df = (date_last - date_first).days

    return df


#-------------------------------------------------------------------
# Add target direction channel, based on src channel
def add_channel_target_C2C(df, chan_name_src, chan_name_target
                , trim_na_at_end=False, add_logistic=False, verbose=True):

    if (verbose): print("        Adding target channel: '%s', based on (next-current) '%s', (%d frames)"
          % (chan_name_target, chan_name_src, len(df)))

    # Num frames expect to lose, due to priming filters
    num_frames_initially = len(df)
    num_frames_expected_to_lose = 1

    # Add temp channel to house tomorrow's value
    chan_name_tmp = 'T_next_tmp'
    df[chan_name_tmp] = df[chan_name_src].shift(-1)
    
    # Form diff between today and tomorrow
    df[chan_name_target] = df[chan_name_tmp] - df[chan_name_src]
    if(add_logistic):
        #df[chan_name_target] = df[chan_name_target].apply(lambda x : 1 if x > 0 else 0)
        df[chan_name_target + '_l'] = df[chan_name_target].apply(lambda x : 1 if x > 0 else 0)

    # Drop temp channel, and invalid frames
    df.drop([chan_name_tmp], axis=1, inplace=True)

    # Trim NaN at end
    if(trim_na_at_end):
        df.dropna(inplace = True)
        if (verbose): print('      -> %d frames' % len(df))
        if not add_logistic:
            assert ( num_frames_initially-num_frames_expected_to_lose) == len(df) \
                    , 'Lost more NA frames (%d - %d <> %d), than expected' \
                    % (num_frames_initially, num_frames_expected_to_lose, len(df))

    return df


def add_macd(df, chan_name_target, chan_name_source, window_long, window_short, use_ema=False):
    """
    Copied from F50_Frank
    :param df:
    :param chan_name_target:
    :param chan_name_source:
    :param window_long:
    :param window_short:
    :param use_ema:
    :return:
    """
    # Sanity checks
    assert window_long > 0 and window_short > 0 and window_long > window_short \
            , "Unusable params: window_long: %d, window_short: %d" % (window_long, window_short)
    assert len(df) > (window_long), "Not enough frames: %d <= %d" % (len(df), window_long)

    # Source column
    series_src = df[chan_name_source]

    # Form MA, and expected num NA frames
    if(use_ema):
        series_short = series_src.ewm(span=window_short, adjust=False).mean()
        series_long  = series_src.ewm(span=window_long,  adjust=False).mean()
        expected_na_frames = 0
    else:
        series_short = series_src.rolling(window = window_short).mean()
        series_long  = series_src.rolling(window = window_long).mean()
        expected_na_frames = window_long - 1

    df[chan_name_target] = series_long - series_short

    # Update chan names
    #assert chan_name_target not in self.__chan_names, "Chan [%s] added twice" % chan_name_target
    #self.__chan_names.append(chan_name_target)
    #self.__expected_na_frames = max(self.__expected_na_frames, window_long-1)

    return df, expected_na_frames



#-------------------------------------------------------------------
def add_channel_target_M_O(df, chan_name_target
                , trim_na_at_end=False, add_logistic=False, verbose=True):
    if (verbose): print("        Adding target channel: '%s', based on next M_O,  (%d frames)"
          % (chan_name_target, len(df)))
    # Channel names
    chan_name_high   = 'HIGH'
    chan_name_low    = 'LOW'
    chan_name_open   = 'OPEN'
    chan_name_close  = 'CLOSE'
    chan_name_vol    = 'VOLUME'
    chan_name_med    = 'Median'

    # Num frames expect to lose, due to priming filters
    num_frames_initially = len(df)
    num_frames_expected_to_lose = 1

    # Add channel: Median - Open, for next frame
    df_future = df.shift(-1)
    df[chan_name_target] =  df_future[chan_name_med] - df_future[chan_name_open]
    if(add_logistic):
        #df[chan_name_target + '_l'] = 1 if(df[chan_name_target] > 0) else 0
        df[chan_name_target + '_l'] = df[chan_name_target].apply(lambda x : 1 if x > 0 else 0)

    # Trim NaN at end
    if(trim_na_at_end):
        # Drop temp channel, and invalid frames
        #df.drop([chan_name_tmp], axis=1, inplace=True)
        df.dropna(inplace = True)
        #print('      -> %d frames' % len(df))
        if not add_logistic:
            assert (num_frames_initially-num_frames_expected_to_lose) == len(df) \
                    , 'Lost more NA frames (%d - %d <> %d), than expected' \
                      % (num_frames_initially, num_frames_expected_to_lose, len(df))

    #print(df.head(2));    print(df.tail(2))

    return df


def drop_na_with_check(df, expected_num_na_frames = -1, verbose=False):

    '''
    Drop na frames, in place, with check on number of frames
    '''
    num_frames_before = len(df)
    #print('df.isna().sum(): %s' % df.isna().sum())
    num_na_count = df.isna().sum().max()
    df.dropna(inplace = True)
    num_frames_after = len(df)
    if(verbose): print('    Dropped %d NA frames, %d initially and %d remaining'
            % (num_na_count, num_frames_before, num_frames_after))

    if(expected_num_na_frames >= 0):
        assert expected_num_na_frames == num_na_count \
                , "Unexpected num of NaN frames found: %d  !=  %d (found)" \
                % (expected_num_na_frames, num_na_count)
        assert num_frames_before == (num_frames_after + expected_num_na_frames) \
                , "Unexpected num of NaN frames: %d - %d  !=  %d    (count na: %d)" \
                % (num_frames_before, expected_num_na_frames, num_frames_after, num_na_count)


#-------------------------------------------------------------------
def split_df_by_chan_names(df, chan_names_in_second_only, single_chan_in_second=None):
    ''' Split df by columns, into df1 and df2
    '''

    df_1 = df.drop(list(chan_names_in_second_only), axis=1, errors='ignore')

    if(single_chan_in_second is not None):
        #chan_names = chan_names[0]
        #df_2 = pd.DataFrame(df[chan_names])
        df_1 = df_1.drop(single_chan_in_second, axis=1, errors='ignore')
        df_2 = pd.DataFrame(df[single_chan_in_second])
    else:
        df_2 = df[chan_names_in_second_only]

    return df_1, df_2


# -----------------------------------------------------------------
def check_columns_present(df, required_column_names):

    #column_names = dataframe.columns.values
    for column_name in required_column_names:
        assert column_name in df.columns.values , "Col '%s' not is cols: %s" % (column_name, df.columns.values)





#-------------------------------------------------------------------
def clamp_extreme_values(df, chan_name, num_sd_to_limit=3, positive_only=False
        , show_only=False, verbose=False):

    if(verbose): print("    Clamp extreme values(%d rows, %d chans.  Looking at '%s' channel)"
                       % (len(df), len(df.columns), chan_name))

    # Calc limits
    stddev = df[chan_name].std() ;  mean = df[chan_name].mean()
    #num_sd_to_limit = 3
    limit_hi = mean + num_sd_to_limit * stddev
    limit_lo = mean - num_sd_to_limit * stddev
    #limit_lo = 0 if(positive_only) else mean - num_SD_to_limit * stddev
    if(positive_only): limit_lo = 0
    if(verbose): print('      [%s] Mn: %.3f,  SD: %.3f, clamp @ %d SD     -> Limits: Lo: %.3f .. Hi %.3f'
          % (chan_name, mean, stddev, num_sd_to_limit, limit_lo, limit_hi))

    # Count extrema found
    num_hi_extrema = len(df[df[chan_name] > limit_hi])
    num_lo_extrema = len(df[df[chan_name] < limit_lo])
    #if(verbose and (num_hi_extrema > 0 or num_lo_extrema > 0)):
    if(num_hi_extrema > 0 or num_lo_extrema > 0):
        if(verbose): print('      Found %d hi extrema, and %d lo extrema '
            % (num_hi_extrema, num_lo_extrema))

    # Clamp to appropriate limit
    if(not show_only):
        df.loc[df[chan_name] > limit_hi, chan_name] = limit_hi
        df.loc[df[chan_name] < limit_lo, chan_name] = limit_lo
        if(verbose): print('              -> New  Min: %.3f .. Max %.3f'
            % (df[chan_name].min(), df[chan_name].max()))

    return df


#-------------------------------------------------------------------
def clamp_extreme_values_all(df, positive_only, show_only, verbose):

    for chan_name in df.columns.values:
        df = clamp_extreme_values(df, chan_name, positive_only, show_only, verbose)

    return df


#-------------------------------------------------------------------
def scale_df(df, range_tuple, verbose):

    # Diagnostics
    if(verbose): print(df.describe()) ; print(df.info())

    # Save column names and datetime index
    columns = df.columns  ; index = df.index
    #assert isinstance(df.index, pd.DatetimeIndex), "df.index is not datetime"

    # Apply uniform scaling
    #scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-0.5, 0.5))
    scaler = MinMaxScaler(feature_range=range_tuple)
    scaled_array = scaler.fit_transform(df)

    # Rebuild df
    df = pd.DataFrame(scaled_array, index=index, columns=columns)

    # Diagnostics
    if(verbose): print(df.describe()) ; print(df.info())

    return df


# -------------------------------------------------------------------
def add_channel_correlation(df, chan_name_1, chan_name_2, threshold, chan_name_acc, add_cs, verbose):
    if (verbose): print("  Adding correlation channel, between '%s' and '%s' -> '%s',  Threshold: %.2F"
                        % (chan_name_1, chan_name_2, chan_name_acc, threshold))

    # Simple error calc
    mean_error = (df[chan_name_1] - df[chan_name_2]).abs().mean()

    # Positive if both same sign
    df[chan_name_acc] = (df[chan_name_1] - threshold) * (df[chan_name_2] - threshold)

    # Logistic
    #df[chan_name_new] = df[chan_name_new].apply(lambda x: 1 if x > 0 else 0)
    df[chan_name_acc] = df[chan_name_acc].apply(lambda x: 1 if x > 0 else 0)

    # Count when both 0
    # series_both_zero = df[[chan_name_1, chan_name_2]].apply(lambda x: 1 if x[0] == x[1] else 0)
    # count_both_zero = series_both_zero.count()
    # series_both_same = df[[chan_name_1, chan_name_2]].apply(lambda x: 1 if x[chan_name_1] == x[chan_name_2] else 0)
    # count_both_same = series_both_same.sum()

    # Running cumsum()
    if (add_cs):
        df[chan_name_acc + '_cs'] = df[chan_name_acc].cumsum()

    # Print mean of channel
    mean_correct = np.mean(df[chan_name_acc])
    # print("      -> Mean: %.3f,  from %d values    (Both zero: %d)"
    #        % (mean, len(df[chan_name_new]), count_both_zero))
    # print("      -> Mean: %.3f,  from %d values    (Both same: %d)"
    #        % (mean, len(df[chan_name_new]), count_both_same))

    corr = df[chan_name_1].corr(df[chan_name_2])
    print('    [% 8s] vs [% 8s]   Corr:% 8.4f' % (chan_name_1, chan_name_2, corr))

    if (verbose): print("      -> Mean: %.3f,  from %d values    "
                        % (mean_correct, len(df[chan_name_acc])))

    return df, mean_correct, mean_error


#-------------------------------------------------------------------
class MyTest(unittest.TestCase):

    def create_dataframe(self):
        
        global output_report_filename
    
        print('  Generating processed data')
        #date_today = datetime.now().date()
        date_today = datetime(year=2016, month=2, day=10)
        date_range = pd.date_range(date_today, date_today + timedelta(10), freq='D')

        np.random.seed(seed=1111)
        data = np.random.randint(1, high=100, size=len(date_range))

        # extreme values
        data[3] = 1000
        #data[4] = -1000

        df = pd.DataFrame({'Date': date_range, 'Val1': data})
        df = df.set_index('Date')
        #print(df)
        #print(get_date_range(df))
        df['Val2'] = 1 - df['Val1']
        df['Val3'] = df['Val2'] / (df['Val1'] + 0.01)

        #print('  Writing processed data')
        #df.to_csv(output_report_filename, header=True, encoding='utf-8' )     
        #df.to_csv(output_report_filename, header=True, float_format='%.3f', index=False, encoding='utf-8' )     
        return df
    
    
    def setUp(self):
        print('setUp()')
       
        
    def tearDown(self):
        print('tearDown()')
    
    
    def test_split_by_date(self):
        print('\ntest_split_by_date()')
        
        df = MyTest.create_dataframe(self)
        date_split = datetime(year=2016, month=2, day=16)
        df0, df1 = split_by_date(df, date_split)
        print("  Split to %d and %d frames" % (len(df0), len(df1)))
        print(df0)
        print(df1)

        # XXX Check all frames allocated to one of the subsets

        #self.assertEqual(len(df0), )

    
    def test_split_by_ratio(self):
        print('\ntest_split_by_ratio()')
        
        df = MyTest.create_dataframe(self)
        df0, df1, df2 = split_by_ratio(df, 0.1, 0.8, 0.1)
        print("  Split %d, into %d, %d, %d frames"
              % (len(df), len(df0), len(df1), len(df2)))
        print(df0)
        print(df1)
        print(df2)
        
        # XXX Check all frames allocated to one of the subsets
        
        self.assertEqual(len(df0) + len(df1) + len(df2) <= len(df) )
    

    def test_clamp_extreme_values(self):
        print('\ntest_clamp_extreme_values()')

        df = MyTest.create_dataframe(self)

        col_name = df.columns.values[0]
        #date_today = datetime(year=2016, month=2, day=12)
        #df[date_today]

        df = clamp_extreme_values(df, col_name, positive_only=True, show_only=False, verbose=True)


    def test_values_correct(self):
        print('\ntest_values_correct()')
        self.assertEqual(4, 4)


    def test_split_df_by_chan_names(self):
        print('\nsplit_df_by_chan_names()')

        df = MyTest.create_dataframe(self)

        chan_names = ['Val2','Val3']
        df_1, df_2 = split_df_by_chan_names(df, chan_names, single_chan=False)
        print(get_rc_as_str(df_1, show_chan_names=True))
        print(get_rc_as_str(df_2, show_chan_names=True))

        df_1, df_2 = split_df_by_chan_names(df, chan_names, single_chan=True)
        print(get_rc_as_str(df_1, show_chan_names=True))
        print(get_rc_as_str(df_2, show_chan_names=True))


    def test_get_slice_before_date(self):

        print('\ntest_get_slice_before_date()')

        df = MyTest.create_dataframe(self)
        print("  Initially: "+ get_rc_as_str(df, show_chan_names=True))
        print("    Date range:  %s" % ( get_date_range_as_str(df, False)))

        dt_end = datetime(year=2016, month=2, day=17)
        df_slice = get_slice_before_date(df, dt_end, num_frames=3)

        print("  Slice: "+ get_rc_as_str(df_slice, show_chan_names=True))
        print("    Date range:  %s" % ( get_date_range_as_str(df_slice, False)))



    def test_scale_df(self):

        print('\ntest_scale_df()')

        df = MyTest.create_dataframe(self)
        column_name = 'Val1'
        print("  Initially: "+ get_rc_as_str(df, show_chan_names=True))
        print("    Max %.3ff,  Min: %.3f" % ( df[column_name].max(), df[column_name].min() ))

        range_tuple = (-5, 5)
        df = scale_df(df, range_tuple, verbose=False)

        print("  scale_df: "+ get_rc_as_str(df, show_chan_names=True))
        print("    Max %.3ff,  Min: %.3f" % ( df[column_name].max(), df[column_name].min() ))


def test_scaling_inverting():

    print('test_scaling_inverting()')
    # Capture scaling from target channel (before prescale to (0..1)
    # Apply that scaling to predicted output

    # Load data
    df = pd.DataFrame({'A':[80.00,90.20,90.95,96.27,91.21],
                           'B':[103.02,107.26,110.35,114.23,114.68],
                           'C':[0.01, 0.2, 0.3, 0.6, 0.99]})
    print(df)

    # Get scaling for target channel
    range_tuple = (0,1)
    scaler = MinMaxScaler(feature_range=range_tuple)
    #scaled_array = scaler.fit_transform(df[['B']])
    #print(scaled_array)

    # Apply to pred channel
    #reverted_array = scaler.inverse_transform(scaled_array)
    #print(reverted_array)
    #reverted_array = scaler.inverse_transform(df[['C']])
    #print(reverted_array)
    #reverted_array = scaler.inverse_transform(scaled_array)
    #print(reverted_array)

    print(df[['B']])
    scaler.fit(df[['B']])
    #reverted_array = scaler.inverse_transform(df[['B']])
    #print(reverted_array)

    reverted_array = scaler.inverse_transform(df[['C']])
    print(reverted_array)
    df['C_inv'] = reverted_array
    print(df)

    # Save and plot


# -----------------------------------------------------------------
def plot_value_and_hist(df, column_name
                        , block=True, title_text=None
                        , show_as_points=False, highlight_points=False):
    '''
    plot_value_and_hist(data1M, 'Close')
    '''
    num_histo_bins = 50

    # Title
    if (title_text == None):
        title_text = column_name + ",  %d frames" % len(df)
    else:
        title_text = title_text + " -> " + column_name + ",  %d frames" % len(df)

    # plot histogram
    # data1D.hist('Close',bins=100)
    # data1M.hist('Close', bins=100, orientation=u'horizontal')

    # fig = plt.figure(figsize=(13, 4))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122, sharey=ax1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3)
                                   , gridspec_kw={'width_ratios': [3, 1]})

    if (show_as_points):
        ax1.scatter(df.index, df[column_name], marker='o', linewidths=0.0, s=10, c='blue')
    else:
        ax1.plot(df[column_name])
    ax1.set_title('[ %s ]                                   Value                             '
                  % title_text)
    if (highlight_points):
        ax1.scatter(df.index, df[column_name], marker='o', linewidths=0.0, s=20, c='yellow')
    plt.sca(ax1);
    plt.xticks(rotation=45)

    ax1.grid()
    ax2.hist(df[column_name], bins=num_histo_bins, orientation=u'horizontal')
    ax2.set_title('Histogram')

    # plt.tight_layout()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.0)  # only seems to affect vertical

    plt.show(block)  # plt.show(block=False)


#--------------------------------------------------------------------------
def remove_trend(df, chan_name_target, value_target, tolerance, min_remaining_frames, verbose=False):
    '''
    Remove trend by dropping frames which push mean away from value_target (error)
    * N.B. Should only be done AFTER all derived, feature and target channels have been added
    :param df:
    :param chan_name_target:
    :param value_target:
    :param tolerance:
    :param min_remaining_frames:
    :return:
    '''
    if(verbose): print("    Remove Trend (%d frames, '%s', target: %.3f, +/- %.3f,  to min frames %d, mean_0: %.3f)"
            % (len(df), chan_name_target, value_target, tolerance, min_remaining_frames, df[chan_name_target].mean()))
    assert len(df) > 100, "Too few frames (%d) to remove trend"%len(df)
    assert tolerance > 0.0 and tolerance < 1.0, "Unusable threshold %.3f"%threshold
    assert value_target >= 0.0, "Unusable value_target %.3f"%value_target
    assert min_remaining_frames < len(df) and min_remaining_frames > 0, \
            "Unusable min_frames %d, w.r.t. %d"  % (min_frames, len(df))
    num_starting_rows = len(df)

    len_initial = len(df)
    fraction_to_remove = 0.01  # Fraction of df to drop, in each iteration
    fraction_to_remove = 0.01  # Fraction of df_to_remove to drop, in each iteration

    # Assume we intend to get 0 mean on target channel
    mean = df[chan_name_target].mean()  ; error = mean - value_target
    if(verbose): print('      Initial mean: % 7.3f,  Err: % 7.3f,    rows: %d' % (mean, error, len(df)))
    while ( abs(error) > tolerance ):

        assert len(df) >= min_remaining_frames, \
                "Too few frames remaining (%d of %d init), and tolerance NOT met (|%.3f| > %.3f)" \
                % (len(df), len_initial, error, tolerance)

        if(error > 0):
            df_to_remove = df[df[chan_name_target] > value_target]
            num_to_remove = max(int(fraction_to_remove * len(df_to_remove)), 1)
            df = df.drop(df_to_remove.sample(n=num_to_remove).index)  # 51%  -> remove 0.01 of highs

        if(error < 0):
            df_to_remove = df[df[chan_name_target] < value_target]
            num_to_remove = max(int(fraction_to_remove * len(df_to_remove)), 1)
            df = df.drop(df_to_remove.sample(n=num_to_remove).index)  # 51%  -> remove 0.01 of highs

        # Recalc mean and error
        mean = df[chan_name_target].mean()  ; error = mean - value_target
        if(verbose): print('      Mean: % 7.3f,  Err: % 7.3f,    rows: %d  (Removed %d)'
                    % (mean, error, len(df), num_to_remove))

    if(verbose): print('      -> Mean: % 7.3f,  Err: % 7.3f,    remaining rows: %d  (Removed %d)'
            % (mean, error, len(df), (num_starting_rows-len(df))))

    return df

#-------------------------------------------------------------------
def test_trend_normalising():

    epic_symbol = 'MRW.L'  # 'LLOY.L'  'MRW.L'  'GLEN.L' 'NG.L', 'BP.L',
    # 'GLEN.L', 'MRW.L', 'NG.L', 'BP.L', 'LLOY.L', 'AV.L', 'VOD.L', 'TSCO.L' 'BLND.L', 'DLG.L', 'GVC.L'
    # , 'RMG.L', 'SGE.L', 'WPP.L', 'AAL.L', 'EVR.L', 'FRES.L', 'RDSB.L', 'RIO.L', 'WG.L'

    column_name_datetime    = 'Datetime'
    frame_size_in_minutes   = 60
    chan_name_src           = 'Median'
    file_name               = '../Data.test/Share_%s_%dm.csv' % (epic_symbol, frame_size_in_minutes)
    column_name_target_M_O  = 'T_M_O=X'  # 'T_M_O=X'
    column_name_target_C_C  = 'T_C_C'
    column_name_target_C_C_l  = column_name_target_C_C +'_l'


    # Load data
    df = load_ts_data(file_name, column_name_datetime, min_frames_needed=100, verbose=True)

    # Show a clear trend
    df = df.head(400)        # Take of first n frames, so we have a trend
    df = df.reset_index()   # Hide weekend gaps (from datetime index)
    print('  DF:  %s' % (get_rc_as_str(df, show_chan_names=True)))

    # Add target chans
    #df = fw_support.add_channels_derived(df_Mn, trim_na_at_start=True, allow_future_calcs=True, verbose=True)
    df['Median'] = (df['HIGH'] + df['LOW']) / 2.0
    df = add_channel_target_M_O(df, column_name_target_M_O, trim_na_at_end=True, add_logistic=False, verbose=True)
    df = add_channel_target_C2C(df, chan_name_src, column_name_target_C_C, trim_na_at_end=True, add_logistic=True, verbose=True)
    # Adds logistic chan: column_name_target_C_C_l  = column_name_target_C_C +'_l'

    # Metrics on target chans
    print('\nInitially' )
    print('\n    Chan: %s:  Mean: % 7.3f' % (column_name_target_M_O, df[column_name_target_M_O].mean()))
    print('    Chan: %s:  Mean: % 7.3f' % (column_name_target_C_C_l, df[column_name_target_C_C_l].mean()))

    # Initial plots
    #plot_value_and_hist(df, chan_name_src, block=False, title_text=' 0 ')

    # Remove trend - linear channel
    print('\nRemoving trend for %s' % column_name_target_M_O)
    df = remove_trend(df, column_name_target_M_O, 0.0, 0.002, int(0.8*len(df)), True)

    # Metrics on target chan
    print('\n    Chan: %s:  Mean: % 7.3f' % (column_name_target_M_O, df[column_name_target_M_O].mean()))

    # Remove trend - logistic channel
    print('\nRemoving trend for %s' % column_name_target_C_C_l)
    df = remove_trend(df, column_name_target_C_C_l, 0.5, 0.002, int(0.8*len(df)), True)

    # Metrics on target chans
    print('\n    Chan: %s:  Mean: % 7.3f' % (column_name_target_C_C_l, df[column_name_target_C_C_l].mean()))

    # N.B.
    #   Do NOT re-add target chans, since will re-eval gaps and produce new mean() + error values

    # Latest plots
    #plot_value_and_hist(df, column_name_target_M_O, block=False, title_text=' 2')
    #plot_value_and_hist(df, 'CLOSE', block=False, title_text=' 3')
    #plot_value_and_hist(df, chan_name_src, block=True, title_text=' 4')


#-------------------------------------------------------------------
if __name__ == "__main__":


    print("\n%s Started  __path__ [%s] __package__ [%s]  "
          % (datetime.now().strftime("%H:%M:%S"), __path__, __package__))
    start_time = time.time()  # Start the clock


    #test_scaling_inverting()
    #test_trend_normalising()

    #unittest.main()

    elapsed_sec = time.time() - start_time
    print('%s Finished    (Elapsed: %.3fs,  (> %d min))'
          % ( datetime.now().strftime('%H:%M:%S'), elapsed_sec, int(elapsed_sec/60)))
    exit(0)
    

