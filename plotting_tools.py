from datetime import datetime, timedelta
import logging
import pandas as pd
import time


#from matplotlib import pyplot
#from sklearn import decomposition


from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm

import unittest


# -----------------------------------------------------------------
def plot_value_and_hist(df, column_name
        , block=True, title_text=None
        , show_as_points=False, highlight_points=False):
    
    '''
    plot_value_and_hist(data1M, 'Close')
    '''
    num_histo_bins = 50
    
    # Title
    if(title_text==None):
        title_text = column_name + ",  %d frames"%len(df)
    else:
        title_text = title_text + " -> " + column_name + ",  %d frames"%len(df)
       
    # plot histogram
    #data1D.hist('Close',bins=100)
    #data1M.hist('Close', bins=100, orientation=u'horizontal')
    
    #fig = plt.figure(figsize=(13, 4))
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122, sharey=ax1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw = {'width_ratios':[3, 1]})
    
    if(show_as_points):
        ax1.scatter(df.index, df[column_name], marker='o', linewidths=0.0, s=10, c='blue')
    else:
        ax1.plot(df[column_name])

    ax1.set_title('[ %s ]                                   Value                             '
                % title_text)
    if(highlight_points):
        ax1.scatter(df.index, df[column_name], marker='o', linewidths=0.0, s=20, c='yellow')
    plt.sca(ax1) ; plt.xticks(rotation=45)

    ax1.grid()
    ax2.hist(df[column_name], bins=num_histo_bins, orientation=u'horizontal')
    ax2.set_title('Histogram')

    #plt.tight_layout()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.0)  # only seems to affect vertical

    plt.show(block=block)    #plt.show(block=False)


# -----------------------------------------------------------------
def plot_value_and_hist_list(dfs, column_name, title_texts):

#    for df in dfs:
    for index in range(len(dfs)):
        df = dfs[index]
        title_text = title_texts[index]
        plot_value_and_hist(df, column_name, False, title_text)
    plt.show(block=True)


# -----------------------------------------------------------------
def plot_corr_all_columns(df, chan_name_trg, is_target_logistic, columns_per_plot, block=False):
    '''
    Plot correlation of all columns, except index, against target chan, in batches of 'columns_per_plot'
    Blocks on last batch
    '''

    histo_alpha = 0.4
    histo_bins  = 25
    gridspec_kw = {'width_ratios': [5, 1]}

    if is_target_logistic:

        df_trg = pd.DataFrame(df[chan_name_trg])
        df_no_trg = df.drop(chan_name_trg, axis=1)

        column_offset = 0
        while column_offset < len(df.columns):


            # Get just columns to plot (for this offset)
            df_tmp = pd.DataFrame(df_no_trg.iloc[:, column_offset:column_offset + (columns_per_plot-1)])
            is_last_plot = (column_offset >= (len(df_tmp.columns) - columns_per_plot))
            df_tmp[chan_name_trg] = df_trg[chan_name_trg]
            df_tmp['Idx'] = df_tmp.index  # Get index as channel, for plotting
            #print(df_tmp[['Idx', df_tmp.columns.values[1], df_tmp.columns.values[2]]].head())

            print('  df_tmp cols: %s' % df_tmp.columns.values)

            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharey=True, dpi=120)
            fig, axes = plt.subplots(columns_per_plot, 2, gridspec_kw = gridspec_kw)
            #print('  Range: %s' % range(0, len(df_tmp.columns) - 1))
            colour_values=df_tmp[chan_name_trg].values
            for column_index in range(0, len(df_tmp.columns) - 1):

                # Plot values
                ax_values = axes[column_index, 0]
                column_name = df_tmp.columns.values[column_index]
                values_y = df_tmp[column_name].values

                ax_values.plot(df_tmp['Idx'].values, values_y, c='lightgrey')
                ax_values.scatter(df_tmp['Idx'].values, values_y, c=colour_values, cmap="bwr", label=column_name)
                ax_values.set_title('Values: %s'% (column_name))
                ax_values.grid();    ax_values.legend(loc='upper left')

                # Histograms
                #print('  Chan[%s]: %d cols  x  %d rows  ' % (column_name, len(df_tmp.columns), len(df_tmp)))
                ax_hist1 = axes[column_index, 1]

                df_subset = df_tmp[df_tmp[chan_name_trg]==0] #;  print('    ss %d cols  x  %d rows   ' % (len(df_subset.columns), len(df_subset)))
                df_subset = df_subset[column_name]

                df_subset.hist(ax=ax_hist1, alpha=histo_alpha, bins=histo_bins, orientation=u'horizontal', color='blue', label='0')

                df_subset = df_tmp[df_tmp[chan_name_trg]==1] #;  print('    ss %d cols  x  %d rows   ' % (len(df_subset.columns), len(df_subset)))
                df_subset = df_subset[column_name]

                df_subset.hist(ax=ax_hist1, alpha=histo_alpha, bins=histo_bins, orientation=u'horizontal', color='yellow', label='1')

                ax_hist1.set_title('Histogram %s'% (column_name))
                ax_hist1.legend()

            fig.set_size_inches(14, 10)
            plt.tight_layout(h_pad=0)   #plt.legend()
            plt.show(block=(is_last_plot and block))

            column_offset += columns_per_plot

    else:

        # Plot correlations with (non-logistic) target channel
        df_tmp = df.drop(chan_name_trg, axis=1)  ; df_tmp.index = df[chan_name_trg]
        #df_tmp = df.set_index(chan_name_target)

        column_index = 0
        while column_index < len(df_tmp.columns):

            is_last_plot = (column_index >= (len(df_tmp.columns) - columns_per_plot))
            #print('  Plotting %d .. %d    (last: %s)'
            #      % (column_index, column_index+columns_per_plot, is_last_plot))

            #df.ix[:,column_index:column_index + columns_per_plot].plot(subplots=True, style='.')
            df_tmp.ix[:,column_index:column_index + columns_per_plot].plot(subplots=True, style='.')

            fig = plt.gcf(); fig.set_size_inches(6, 10) ; plt.tight_layout(h_pad=0)
            # ; fig.suptitle('Correlations against '+ chan_name_target)
            # plt.legend(loc='best')
            plt.show(block=(is_last_plot and block))

            column_index += columns_per_plot

        #plt.close('all')


# -----------------------------------------------------------------
def plot_all_columns(df, columns_per_plot, block=False):
    '''
    Plot all columns, except index, in batches of 'columns_per_plot'
    '''

    column_offset = 0
    while column_offset < len(df.columns):

        is_last_plot = (column_offset >= (len(df.columns) - columns_per_plot))
        #print('  Plotting %d .. %d    (last: %s)'
        #      % (column_offset, column_offset+columns_per_plot, is_last_plot))

        # Plot selection of columns
        #df.ix[:,column_offset:column_offset + columns_per_plot].plot(subplots=True, style='.')
        df_tmp = df.iloc[:, column_offset:column_offset + columns_per_plot]
        df_tmp.plot(subplots=True)

        fig = plt.gcf(); fig.set_size_inches(14, 10) ; plt.tight_layout(h_pad=0)
        # ; fig.suptitle('Correlations against '+ chan_name_target)

        plt.show(block=(is_last_plot and block))

        column_offset += columns_per_plot

    #plt.close('all')


#-------------------------------------------------------------------
def compare_histograms(df1, df2, column_name_src):

    # Get worst-case max and min
    max_value = max(df1[column_name_src].max(), df1[column_name_src].max())
    min_value = min(df1[column_name_src].min(), df1[column_name_src].min())
    print("  compare_histograms()" )
    print("    Overall  min %.2f  .. max %.2f" % (min_value, max_value))
    range = (min_value, max_value)
    print("    Range: %s,%s" % range)

    # Build histograms
    num_bins = 50
    h1, bin_edges = np.histogram(df1[column_name_src], range=range, bins=num_bins, density=True)
    h2, bin_edges = np.histogram(df2[column_name_src], range=range, bins=num_bins, density=True)
    print('    Sums  h1: %.6f,    h2: %.6f' % (np.sum(h1), np.sum(h2)))
    print(h1) ; print(h2) ; plt.bar(np.arange(len(h1)), h1)  ; plt.bar(np.arange(len(h2)), h2)   ; plt.show()

    # Subtract
    h3 = h1 - h2
    print(h3)  ; plt.bar(np.arange(len(h3)), h3);    plt.show()
    print('    Diff:  %.6f,  %.6f,  sqrt: %.6f' % (np.sum(h3), np.sum(h3*h3), (np.sum(h3*h3) ** 0.5) ))

    # XXX Add some tests here to get expected ranges
    # Identical (scaled, offset) data
    #df1[column_name_src].hist(range=range, bins=num_bins) ;     df2[column_name_src].hist(range=range, bins=num_bins)


# -----------------------------------------------------------------
def do_plot_animate(dfs, num_to_show, interval_ms, plot_trace_lines):
    '''
    Plot timeseries df, channels: x,y,z,T
    x,y,z : coords
    T: target value
    
    markers=["o",">","x","<","D","p","^","s","v"]
    def rep(s, m):
        a, b = divmod(m, len(s))
        return s * a + s[:b]
    markers3=rep(markers,len(uniqueESNs))

    ax.scatter(... ,marker=markers3[i])
    '''
    num_frames = len(dfs[0])
    print('Plot and animate (%d rows,  x  %d channels)' % (len(dfs[0]), len(dfs[0].columns)))
    num_dataframes = len(dfs)
    sps = []
    num_to_show = num_to_show if num_to_show > 0 else num_frames

    # check all DF have usable dimensions
    assert dfs is not None
    assert dfs[0] is not None
    assert len(dfs[0].columns) == 4  # x,y,z and T
    for df in dfs:
        assert df.shape == dfs[0].shape
        
    def update_graph(i):
        
        # Calc ctart and en points
        i_start = 0 if i < num_to_show else (i - num_to_show + 1)  ; i_end = i + 1
        #i_start = i  ; i_end = i + num_to_show

        #print('  update_graph(%d  -> %d  %d)  col: %s' % (i, i_start, i_end, data[i_start:i_end, 2]))
        #print('  update_graph(%d of %d)  col: %s' % (i, num_frames, dfs[0][i_start:i_end, 2]))
        print('  update_graph(%d of %d) ' % (i, num_frames))

        for index in range(num_dataframes):
            
            data = dfs[index].values
            sp = sps[index]
            
            # Set points
            sp._offsets3d = data[i_start:i_end, 0], data[i_start:i_end, 1], data[i_start:i_end, 2]
            
            # Set colour
            sp.set_array(data[i_start:i_end, 3])
            
            # Plot trace lines
            if(plot_trace_lines and (i > 0) ):
                _ = ax.plot( [data[i_end-1, 0], data[i_end-2, 0]]
                        ,    [data[i_end-1, 1], data[i_end-2, 1]]
                        , zs=[data[i_end-1, 2], data[i_end-2, 2]]
                        , c='k', alpha=0.5, linestyle='dotted', linewidth=1)

                if(len(ax.lines) > (num_dataframes*(num_to_show-1)) ):
                    l = ax.lines.pop(0); del l      

        #scat3D.set_alpha(0.1)

        title.set_text('Frame %03d' % i)

    def on_press(event):
        nonlocal anim_running
        if event.key.isspace():
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True

    # Create plot
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    #fig.canvas.mpl_connect('button_press_event', onClick)
    fig.canvas.mpl_connect('key_press_event', on_press)

    ax = fig.add_subplot(111, projection='3d')
    
    title = ax.set_title('3D Test')
    #title = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # Fade grid
    ax.xaxis._axinfo["grid"]['color'] = "#f0f0f0"
    ax.yaxis._axinfo["grid"]['color'] = "#f0f0f0"
    ax.zaxis._axinfo["grid"]['color'] = "#f0f0f0"

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #cm = plt.cm.get_cmap('RdYlBu')
    #cm = plt.cm.get_cmap('gnuplot')
    #cm = plt.cm.get_cmap('bwr')
    cm = plt.cm.get_cmap('hot')
    for index in range(num_dataframes):

        data = dfs[index].values
        
        sp = ax.scatter([], [], [], cmap=cm, marker='o', linewidths=0.0, s=20)
        #sp = ax.scatter([], [], [], c='k', cmap=cm, marker='o', linewidths=0.0, s=20)
        sp.set_cmap(cm) # cmap argument above is ignored, so set it manually
        sp.set_array(data[:, 3])  # Allow prescaling of cmap  # XXX should be col 3
        sps.append(sp)
        

    # Animate
    anim_running = True
    #ani = animation.FuncAnimation(fig, update_graph, repeat=False, save_count=40, interval=100)
    #ani = animation.FuncAnimation(fig, update_graph, repeat=False, interval=1000, fargs=(plotted_line_queue))
    anim = animation.FuncAnimation(fig, update_graph, repeat=False
                , frames=num_frames, interval=interval_ms)

    plt.colorbar(sps[0]); # plt.colorbar(sp2)
    plt.show(block=True)


#-------------------------------------------------------------------
def plot_heatmap(df, column_name_x, column_name_y, column_name_colour, title=""):

    '''
    Plot heatmap of 2 channels, with colour dictated by third channel
    '''

    # Sanity checks

    # get values for colour, x and y
    vals_colour = df[column_name_colour]
    vals_x      = df[column_name_x]
    vals_y      = df[column_name_y]

    #cm = plt.cm.get_cmap(name='plasma', lut=None)
    # map = cm.get_cmap('Spectral') # Colour map (there are many others)
    # Colormap values are:
    # Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r
    # , Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges
    # , Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r
    # , PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r
    # , RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r
    # , Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b
    # , Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r
    # , YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r
    # , bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag
    # , flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar
    # , gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r
    # , gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r
    # , jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma
    # , plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring
    # , spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r
    # , terrain, terrain_r, viridis, viridis_r, winter, winter_r


    plt.tripcolor(vals_x, vals_y, vals_colour, cmap="hot")
    #plt.tricontour(vals_x, vals_y, vals_colour, 20)
    plt.plot(vals_x, vals_y, 'k. ')
    plt.xlabel(column_name_x) ; plt.ylabel(column_name_y)
    plt.colorbar()
    plt.title(title)

    plt.show(block=True)


#-------------------------------------------------------------------
def plot_confusion_matrix(cm,
                          target_names,  # eg ['True', 'False']
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Ref
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure() # ; plt.figure(figsize=(10,9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        colour_text = "black" if cm[i, j] > thresh else "white"
        text = "{:0.4f}".format(cm[i, j]) if normalize else "{:,}".format(cm[i, j])
        plt.text(j, i, text, horizontalalignment="center", color=colour_text)

    plt.tight_layout()
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values\naccuracy=%.4f; misclassification=%.4f'%(accuracy, misclass))

    plt.show(block=True)


#-------------------------------------------------------------------
def plot_value_and_volume(df, value_column_name, volume_column_name
                          , column_name_others_list=None, block=False):

    # Figure and axes
    plt.figure(figsize=(10, 5))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)

    # plots
    top.plot(df.index, df[value_column_name])  # df.index gives the dates
    if(column_name_others_list is not None):
        for column_name in column_name_others_list:
            top.plot(df.index, df[column_name])
        top.legend(loc='best')
    bottom.bar(df.index, df[volume_column_name])

    # Labels, titles
    top.grid(which='both')  # which='major', axis='both'
    top.axes.get_xaxis().set_visible(False)
    top.set_title('df[%s] + volume' % value_column_name)
    top.set_ylabel(value_column_name)
    bottom.set_ylabel(volume_column_name)
    bottom.grid(which='both')
    plt.sca(bottom); plt.xticks(rotation=45)

    plt.show(block=block)


#-------------------------------------------------------------------
def plot_residuals(df, col_name_trg, col_name_est):
    """
    Plot residuals (err)
    :param df:
    :param col_name_target:
    :param col_name_est:
    :return:
    """
    plt.plot(df[col_name_trg], df[col_name_trg], color='0.85')  # color = '0.75' color='grey'
    plt.scatter(df[col_name_trg], df[col_name_est], color='r')

    plt.title("Residuals ([%s] vs [%s])" % (col_name_trg, col_name_est))
    plt.xlabel(col_name_trg) ; plt.ylabel(col_name_est)
    #plt.set_xlabel(col_name_trg) ; plt.set_ylabel(col_name_est)

    #plt.legend(loc='best');
    plt.gcf().set_size_inches(5, 5);  plt.tight_layout();
    plt.show(block=True)



#-------------------------------------------------------------------
class SimpleUnitTest(unittest.TestCase):
    
    # Function run before each test
    def setUp(self):
        
        print('setUp()')
        
        
    # Function run after each test
    def tearDown(self):
        
        print('tearDown()')
    

    def _get_ts_data(self):

        columns = ['Open', 'High', 'Low', 'Close', 'Volume']    ; num_periods=100

        # Create data channels
        df = pd.DataFrame(np.random.randn(num_periods, len(columns)), columns=columns)

        # Ensure volume is large integer
        df['Volume'] = (abs(df['Volume'])) * 1e6  ; df['Volume'] = df['Volume'].astype(int)

        # Add datetime index
        date_today = datetime.now()
        df['Date'] = [date_today + timedelta(days=x) for x in range(num_periods)]
        df = df.set_index('Date')
        #print(df.head()); print(df.tail())

        return df, df.columns.values


    # Test 1
    def xtest_plot_value_hist(self):
        
        print('test_plot_value_hist()')
        
        df = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
        plot_value_and_hist(df, 'A')
        
    
    # Test 2
    def xtest_plot_animate(self):
        
        print('test_plot_animate()')
        
        df = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
        #df = pd.DataFrame(np.random.randint(0,10, size=(1000, 4)), columns=list('ABCD'))
        #df = pd.DataFrame(np.random.randn(0,10, size=(1000, 4)), columns=list('ABCD'))
        
        do_plot_animate([df], num_to_show=3, interval_ms=100, plot_trace_lines=True)

    # Test 3
    def xtest_plot_heatmap(self):

        print('test_plot_heatmap()')
        '''
        df1 = pd.DataFrame({
        'First': ['D','C','B','A','C','A','B','D','B','C'],
        'Second': ['E','E','C','D','D','E','E','B','A','A'],
        'Value': [1,2,3,4,5,6,7,8,9,10]})

        print(df1)
        df = df1.pivot('First','Second','Value')
        print(df)

        df1 = pd.DataFrame({
        'A': [0.14, 0.15, 0.17, 0.15, 0.11],
        'B': [1, 2, 3, 2, 3],
        'C': [10, 11, 9, 10, 13]})
        #plot_heatmap(df1)
        '''

        df = pd.read_csv('Data.test/correlation_values.csv')
        plot_heatmap(df, 'macd0_win_long', 'macd0_win_short', 'Corr', 'Correlations')



        #plot_heatmap(df, column_for_colour='A')

    # Test 4
    def xtest_plot_value_volume(self):
        print('test_plot_value_volume()')

        df, column_names = self._get_ts_data()

        #plot_tools.plot_value_and_volume(df_Mn, 'CLOSE', 'VOLUME')
        #plot_tools.plot_value_and_volume(df_Mn, 'CLOSE', 'VOLUME', ['F50_SAR_0.005_0.05_s1'])

        #other_chans = ['HIGH', 'LOW', 'F50_SAR_0.005_0.05_s1', 'F50_SAR_0.005_0.05_s0']
        #plot_tools.plot_value_and_volume(df_Mn.reset_index(), 'CLOSE', 'VOLUME', other_chans)

        other_chans = [column_names[2], column_names[3]]
        plot_value_and_volume(df, column_names[0], column_names[-1], other_chans)



    # Test 5
    def xtest_plot_all_columns(self):
        print('test_plot_all_columns()')

        df, column_names = self._get_ts_data()

        plot_all_columns(df, 3)


    # Test 6
    def test_plot_residuals(self):
        print('test_plot_residuals()')

        x = np.linspace(-1.2, 1.2, 20)
        y = np.sin(x)
        dy = (np.random.rand(20) - 0.5) * 0.5

        '''
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.scatter(x, y + dy)
        ax.vlines(x, y, y + dy)
        plt.show()
        '''

        #df = pd.DataFrame([[0.1, 2.5, 3], [0, 'bob', 1], [10, '-', 30]], columns=['A', 'B', 'C'])
        #df = pd.DataFrame([x, y, dy], columns=['A', 'B', 'C'])
        df = pd.DataFrame({'x' : x, 'y':y, 'y^':dy})
        print(df)
        plot_residuals(df, col_name_trg='y', col_name_est='y^')




#-------------------------------------------------------------------
if __name__ == "__main__":
    
    print('Running unit tests')
    print("Now is %s" % (datetime.today().strftime('%Y-%m-%d')))
    print("Now is %s" % (datetime.now().strftime("YYYYMMDD HH:mm:ss (%Y%m%d %H:%M:%S)")))

    # Unit tests class must be BEFORE this code
    unittest.main()



#-------------------------------------------------------------
# Unused
def plot_confusion_matrix_2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

