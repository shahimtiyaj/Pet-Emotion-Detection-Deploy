import pandas as pd
import numpy as np
from scipy import stats
import pickle
import tensorflow
from tensorflow import keras

from sklearn.preprocessing import StandardScaler 

def feature_acc(df):
    ax_list = []
    ay_list = []
    az_list = []

    gx_list = []
    gy_list = []
    gz_list = []

    mx_list = []
    my_list = []
    mz_list = []

    window_size = 10
    step_size = 5
#     print(df.shape)

    # creating overlaping windows of size window-size 100
    for i in range(0, df.shape[0] - window_size, step_size):
        ax = df['Acc_x'].values[i: i + 10]
        ay = df['Acc_y'].values[i: i + 10]
        az = df['Acc_z'].values[i: i + 10]

        gx = df['Gyro_x'].values[i: i + 10]
        gy = df['Gyro_y'].values[i: i + 10]
        gz = df['Gyro_z'].values[i: i + 10]

        mx = df['Mag_x'].values[i: i + 10]
        my = df['Mag_y'].values[i: i + 10]
        mz = df['Mag_z'].values[i: i + 10]

        ax_list.append(ax)
        ay_list.append(ay)
        az_list.append(az)

        gx_list.append(gx)
        gy_list.append(gy)
        gz_list.append(gz)

        mx_list.append(mx)
        my_list.append(my)
        mz_list.append(mz)
       
        X_train = pd.DataFrame()
  
    # mean
    X_train['Acc_x_mean'] = pd.Series(ax_list).apply(lambda x: x.mean())
    X_train['Acc_y_mean'] = pd.Series(ay_list).apply(lambda x: x.mean())
    X_train['Acc_z_mean'] = pd.Series(az_list).apply(lambda x: x.mean())

    # std dev
    X_train['Acc_x_std'] = pd.Series(ax_list).apply(lambda x: x.std())
    X_train['Acc_y_std'] = pd.Series(ay_list).apply(lambda x: x.std())
    X_train['Acc_z_std'] = pd.Series(az_list).apply(lambda x: x.std())

    # avg absolute diff
    X_train['Acc_x_aad'] = pd.Series(ax_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Acc_y_aad'] = pd.Series(ay_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Acc_z_aad'] = pd.Series(az_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['Acc_x_min'] = pd.Series(ax_list).apply(lambda x: x.min())
    X_train['Acc_y_min'] = pd.Series(ay_list).apply(lambda x: x.min())
    X_train['Acc_z_min'] = pd.Series(az_list).apply(lambda x: x.min())

    # max
    X_train['Acc_x_max'] = pd.Series(ax_list).apply(lambda x: x.max())
    X_train['Acc_y_max'] = pd.Series(ay_list).apply(lambda x: x.max())
    X_train['Acc_z_max'] = pd.Series(az_list).apply(lambda x: x.max())

    # max-min diff
    X_train['Acc_x_maxmin_diff'] = X_train['Acc_x_max'] - X_train['Acc_x_min']
    X_train['Acc_y_maxmin_diff'] = X_train['Acc_y_max'] - X_train['Acc_y_min']
    X_train['Acc_z_maxmin_diff'] = X_train['Acc_z_max'] - X_train['Acc_z_min']

    # median
    X_train['Acc_x_median'] = pd.Series(ax_list).apply(lambda x: np.median(x))
    X_train['Acc_y_median'] = pd.Series(ay_list).apply(lambda x: np.median(x))
    X_train['Acc_z_median'] = pd.Series(az_list).apply(lambda x: np.median(x))

    # median abs dev 
    X_train['Acc_x_mad'] = pd.Series(ax_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Acc_y_mad'] = pd.Series(ay_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Acc_z_mad'] = pd.Series(az_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    X_train['Acc_x_IQR'] = pd.Series(ax_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Acc_y_IQR'] = pd.Series(ay_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Acc_z_IQR'] = pd.Series(az_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # negtive count
    X_train['Acc_x_neg_count'] = pd.Series(ax_list).apply(lambda x: np.sum(x < 0))
    X_train['Acc_y_neg_count'] = pd.Series(ay_list).apply(lambda x: np.sum(x < 0))
    X_train['Acc_z_neg_count'] = pd.Series(az_list).apply(lambda x: np.sum(x < 0))

    # positive count
    X_train['Acc_x_pos_count'] = pd.Series(ax_list).apply(lambda x: np.sum(x > 0))
    X_train['Acc_y_pos_count'] = pd.Series(ay_list).apply(lambda x: np.sum(x > 0))
    X_train['Acc_z_pos_count'] = pd.Series(az_list).apply(lambda x: np.sum(x > 0))

    # values above mean
    X_train['Acc_x_above_mean'] = pd.Series(ax_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['Acc_y_above_mean'] = pd.Series(ay_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['Acc_z_above_mean'] = pd.Series(az_list).apply(lambda x: np.sum(x > x.mean()))

    # skewness
    X_train['Acc_x_skewness'] = pd.Series(ax_list).apply(lambda x: stats.skew(x))
    X_train['Acc_y_skewness'] = pd.Series(ay_list).apply(lambda x: stats.skew(x))
    X_train['Acc_z_skewness'] = pd.Series(az_list).apply(lambda x: stats.skew(x))

    # kurtosis
    X_train['Acc_x_kurtosis'] = pd.Series(ax_list).apply(lambda x: stats.kurtosis(x))
    X_train['Acc_y_kurtosis'] = pd.Series(ay_list).apply(lambda x: stats.kurtosis(x))
    X_train['z_kurtosis'] = pd.Series(az_list).apply(lambda x: stats.kurtosis(x))

    # # energy
    X_train['Acc_x_energy'] = pd.Series(ax_list).apply(lambda x: np.sum(x**2)/10)
    X_train['Acc_y_energy'] = pd.Series(ay_list).apply(lambda x: np.sum(x**2)/10)
    X_train['Acc_z_energy'] = pd.Series(az_list).apply(lambda x: np.sum(x**2/10))

    # signal magnitude area
    X_train['Acc_sma'] =    pd.Series(ax_list).apply(lambda x: np.sum(abs(x)/10)) + pd.Series(ay_list).apply(lambda x: np.sum(abs(x)/10)) \
                      + pd.Series(az_list).apply(lambda x: np.sum(abs(x)/10))

    #////////////////////////////  Gyrometer  ////////////////////////////////////////

    # mean
    X_train['Gyro_x_mean'] = pd.Series(gx_list).apply(lambda x: x.mean())
    X_train['Gyro_y_mean'] = pd.Series(gy_list).apply(lambda x: x.mean())
    X_train['Gyro_z_mean'] = pd.Series(gz_list).apply(lambda x: x.mean())

    # std dev
    X_train['Gyro_x_std'] = pd.Series(gx_list).apply(lambda x: x.std())
    X_train['Gyro_y_std'] = pd.Series(gy_list).apply(lambda x: x.std())
    X_train['Gyro_z_std'] = pd.Series(gz_list).apply(lambda x: x.std())

    # avg absolute diff
    X_train['Gyro_x_aad'] = pd.Series(gx_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Gyro_y_aad'] = pd.Series(gy_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Gyro_z_aad'] = pd.Series(gz_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['Gyro_x_min'] = pd.Series(gx_list).apply(lambda x: x.min())
    X_train['Gyro_y_min'] = pd.Series(gy_list).apply(lambda x: x.min())
    X_train['Gyro_z_min'] = pd.Series(gz_list).apply(lambda x: x.min())

    # max
    X_train['Gyro_x_max'] = pd.Series(gx_list).apply(lambda x: x.max())
    X_train['Gyro_y_max'] = pd.Series(gy_list).apply(lambda x: x.max())
    X_train['Gyro_z_max'] = pd.Series(gz_list).apply(lambda x: x.max())

    # max-min diff
    X_train['Gyro_x_maxmin_diff'] = X_train['Gyro_x_max'] - X_train['Gyro_x_min']
    X_train['Gyro_y_maxmin_diff'] = X_train['Gyro_y_max'] - X_train['Gyro_y_min']
    X_train['Gyro_z_maxmin_diff'] = X_train['Gyro_z_max'] - X_train['Gyro_z_min']

    # median
    X_train['Gyro_x_median'] = pd.Series(gx_list).apply(lambda x: np.median(x))
    X_train['Gyro_y_median'] = pd.Series(gy_list).apply(lambda x: np.median(x))
    X_train['Gyro_z_median'] = pd.Series(gz_list).apply(lambda x: np.median(x))

    # median abs dev 
    X_train['Gyro_x_mad'] = pd.Series(gx_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Gyro_y_mad'] = pd.Series(gy_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Gyro_z_mad'] = pd.Series(gz_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    X_train['Gyro_x_IQR'] = pd.Series(gx_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Gyro_y_IQR'] = pd.Series(gy_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Gyro_z_IQR'] = pd.Series(gz_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # negtive count
    X_train['Gyro_x_neg_count'] = pd.Series(gx_list).apply(lambda x: np.sum(x < 0))
    X_train['Gyro_y_neg_count'] = pd.Series(gy_list).apply(lambda x: np.sum(x < 0))
    X_train['Gyro_z_neg_count'] = pd.Series(gz_list).apply(lambda x: np.sum(x < 0))

    # positive count
    X_train['Gyro_x_pos_count'] = pd.Series(gx_list).apply(lambda x: np.sum(x > 0))
    X_train['Gyro_y_pos_count'] = pd.Series(gy_list).apply(lambda x: np.sum(x > 0))
    X_train['Gyro_z_pos_count'] = pd.Series(gz_list).apply(lambda x: np.sum(x > 0))

    # values above mean
    X_train['Gyro_x_above_mean'] = pd.Series(gx_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['Gyro_y_above_mean'] = pd.Series(gy_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['Gyro_z_above_mean'] = pd.Series(gz_list).apply(lambda x: np.sum(x > x.mean()))

    # skewness
    X_train['Gyro_x_skewness'] = pd.Series(gx_list).apply(lambda x: stats.skew(x))
    X_train['Gyro_y_skewness'] = pd.Series(gy_list).apply(lambda x: stats.skew(x))
    X_train['Gyro_z_skewness'] = pd.Series(gz_list).apply(lambda x: stats.skew(x))

    # kurtosis
    X_train['Gyro_x_kurtosis'] = pd.Series(gx_list).apply(lambda x: stats.kurtosis(x))
    X_train['Gyro_y_kurtosis'] = pd.Series(gy_list).apply(lambda x: stats.kurtosis(x))
    X_train['Gyro_z_kurtosis'] = pd.Series(gz_list).apply(lambda x: stats.kurtosis(x))

    # # energy
    X_train['Gyro_x_energy'] = pd.Series(gx_list).apply(lambda x: np.sum(x**2)/10)
    X_train['Gyro_y_energy'] = pd.Series(gy_list).apply(lambda x: np.sum(x**2)/10)
    X_train['Gyro_z_energy'] = pd.Series(gz_list).apply(lambda x: np.sum(x**2/10))

    # signal magnitude area
    X_train['Gyro_sma'] =    pd.Series(gx_list).apply(lambda x: np.sum(abs(x)/10)) + pd.Series(gy_list).apply(lambda x: np.sum(abs(x)/10)) \
                      + pd.Series(gz_list).apply(lambda x: np.sum(abs(x)/10))

    #//////////////////////////////////// Magnato meter//////////////////

    # mean
    X_train['Mag_x_mean'] = pd.Series(mx_list).apply(lambda x: x.mean())
    X_train['Mag_y_mean'] = pd.Series(my_list).apply(lambda x: x.mean())
    X_train['Mag_z_mean'] = pd.Series(mz_list).apply(lambda x: x.mean())

    # std dev
    X_train['Mag_x_std'] = pd.Series(mx_list).apply(lambda x: x.std())
    X_train['Mag_y_std'] = pd.Series(my_list).apply(lambda x: x.std())
    X_train['Mag_z_std'] = pd.Series(mz_list).apply(lambda x: x.std())

    # avg absolute diff
    X_train['Mag_x_aad'] = pd.Series(mx_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Mag_y_aad'] = pd.Series(my_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Mag_z_aad'] = pd.Series(mz_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['Mag_x_min'] = pd.Series(mx_list).apply(lambda x: x.min())
    X_train['Mag_y_min'] = pd.Series(my_list).apply(lambda x: x.min())
    X_train['Mag_z_min'] = pd.Series(mz_list).apply(lambda x: x.min())

    # max
    X_train['Mag_x_max'] = pd.Series(mx_list).apply(lambda x: x.max())
    X_train['Mag_y_max'] = pd.Series(my_list).apply(lambda x: x.max())
    X_train['Mag_z_max'] = pd.Series(mz_list).apply(lambda x: x.max())

    # max-min diff
    X_train['Mag_x_maxmin_diff'] = X_train['Mag_x_max'] - X_train['Mag_x_min']
    X_train['Mag_y_maxmin_diff'] = X_train['Mag_y_max'] - X_train['Mag_y_min']
    X_train['Mag_z_maxmin_diff'] = X_train['Mag_z_max'] - X_train['Mag_z_min']

    # median
    X_train['Mag_x_median'] = pd.Series(mx_list).apply(lambda x: np.median(x))
    X_train['Mag_y_median'] = pd.Series(my_list).apply(lambda x: np.median(x))
    X_train['Mag_z_median'] = pd.Series(mz_list).apply(lambda x: np.median(x))

    # median abs dev 
    X_train['Mag_x_mad'] = pd.Series(mx_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Mag_y_mad'] = pd.Series(my_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Mag_z_mad'] = pd.Series(mz_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    X_train['Mag_x_IQR'] = pd.Series(mx_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Mag_y_IQR'] = pd.Series(my_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Mag_z_IQR'] = pd.Series(mz_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # negtive count
    X_train['Mag_x_neg_count'] = pd.Series(mx_list).apply(lambda x: np.sum(x < 0))
    X_train['Mag_y_neg_count'] = pd.Series(my_list).apply(lambda x: np.sum(x < 0))
    X_train['Mag_z_neg_count'] = pd.Series(mz_list).apply(lambda x: np.sum(x < 0))

    # positive count
    X_train['Mag_x_pos_count'] = pd.Series(mx_list).apply(lambda x: np.sum(x > 0))
    X_train['Mag_y_pos_count'] = pd.Series(my_list).apply(lambda x: np.sum(x > 0))
    X_train['Mag_z_pos_count'] = pd.Series(mz_list).apply(lambda x: np.sum(x > 0))

    # values above mean
    X_train['Mag_x_above_mean'] = pd.Series(mx_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['Mag_y_above_mean'] = pd.Series(my_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['Mag_z_above_mean'] = pd.Series(mz_list).apply(lambda x: np.sum(x > x.mean()))

    # skewness
    X_train['Mag_x_skewness'] = pd.Series(mx_list).apply(lambda x: stats.skew(x))
    X_train['Mag_y_skewness'] = pd.Series(my_list).apply(lambda x: stats.skew(x))
    X_train['Mag_z_skewness'] = pd.Series(mz_list).apply(lambda x: stats.skew(x))

    # kurtosis
    X_train['Mag_x_kurtosis'] = pd.Series(mx_list).apply(lambda x: stats.kurtosis(x))
    X_train['Mag_y_kurtosis'] = pd.Series(my_list).apply(lambda x: stats.kurtosis(x))
    X_train['Mag_z_kurtosis'] = pd.Series(mz_list).apply(lambda x: stats.kurtosis(x))

    # # energy
    X_train['Mag_x_energy'] = pd.Series(mx_list).apply(lambda x: np.sum(x**2)/10)
    X_train['Mag_y_energy'] = pd.Series(my_list).apply(lambda x: np.sum(x**2)/10)
    X_train['Mag_z_energy'] = pd.Series(mz_list).apply(lambda x: np.sum(x**2/10))

    # signal magnitude area
    X_train['Mag_sma'] =    pd.Series(mx_list).apply(lambda x: np.sum(abs(x)/10)) + pd.Series(my_list).apply(lambda x: np.sum(abs(x)/10)) \
                      + pd.Series(mz_list).apply(lambda x: np.sum(abs(x)/10))
    
    ##/////////////////////////////////////FFT //////////////////////////////
    
    # converting the signals from time domain to frequency domain using FFT
    ax_list_fft = pd.Series(ax_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])
    ay_list_fft = pd.Series(ay_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])
    az_list_fft = pd.Series(az_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])

    gx_list_fft = pd.Series(gx_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])
    gy_list_fft = pd.Series(gy_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])
    gz_list_fft = pd.Series(gz_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])


    mx_list_fft = pd.Series(mx_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])
    my_list_fft = pd.Series(my_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])
    mz_list_fft = pd.Series(mz_list).apply(lambda x: np.abs(np.fft.fft(x))[1:11])


    # Statistical Features on raw x, y and z in frequency domain
    # FFT mean
    X_train['Acc_x_mean_fft'] = pd.Series(ax_list_fft).apply(lambda x: x.mean())
    X_train['Acc_y_mean_fft'] = pd.Series(ay_list_fft).apply(lambda x: x.mean())
    X_train['Acc_z_mean_fft'] = pd.Series(az_list_fft).apply(lambda x: x.mean())

    # FFT std dev
    X_train['Acc_x_std_fft'] = pd.Series(ax_list_fft).apply(lambda x: x.std())
    X_train['Acc_y_std_fft'] = pd.Series(ay_list_fft).apply(lambda x: x.std())
    X_train['Acc_z_std_fft'] = pd.Series(az_list_fft).apply(lambda x: x.std())

    # FFT avg absolute diff
    X_train['Acc_x_aad_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Acc_y_aad_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Acc_z_aad_fft'] = pd.Series(az_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # FFT min
    X_train['Acc_x_min_fft'] = pd.Series(ax_list_fft).apply(lambda x: x.min())
    X_train['Acc_y_min_fft'] = pd.Series(ay_list_fft).apply(lambda x: x.min())
    X_train['Acc_z_min_fft'] = pd.Series(az_list_fft).apply(lambda x: x.min())

    # FFT max
    X_train['Acc_x_max_fft'] = pd.Series(ax_list_fft).apply(lambda x: x.max())
    X_train['Acc_y_max_fft'] = pd.Series(ay_list_fft).apply(lambda x: x.max())
    X_train['Acc_z_max_fft'] = pd.Series(az_list_fft).apply(lambda x: x.max())

    # FFT max-min diff
    X_train['Acc_x_maxmin_diff_fft'] = X_train['Acc_x_max_fft'] - X_train['Acc_x_min_fft']
    X_train['Acc_y_maxmin_diff_fft'] = X_train['Acc_y_max_fft'] - X_train['Acc_y_min_fft']
    X_train['Acc_z_maxmin_diff_fft'] = X_train['Acc_z_max_fft'] - X_train['Acc_z_min_fft']

    # FFT median
    X_train['Acc_x_median_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.median(x))
    X_train['Acc_y_median_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.median(x))
    X_train['Acc_z_median_fft'] = pd.Series(az_list_fft).apply(lambda x: np.median(x))

    # FFT median abs dev 
    X_train['Acc_x_mad_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Acc_y_mad_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Acc_z_mad_fft'] = pd.Series(az_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # FFT Interquartile range
    X_train['Acc_x_IQR_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Acc_y_IQR_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Acc_z_IQR_fft'] = pd.Series(az_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # FFT values above mean
    X_train['Acc_x_above_mean_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['Acc_y_above_mean_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['Acc_z_above_mean_fft'] = pd.Series(az_list_fft).apply(lambda x: np.sum(x > x.mean()))

    # # FFT number of peaks
    # X_train['x_peak_count_fft'] = pd.Series(ax_list_fft).apply(lambda x: len(find_peaks(x)[0]))
    # X_train['y_peak_count_fft'] = pd.Series(ay_list_fft).apply(lambda x: len(find_peaks(x)[0]))
    # X_train['z_peak_count_fft'] = pd.Series(az_list_fft).apply(lambda x: len(find_peaks(x)[0]))

    # FFT skewness
    X_train['Acc_x_skewness_fft'] = pd.Series(ax_list_fft).apply(lambda x: stats.skew(x))
    X_train['Acc_y_skewness_fft'] = pd.Series(ay_list_fft).apply(lambda x: stats.skew(x))
    X_train['Acc_z_skewness_fft'] = pd.Series(az_list_fft).apply(lambda x: stats.skew(x))

    # FFT kurtosis
    X_train['Acc_x_kurtosis_fft'] = pd.Series(ax_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['Acc_y_kurtosis_fft'] = pd.Series(ay_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['Acc_z_kurtosis_fft'] = pd.Series(az_list_fft).apply(lambda x: stats.kurtosis(x))

    # FFT energy
    X_train['Acc_x_energy_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.sum(x**2)/10)
    X_train['Acc_y_energy_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.sum(x**2)/10)
    X_train['Acc_z_energy_fft'] = pd.Series(az_list_fft).apply(lambda x: np.sum(x**2/10))

    # FFT Signal magnitude area
    X_train['Acc_sma_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.sum(abs(x)/10)) + pd.Series(ay_list_fft).apply(lambda x: np.sum(abs(x)/10)) \
                         + pd.Series(az_list_fft).apply(lambda x: np.sum(abs(x)/10))

    #/////////////////////////////// Gyro /////////////////////////////////////


    # FFT mean
    X_train['Gyro_x_mean_fft'] = pd.Series(gx_list_fft).apply(lambda x: x.mean())
    X_train['Gyro_y_mean_fft'] = pd.Series(gy_list_fft).apply(lambda x: x.mean())
    X_train['Gyro_z_mean_fft'] = pd.Series(gz_list_fft).apply(lambda x: x.mean())

    # FFT std dev
    X_train['Gyro_x_std_fft'] = pd.Series(gx_list_fft).apply(lambda x: x.std())
    X_train['Gyro_y_std_fft'] = pd.Series(gy_list_fft).apply(lambda x: x.std())
    X_train['Gyro_z_std_fft'] = pd.Series(gz_list_fft).apply(lambda x: x.std())

    # FFT avg absolute diff
    X_train['Gyro_x_aad_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Gyro_y_aad_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Gyro_z_aad_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # FFT min
    X_train['Gyro_x_min_fft'] = pd.Series(gx_list_fft).apply(lambda x: x.min())
    X_train['Gyro_y_min_fft'] = pd.Series(gy_list_fft).apply(lambda x: x.min())
    X_train['Gyro_z_min_fft'] = pd.Series(gz_list_fft).apply(lambda x: x.min())

    # FFT max
    X_train['Gyro_x_max_fft'] = pd.Series(gx_list_fft).apply(lambda x: x.max())
    X_train['Gyro_y_max_fft'] = pd.Series(gy_list_fft).apply(lambda x: x.max())
    X_train['Gyro_z_max_fft'] = pd.Series(gz_list_fft).apply(lambda x: x.max())

    # FFT max-min diff
    X_train['Gyro_x_maxmin_diff_fft'] = X_train['Gyro_x_max_fft'] - X_train['Gyro_x_min_fft']
    X_train['Gyro_y_maxmin_diff_fft'] = X_train['Gyro_y_max_fft'] - X_train['Gyro_y_min_fft']
    X_train['Gyro_z_maxmin_diff_fft'] = X_train['Gyro_z_max_fft'] - X_train['Gyro_z_min_fft']

    # FFT median
    X_train['Gyro_x_median_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.median(x))
    X_train['Gyro_y_median_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.median(x))
    X_train['Gyro_z_median_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.median(x))

    # FFT median abs dev 
    X_train['Gyro_x_mad_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Gyro_y_mad_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Gyro_z_mad_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # FFT Interquartile range
    X_train['Gyro_x_IQR_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Gyro_y_IQR_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Gyro_z_IQR_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # FFT values above mean
    X_train['Gyro_x_above_mean_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['Gyro_y_above_mean_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['Gyro_z_above_mean_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.sum(x > x.mean()))

    # FFT skewness
    X_train['Gyro_x_skewness_fft'] = pd.Series(gx_list_fft).apply(lambda x: stats.skew(x))
    X_train['Gyro_y_skewness_fft'] = pd.Series(gy_list_fft).apply(lambda x: stats.skew(x))
    X_train['Gyro_z_skewness_fft'] = pd.Series(gz_list_fft).apply(lambda x: stats.skew(x))

    # FFT kurtosis
    X_train['Gyro_x_kurtosis_fft'] = pd.Series(gx_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['Gyro_y_kurtosis_fft'] = pd.Series(gy_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['Gyro_z_kurtosis_fft'] = pd.Series(gz_list_fft).apply(lambda x: stats.kurtosis(x))

    # FFT energy
    X_train['Gyro_x_energy_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.sum(x**2)/10)
    X_train['Gyro_y_energy_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.sum(x**2)/10)
    X_train['Gyro_z_energy_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.sum(x**2/10))

    # FFT Signal magnitude area
    X_train['Gyro_sma_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.sum(abs(x)/10)) + pd.Series(gy_list_fft).apply(lambda x: np.sum(abs(x)/10)) \
                         + pd.Series(gz_list_fft).apply(lambda x: np.sum(abs(x)/10))

    #/////////////////////////// Magnato meter ///////////////////////


    ##FFT mean
    X_train['Mag_x_mean_fft'] = pd.Series(mx_list_fft).apply(lambda x: x.mean())
    X_train['Mag_y_mean_fft'] = pd.Series(my_list_fft).apply(lambda x: x.mean())
    X_train['Mag_z_mean_fft'] = pd.Series(mz_list_fft).apply(lambda x: x.mean())

    # FFT std dev
    X_train['Mag_x_std_fft'] = pd.Series(mx_list_fft).apply(lambda x: x.std())
    X_train['Mag_y_std_fft'] = pd.Series(my_list_fft).apply(lambda x: x.std())
    X_train['Mag_z_std_fft'] = pd.Series(mz_list_fft).apply(lambda x: x.std())

    # FFT avg absolute diff
    X_train['Mag_x_aad_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Mag_y_aad_fft'] = pd.Series(my_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['Mag_z_aad_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # FFT min
    X_train['Mag_x_min_fft'] = pd.Series(mx_list_fft).apply(lambda x: x.min())
    X_train['Mag_y_min_fft'] = pd.Series(my_list_fft).apply(lambda x: x.min())
    X_train['Mag_z_min_fft'] = pd.Series(mz_list_fft).apply(lambda x: x.min())

    # FFT max
    X_train['Mag_x_max_fft'] = pd.Series(mx_list_fft).apply(lambda x: x.max())
    X_train['Mag_y_max_fft'] = pd.Series(my_list_fft).apply(lambda x: x.max())
    X_train['Mag_z_max_fft'] = pd.Series(mz_list_fft).apply(lambda x: x.max())

    # FFT max-min diff
    X_train['Mag_x_maxmin_diff_fft'] = X_train['Mag_x_max_fft'] - X_train['Mag_x_min_fft']
    X_train['Mag_y_maxmin_diff_fft'] = X_train['Mag_y_max_fft'] - X_train['Mag_y_min_fft']
    X_train['Mag_z_maxmin_diff_fft'] = X_train['Mag_z_max_fft'] - X_train['Mag_z_min_fft']

    # FFT median
    X_train['Mag_x_median_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.median(x))
    X_train['Mag_y_median_fft'] = pd.Series(my_list_fft).apply(lambda x: np.median(x))
    X_train['Mag_z_median_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.median(x))

    # FFT median abs dev 
    X_train['Mag_x_mad_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Mag_y_mad_fft'] = pd.Series(my_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['Mag_z_mad_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # FFT Interquartile range
    X_train['Mag_x_IQR_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Mag_y_IQR_fft'] = pd.Series(my_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['Mag_z_IQR_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # FFT values above mean
    X_train['Mag_x_above_mean_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['Mag_y_above_mean_fft'] = pd.Series(my_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['Mag_z_above_mean_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.sum(x > x.mean()))

    # FFT skewness
    X_train['Mag_x_skewness_fft'] = pd.Series(mx_list_fft).apply(lambda x: stats.skew(x))
    X_train['Mag_y_skewness_fft'] = pd.Series(my_list_fft).apply(lambda x: stats.skew(x))
    X_train['Mag_z_skewness_fft'] = pd.Series(mz_list_fft).apply(lambda x: stats.skew(x))

    # FFT kurtosis
    X_train['Mag_x_kurtosis_fft'] = pd.Series(mx_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['Mag_y_kurtosis_fft'] = pd.Series(my_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['Mag_z_kurtosis_fft'] = pd.Series(mz_list_fft).apply(lambda x: stats.kurtosis(x))

    # FFT energy
    X_train['Mag_x_energy_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.sum(x**2)/10)
    X_train['Mag_y_energy_fft'] = pd.Series(my_list_fft).apply(lambda x: np.sum(x**2)/10)
    X_train['Mag_z_energy_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.sum(x**2/10))

    # FFT Signal magnitude area
    X_train['Mag_sma_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.sum(abs(x)/10)) + pd.Series(my_list_fft).apply(lambda x: np.sum(abs(x)/10)) \
                         + pd.Series(mz_list_fft).apply(lambda x: np.sum(abs(x)/10))
    
    # Max Indices and Min indices 

    # index of max value in time domain
    X_train['Acc_x_argmax'] = pd.Series(ax_list).apply(lambda x: np.argmax(x))
    X_train['Acc_y_argmax'] = pd.Series(ay_list).apply(lambda x: np.argmax(x))
    X_train['Acc_z_argmax'] = pd.Series(az_list).apply(lambda x: np.argmax(x))

    # index of min value in time domain
    X_train['Acc_x_argmin'] = pd.Series(ax_list).apply(lambda x: np.argmin(x))
    X_train['Acc_y_argmin'] = pd.Series(ay_list).apply(lambda x: np.argmin(x))
    X_train['Acc_z_argmin'] = pd.Series(az_list).apply(lambda x: np.argmin(x))

    # absolute difference between above indices
    X_train['Acc_x_arg_diff'] = abs(X_train['Acc_x_argmax'] - X_train['Acc_x_argmin'])
    X_train['Acc_y_arg_diff'] = abs(X_train['Acc_y_argmax'] - X_train['Acc_y_argmin'])
    X_train['Acc_z_arg_diff'] = abs(X_train['Acc_z_argmax'] - X_train['Acc_z_argmin'])

    # index of max value in frequency domain
    X_train['Acc_x_argmax_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))
    X_train['Acc_y_argmax_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))
    X_train['Acc_z_argmax_fft'] = pd.Series(az_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))

    #index of min value in frequency domain3
    X_train['Acc_x_argmin_fft'] = pd.Series(ax_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))
    X_train['Acc_y_argmin_fft'] = pd.Series(ay_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))
    X_train['Acc_z_argmin_fft'] = pd.Series(az_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))

    # absolute difference between above indices
    X_train['Acc_x_arg_diff_fft'] = abs(X_train['Acc_x_argmax_fft'] - X_train['Acc_x_argmin_fft'])
    X_train['Acc_y_arg_diff_fft'] = abs(X_train['Acc_y_argmax_fft'] - X_train['Acc_y_argmin_fft'])
    X_train['Acc_z_arg_diff_fft'] = abs(X_train['Acc_z_argmax_fft'] - X_train['Acc_z_argmin_fft'])


    #/////////////////////////// Gyro /////////////////////////

    # index of max value in time domain
    X_train['Gyro_x_argmax'] = pd.Series(gx_list).apply(lambda x: np.argmax(x))
    X_train['Gyro_y_argmax'] = pd.Series(gy_list).apply(lambda x: np.argmax(x))
    X_train['Gyro_z_argmax'] = pd.Series(gz_list).apply(lambda x: np.argmax(x))

    # index of min value in time domain
    X_train['Gyro_x_argmin'] = pd.Series(gx_list).apply(lambda x: np.argmin(x))
    X_train['Gyro_y_argmin'] = pd.Series(gy_list).apply(lambda x: np.argmin(x))
    X_train['Gyro_z_argmin'] = pd.Series(gz_list).apply(lambda x: np.argmin(x))

    # absolute difference between above indices
    X_train['Gyro_x_arg_diff'] = abs(X_train['Gyro_x_argmax'] - X_train['Gyro_x_argmin'])
    X_train['Gyro_y_arg_diff'] = abs(X_train['Gyro_y_argmax'] - X_train['Gyro_y_argmin'])
    X_train['Gyro_z_arg_diff'] = abs(X_train['Gyro_z_argmax'] - X_train['Gyro_z_argmin'])

    # index of max value in frequency domain
    X_train['Gyro_x_argmax_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))
    X_train['Gyro_y_argmax_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))
    X_train['Gyro_z_argmax_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))

    # index of min value in frequency domain
    X_train['Gyro_x_argmin_fft'] = pd.Series(gx_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))
    X_train['Gyro_y_argmin_fft'] = pd.Series(gy_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))
    X_train['Gyro_z_argmin_fft'] = pd.Series(gz_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))

    # absolute difference between above indices
    X_train['Gyro_x_arg_diff_fft'] = abs(X_train['Gyro_x_argmax_fft'] - X_train['Gyro_x_argmin_fft'])
    X_train['Gyro_y_arg_diff_fft'] = abs(X_train['Gyro_y_argmax_fft'] - X_train['Gyro_y_argmin_fft'])
    X_train['Gyro_z_arg_diff_fft'] = abs(X_train['Gyro_z_argmax_fft'] - X_train['Gyro_z_argmin_fft'])


    #////////////////////////////// Magnato /////////////////////////

    # index of max value in time domain
    X_train['Mag_x_argmax'] = pd.Series(mx_list).apply(lambda x: np.argmax(x))
    X_train['Mag_y_argmax'] = pd.Series(my_list).apply(lambda x: np.argmax(x))
    X_train['Mag_z_argmax'] = pd.Series(mz_list).apply(lambda x: np.argmax(x))

    # index of min value in time domain
    X_train['Mag_x_argmin'] = pd.Series(mx_list).apply(lambda x: np.argmin(x))
    X_train['Mag_y_argmin'] = pd.Series(my_list).apply(lambda x: np.argmin(x))
    X_train['Mag_z_argmin'] = pd.Series(mz_list).apply(lambda x: np.argmin(x))

    # absolute difference between above indices
    X_train['Mag_x_arg_diff'] = abs(X_train['Mag_x_argmax'] - X_train['Mag_x_argmin'])
    X_train['Mag_y_arg_diff'] = abs(X_train['Mag_y_argmax'] - X_train['Mag_y_argmin'])
    X_train['Mag_z_arg_diff'] = abs(X_train['Mag_z_argmax'] - X_train['Mag_z_argmin'])

    # index of max value in frequency domain
    X_train['Mag_x_argmax_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))
    X_train['Mag_y_argmax_fft'] = pd.Series(my_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))
    X_train['Mag_z_argmax_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:11]))

    # index of min value in frequency domain
    X_train['Mag_x_argmin_fft'] = pd.Series(mx_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))
    X_train['Mag_y_argmin_fft'] = pd.Series(my_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))
    X_train['Mag_z_argmin_fft'] = pd.Series(mz_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:11]))

    # absolute difference between above indices
    X_train['Mag_x_arg_diff_fft'] = abs(X_train['Mag_x_argmax_fft'] - X_train['Mag_x_argmin_fft'])
    X_train['Mag_y_arg_diff_fft'] = abs(X_train['Mag_y_argmax_fft'] - X_train['Mag_y_argmin_fft'])
    X_train['Mag_z_arg_diff_fft'] = abs(X_train['Mag_z_argmax_fft'] - X_train['Mag_z_argmin_fft'])


    return X_train