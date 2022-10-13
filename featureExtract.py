import scipy.stats as stats
import numpy as np

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 81

    frames = []
    labels = []

    for i in range(0, len(df) - frame_size, hop_size):
      
        N_AccX = df['Acc_x'].values[i: i + frame_size]
        N_AccY = df['Acc_y'].values[i: i + frame_size]
        N_AccZ = df['Acc_z'].values[i: i + frame_size]
        N_GyroX = df['Gyro_x'].values[i: i + frame_size]
        N_GyroY = df['Gyro_y'].values[i: i + frame_size]
        N_GyroZ = df['Gyro_z'].values[i: i + frame_size]
        T_AccX = df['Mag_x'].values[i: i + frame_size]
        T_AccY = df['Mag_y'].values[i: i + frame_size]
        T_AccZ = df['Mag_z'].values[i: i + frame_size]
 
        roll_std_NAccX = df['roll_mean_AccX'].values[i: i + frame_size]
        roll_IQR_TGyroX = df['roll_mean_AccY'].values[i: i + frame_size]
        roll_EME_TGyroZ = df['roll_mean_NAccZ'].values[i: i + frame_size]
        roll_EME_TGyroY = df['roll_std_AccX'].values[i: i + frame_size]
        roll_skew_NGyroZ = df['roll_std_AccY'].values[i: i + frame_size]
        roll_max_NAccX = df['roll_std_NAccZ'].values[i: i + frame_size]
        roll_mad_NGyroZ = df['roll_min_AccX'].values[i: i + frame_size]
        roll_mean_NGyroZ = df['roll_min_NAccY'].values[i: i + frame_size]
        roll_max_NGyroX = df['roll_min_NAccZ'].values[i: i + frame_size]
        roll_mean_NAccY = df['roll_max_AccX'].values[i: i + frame_size]
        roll_mean_NAccX = df['roll_max_NAccY'].values[i: i + frame_size]
        roll_mad_NAccY = df['roll_max_NAccZ'].values[i: i + frame_size]
        roll_EME_TAccX = df['roll_EME_AccX'].values[i: i + frame_size]
        roll_kurt_NAccX = df['roll_EME_NAccY'].values[i: i + frame_size]
        roll_EME_TAccZ = df['roll_EME_NAccZ'].values[i: i + frame_size]
        roll_min_TGyroX = df['roll_IQR_AccX'].values[i: i + frame_size]
        roll_IQR_NAccZ = df['roll_IQR_NAccY'].values[i: i + frame_size]
        roll_skew_NAccZ = df['roll_IQR_NAccZ'].values[i: i + frame_size]
        roll_IQR_TAccX = df['Acc-SMA'].values[i: i + frame_size]
        roll_kurt_NGyroY = df['roll_skew_AccX'].values[i: i + frame_size]
        roll_mean_TAccZ = df['roll_skew_NAccY'].values[i: i + frame_size]
        roll_IQR_TAccZ = df['roll_skew_NAccZ'].values[i: i + frame_size]
        roll_mad_TAccY = df['roll_mean_NGyroX'].values[i: i + frame_size]
        roll_min_NAccY = df['roll_mean_NGyroY'].values[i: i + frame_size]
        roll_mean_TGyroX = df['roll_mean_NGyroZ'].values[i: i + frame_size]
        roll_std_TGyroZ = df['roll_std_NGyroX'].values[i: i + frame_size]
        roll_std_NGyroY = df['roll_std_NGyroY'].values[i: i + frame_size]
        roll_min_NAccX = df['roll_std_NGyroZ'].values[i: i + frame_size]
        roll_mad_TAccX = df['roll_mad_NGyroX'].values[i: i + frame_size]
        roll_IQR_TAccY = df['roll_mad_NGyroY'].values[i: i + frame_size]
        roll_min_TAccZ = df['roll_mad_NGyroZ'].values[i: i + frame_size]
        roll_max_NGyroZ = df['roll_min_NGyroX'].values[i: i + frame_size]
        roll_min_NAccZ = df['roll_min_NGyroY'].values[i: i + frame_size]
        roll_kurt_NGyroZ = df['roll_min_NGyroZ'].values[i: i + frame_size]
        roll_mad_NGyroY = df['roll_max_NGyroX'].values[i: i + frame_size]
        roll_max_NAccY = df['roll_max_NGyroY'].values[i: i + frame_size]
        roll_min_TAccX = df['roll_max_NGyroZ'].values[i: i + frame_size]
        roll_mean_TGyroZ = df['roll_EME_NGyroX'].values[i: i + frame_size]
        roll_mad_TGyroZ = df['roll_EME_NGyroY'].values[i: i + frame_size]
        roll_EME_NGyroZ = df['roll_EME_NGyroZ'].values[i: i + frame_size]
        roll_IQR_NAccX = df['roll_IQR_NGyroX'].values[i: i + frame_size]
        roll_min_TGyroY = df['roll_IQR_NGyroY'].values[i: i + frame_size]
        N_Acc_SMA = df['roll_IQR_NGyroZ'].values[i: i + frame_size]
        roll_EME_TGyroX = df['N-Gyro-SMA'].values[i: i + frame_size]
        roll_mad_NGyroX = df['roll_skew_NGyroX'].values[i: i + frame_size]
        roll_mad_TGyroX = df['roll_skew_NGyroY'].values[i: i + frame_size]
        roll_IQR_TGyroZ = df['roll_skew_NGyroZ'].values[i: i + frame_size]
        roll_max_TGyroY = df['roll_mean_TAccX'].values[i: i + frame_size]
        roll_kurt_NGyroX = df['roll_mean_TAccY'].values[i: i + frame_size]
        roll_min_TGyroZ = df['roll_mean_TAccZ'].values[i: i + frame_size]
        T_Acc_SMA = df['roll_std_TAccX'].values[i: i + frame_size]
        roll_max_NAccZ = df['roll_std_TAccY'].values[i: i + frame_size]
        roll_kurt_NAccY = df['roll_std_TAccZ'].values[i: i + frame_size]
        roll_mad_NAccX = df['roll_mad_TAccX'].values[i: i + frame_size]
        roll_skew_NGyroX = df['roll_mad_TAccY'].values[i: i + frame_size]
        roll_mad_TGyroY = df['roll_mad_TAccZ'].values[i: i + frame_size]
        roll_std_NAccZ = df['roll_min_TAccX'].values[i: i + frame_size]
        roll_std_TAccZ = df['roll_min_TAccY'].values[i: i + frame_size]
        roll_mad_TAccZ = df['roll_min_TAccZ'].values[i: i + frame_size]
        roll_std_TGyroY = df['roll_max_TAccX'].values[i: i + frame_size]
        roll_max_TAccX = df['roll_max_TAccY'].values[i: i + frame_size]
        roll_min_NGyroY = df['roll_max_TAccZ'].values[i: i + frame_size]
        roll_skew_TGyroZ = df['roll_EME_TAccX'].values[i: i + frame_size]
        roll_std_NGyroZ = df['roll_EME_TAccY'].values[i: i + frame_size]
        roll_mean_NGyroX = df['roll_EME_TAccZ'].values[i: i + frame_size]
        roll_IQR_NGyroX = df['roll_IQR_TAccX'].values[i: i + frame_size]
        roll_mean_TAccY = df['roll_IQR_TAccY'].values[i: i + frame_size]
        roll_max_TGyroZ = df['roll_IQR_TAccZ'].values[i: i + frame_size]
        roll_min_NGyroZ = df['T-Acc-SMA'].values[i: i + frame_size]
        roll_EME_NGyroX = df['roll_skew_TAccX'].values[i: i + frame_size]
        roll_skew_TGyroX = df['roll_skew_TAccY'].values[i: i + frame_size]
        roll_std_TGyroX = df['roll_skew_TAccZ'].values[i: i + frame_size]
        
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([N_AccX,N_AccY,N_AccZ,N_GyroX,N_GyroY,N_GyroZ,T_AccX,T_AccY,T_AccZ,roll_std_NAccX,roll_IQR_TGyroX,roll_EME_TGyroZ,roll_EME_TGyroY,roll_skew_NGyroZ, 
                       roll_max_NAccX,roll_mad_NGyroZ,roll_mean_NGyroZ,roll_max_NGyroX,roll_mean_NAccY,
                       roll_mean_NAccX,roll_mad_NAccY,roll_EME_TAccX,roll_kurt_NAccX,roll_EME_TAccZ,
                       roll_min_TGyroX,roll_IQR_NAccZ,roll_skew_NAccZ,roll_IQR_TAccX,roll_kurt_NGyroY,
                       roll_mean_TAccZ,roll_IQR_TAccZ,roll_mad_TAccY,roll_min_NAccY,roll_mean_TGyroX,
                       roll_std_TGyroZ,roll_std_NGyroY,roll_min_NAccX,roll_mad_TAccX,roll_IQR_TAccY,
                       roll_min_TAccZ,roll_max_NGyroZ,roll_min_NAccZ,roll_kurt_NGyroZ,roll_mad_NGyroY,
                       roll_max_NAccY,roll_min_TAccX,roll_mean_TGyroZ,roll_mad_TGyroZ,roll_EME_NGyroZ,
                       roll_IQR_NAccX,roll_min_TGyroY,N_Acc_SMA,roll_EME_TGyroX,roll_mad_NGyroX,
                       roll_mad_TGyroX,roll_IQR_TGyroZ,roll_max_TGyroY,roll_kurt_NGyroX,
                       roll_min_TGyroZ,T_Acc_SMA,roll_max_NAccZ,roll_kurt_NAccY,roll_mad_NAccX,
                       roll_skew_NGyroX,roll_mad_TGyroY,roll_std_NAccZ,roll_std_TAccZ,roll_mad_TAccZ,
                       roll_std_TGyroY,roll_max_TAccX,roll_min_NGyroY,roll_skew_TGyroZ,roll_std_NGyroZ,
                       roll_mean_NGyroX,roll_IQR_NGyroX,roll_mean_TAccY,roll_max_TGyroZ,roll_min_NGyroZ,
                       roll_EME_NGyroX,roll_skew_TGyroX,roll_std_TGyroX])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels