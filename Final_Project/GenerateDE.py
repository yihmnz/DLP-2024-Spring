import copy
import _pickle as cPickle
import os
import numpy as np
import os.path as osp
import h5py
import scipy.signal
import math
from tqdm import tqdm
# Function of DE from STFT
def calculate_de_from_stft(f, Zxx, fStart, fEnd):
    '''
    Compute DE from STFT
    --------
    input:  f       frequency array from STFT
            Zxx     STFT result (trials, n electrodes, k frequency bands,  m time points)
            fStart  start frequency of each frequency band
            fEnd    end frequency of each frequency band
    output: de      DE [n*l*k] n electrodes, l windows, k frequency bands
    '''
    # Initialize parameters
    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    
    for i in range(len(fStart)): # five frequnecy band
        fStartNum[i] = np.searchsorted(f, fStart[i])
        fEndNum[i] = np.searchsorted(f, fEnd[i])

    trials, n, m, k = Zxx.shape[0], Zxx.shape[1], Zxx.shape[3], len(fStart)
    de = np.zeros([trials, n, m, k])

    for tt in range(trials):
        for j in range(n):
            for l in range(m):
                magFFTdata = np.abs(Zxx[tt, j, :, l])
                for p in range(k):
                    E = 0
                    for p0 in range(fStartNum[p], fEndNum[p]):
                        E += magFFTdata[p0] * magFFTdata[p0]
                    E /= (fEndNum[p] - fStartNum[p] + 1)
                    de[tt, j, l, p] = math.log(100 * E, 2)
    return de

# Sliding window epochs
def sliding_windows_generator(data, label, type, time_length=256, step_size=16, batch_size=18600):
    if type == 'A':
        label = label[:, 1]
    elif type == 'V':
        label = label[:, 0]
    elif type == 'L':
        label = label[:, 3]
    label = np.where(label <= 5, 0, label)
    label = np.where(label > 5, 1, label)
    print('Binary label generated!')

    length = data.shape  # (video, channel, freq, time)
    print(data.shape)
    num_windows = (length[2] - time_length) // step_size + 1
    print(num_windows*40)
    while True:
        Data = []
        Label = []

        for video in range(length[0]):
            start = 0
            while start + time_length <= length[2]:
                Data.append(data[video, :, start:start + time_length, :])
                Label.append(label[video])
                start += step_size

                if len(Data) == batch_size:
                    return np.array(Data), np.array(Label)

# Main processing function
def main(sub):
    datapath = '/mnt/left/home/2023/angelina/DLCourse/Final/Emotion_intensity/datasets'
    print(sub)            
    sub_code = f's0{sub}.dat' if sub < 10 else f's{sub}.dat'
    subject_path = os.path.join(datapath, sub_code)
    subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
    labels = subject['labels']
    data = subject['data'][:, 0:32, 3*128:] # video, channel, time (40, 32, 7680)
    print('Original Datashape: ', data.shape)

    # STFT
    f, t, Zxx = scipy.signal.stft(data[:,:,:], fs=128, window='hann', nperseg=256,
                                    noverlap=255, nfft=None, detrend=False, 
                                    return_onesided=True, boundary='zeros', 
                                    padded=True, axis=-1, scaling='spectrum')
    print('STFT DataShape: ', Zxx.shape) #video, channel, freq resolution, time(40, 32, 129, 7681)

    # Set frequency boundary
    fStart = [0, 4, 8, 16, 22]
    fEnd = [4, 8, 16, 22, 40]

    # Calculate DE
    DifEnt = calculate_de_from_stft(f, Zxx, fStart, fEnd) 
    print('DE DataShape: ', DifEnt.shape) # DE DataShape:  (40, 32, 7681, 5)

    # Epoch
    d_a, l_a = sliding_windows_generator(DifEnt, labels, 'A')
    print('DataShape: ', d_a.shape, 'LabelShape:', l_a.shape) # DataShape:  (18600, 32, 256, 5) LabelShape: (18600,)
    d_v, l_v = sliding_windows_generator(DifEnt, labels, 'V')

    # Save files
    save_path = os.getcwd()
    ## Arousal
    data_type = 'data_raw_A'
    save_path_a = osp.join(save_path, data_type)
    if not os.path.exists(save_path_a):
        os.makedirs(save_path_a)

    name = 'sub' + str(sub) + '.hdf'
    save_path_a = osp.join(save_path_a, name)
    with h5py.File(save_path_a, 'w') as dataset:
        dataset['data'] = d_a
        dataset['label'] = l_a

    ## Variance
    data_type = 'data_raw_V'
    save_path_v = osp.join(save_path, data_type)
    if not os.path.exists(save_path_v):
        os.makedirs(save_path_v)

    name = 'sub' + str(sub) + '.hdf'
    save_path_v = osp.join(save_path_v, name)
    with h5py.File(save_path_v, 'w') as dataset:
        dataset['data'] = d_v
        dataset['label'] = l_v

if __name__ == '__main__':
    for sub in tqdm(range(1,33)):
        main(sub)
