import re
from enum import Enum

import numpy as np
import pandas as pd
from scipy.io import wavfile

class label_type(Enum):
    garbage=0
    filler=1
    laughter=2

def how_many_windows_do_i_need(length_sequence, window_size, step):
    """This function calculates how many windows do you need
        with corresponding length of sequence, window_size and
        window_step
        for example, if your sequence length=10, window_size=4 and
        window_step=2 then:
        |_ _ _ _| _ _ _ _ _ _
        _ _ |_ _ _ _| _ _ _ _
        _ _ _ _ |_ _ _ _| _ _
        _ _ _ _ _ _ |_ _ _ _|
        ==> you need 4 windows with this parameters

    :param length_sequence: int, the length of sequence
    :param window_size: int, the length of window
    :param step: int
    :return: int, number of windows needed for this sequence
    """
    start_idx=0
    counter=0
    while True:
        if start_idx+window_size>=length_sequence:
            break
        start_idx+=step
        counter+=1
    if start_idx!=length_sequence:
        counter+=1
    return counter


"Что я вообще делаю? Найди правильное решение, тупо загружать файлы и лэйблы к ним" \
" можно, например, создать класс data_instance, а потом класс, содержащий их в листе и проводящий" \
" с ними операции обработки. лэйблы там же"

def load_wav_file(path_to_data):
    frame_rate, data = wavfile.read(path_to_data)
    return data, frame_rate

def load_labels(path_to_labels):
    filenames_and_labels=[]
    with open(path_to_labels) as fp:
        line=fp.readline()
        line = fp.readline().replace('\n', '')
        while line:
            list_of_lines=[]
            while line!='.':
                print(line)
                if line.find(".")==-1:
                    list_of_lines.append(line)
                else:
                    filename = line.replace('\"', '').replace('*', '').replace('/', '').split('.')[0]
                line = fp.readline().replace('\n','')
            filenames_and_labels.append([filename, list_of_lines])
            print(line)
            line = fp.readline().replace('\n','')
    return filenames_and_labels



def convert_parsed_lines_to_num_classes(parsed_list, length_label_sequence=1100):
    filenames_labels={}
    labels_frame_rate=100
    for idx_list in range(len(parsed_list)):
        instance_lines=parsed_list[idx_list][1]
        instance_labels=np.zeros(shape=(length_label_sequence,))
        for line in instance_lines:
            tmp=line.split(' ')
            start_idx=int(np.array(tmp[0]).astype('int') / 100000)
            end_idx = int(np.array(tmp[1]).astype('int') / 100000)
            label_value=label_type[tmp[2]].value
            instance_labels[start_idx:end_idx]=label_value
        filenames_labels[parsed_list[idx_list][0]] =instance_labels.astype('int32')
    return filenames_labels, labels_frame_rate



class database_instance():
    """This class represents one instance of database,
       including data and labels"""

    def __init__(self):

        self.filename=None
        self.data_window_size = None
        self.data_window_step = None
        self.labels_window_size = None
        self.labels_window_step = None
        self.data = None
        self.data_frame_rate=None
        self.cutted_data = None
        self.labels = None
        self.labels_frame_rate=None
        self.cutted_labels = None

    def load_data(self, path_to_data):
        self.data, self.data_frame_rate=load_wav_file(path_to_data)
        self.filename=path_to_data.split('\\')[-1].split('.')[0]

    def load_labels(self, path_to_labels):
        unparsed_labels=load_labels(path_to_labels)
        dict_labels, self.labels_frame_rate=convert_parsed_lines_to_num_classes(unparsed_labels)
        self.labels=dict_labels[self.filename]

    def pad_the_sequence(self, sequence, window_size,  mode, padding_value=0):
        result=np.ones(shape=(window_size))*padding_value
        if mode=='left':
            result[(window_size-sequence.shape[0]):]=sequence
        elif mode=='right':
            result[:sequence.shape[0]]=sequence
        elif mode=='center':
            start=(window_size-sequence.shape[0])//2
            end=start+sequence.shape[0]
            result[start:end]=sequence
        else:
            raise AttributeError('mode can be either left, right or center')
        return result

    def cut_data_and_labels_on_windows(self, window_size, window_step):
        self.data_window_size=int(window_size*self.data_frame_rate)
        self.data_window_step=int(window_step*self.data_frame_rate)
        self.labels_window_size=int(window_size*self.labels_frame_rate)
        self.labels_window_step=int(window_step*self.labels_frame_rate)
        # arrays for cutting window
        num_windows = how_many_windows_do_i_need(self.data.shape[0], self.data_window_size, self.data_window_step)
        self.cutted_data = np.zeros(shape=(num_windows, self.data_window_size))
        self.cutted_labels= np.zeros(shape=(num_windows, self.labels_window_size))
        # start of cutting
        data_start_idx=0
        labels_start_idx=0
        for idx_window in range(num_windows-1):
            data_end_idx=data_start_idx+self.data_window_size
            labels_end_idx=labels_start_idx+self.labels_window_size
            self.cutted_data[idx_window]=self.data[data_start_idx:data_end_idx]
            self.cutted_labels[idx_window]=self.labels[labels_start_idx:labels_end_idx]
            data_start_idx=data_start_idx+self.data_window_step
            labels_start_idx=labels_start_idx+self.labels_window_step
        # processing the data on last step of cutting
        # (remaining sequence can be less than window size at the end of data raw)
        data_end_idx=self.data.shape[0]
        data_start_idx=data_end_idx-self.data_window_size
        labels_end_idx=self.labels.shape[0]
        labels_start_idx=labels_end_idx-self.labels_window_size
        self.cutted_data[num_windows-1]=self.data[data_start_idx:data_end_idx]
        self.cutted_labels[num_windows-1]=self.labels[labels_start_idx:labels_end_idx]
        self.cutted_data=self.cutted_data.astype('float32')
        self.cutted_labels=self.cutted_labels.astype('int32')
        return self.cutted_data, self.cutted_labels

    def load_and_preprocess_data_and_labels(self, path_to_data, path_to_labels):
        self.load_data(path_to_data)
        self.load_labels(path_to_labels)



if __name__ == "__main__":
    path_to_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\ComParE_2013_Vocalization\\ComParE2013_Voc\\lab\\train.mlf'
    path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\ComParE_2013_Vocalization\\ComParE2013_Voc\\wav\\S0001.wav'
    window_size=1.5
    window_step=0.5
    instance=database_instance()
    instance.load_and_preprocess_data_and_labels(path_to_data, path_to_labels)
    instance.cut_data_and_labels_on_windows(window_size, window_step)

