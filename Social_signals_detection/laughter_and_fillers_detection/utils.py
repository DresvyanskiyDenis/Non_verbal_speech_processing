import os
import re
from enum import Enum

import numpy as np
import pandas as pd
from scipy.io import wavfile

class label_type(Enum):
    """Thie Enum represents type of classes in ComParE_2013_Vocalization Sub-challenge
    """
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

def load_wav_file(path_to_data):
    """This function loads wavfile with corresponding path and returns it as
       ndarray with original frame rate

    :param path_to_data: String
    :return: ndarray, readed file
             int, frame rate of readed filename
    """
    frame_rate, data = wavfile.read(path_to_data)
    return data, frame_rate

def load_labels(path_to_labels):
    """This file loads labels from one filename, which contains all labels.
       As file has .lst type of file, this function also parses file raw by raw

       Data format in file:
       "*/S0030.lab"
       0 26500000 garbage
       26500000 32800000 filler
       32800000 110000000 garbage
       .
       "*/S0031.lab"
       0 46700000 garbage
       46700000 57400000 laughter
       57400000 110000000 garbage
       .
       "*/S0032.lab"
       0 44100000 garbage
       44100000 47700000 laughter
       47700000 74518125 garbage
       .
       etc.

       As a result, it will return list, elements of which represents labels (still not parsed to float/int values)
       of one audiofile
       returned data format:
        ['S0001', ['0 66100000 garbage', '66100000 69000000 filler', '69000000 110000000 garbage']],
        ['S0002', ['0 80300000 garbage', '80300000 91800000 laughter', '91800000 110000000 garbage']],
        ['S0003', ['0 40200000 garbage', '40200000 42700000 filler', '42700000 93000000 garbage', '93000000 109200000 laughter', '109200000 110000000 garbage']]
        ...

    :param path_to_labels: String
    :return: List
    """
    filenames_and_labels=[]
    with open(path_to_labels) as fp:
        line=fp.readline()
        line = fp.readline().replace('\n', '')
        while line:
            list_of_lines=[]
            while line!='.':
                #print(line)
                if line.find(".")==-1:
                    list_of_lines.append(line)
                else:
                    filename = line.replace('\"', '').replace('*', '').replace('/', '').split('.')[0]
                line = fp.readline().replace('\n','')
            filenames_and_labels.append([filename, list_of_lines])
            #print(line)
            line = fp.readline().replace('\n','')
    return filenames_and_labels

def convert_parsed_lines_to_num_classes(parsed_list, length_label_sequence=1100):
    """This function parsed list obtained from load_labels() function
       parsed_list data format:
        ['S0001', ['0 66100000 garbage', '66100000 69000000 filler', '69000000 110000000 garbage']],
        ['S0002', ['0 80300000 garbage', '80300000 91800000 laughter', '91800000 110000000 garbage']],
        ['S0003', ['0 40200000 garbage', '40200000 42700000 filler', '42700000 93000000 garbage', '93000000 109200000 laughter', '109200000 110000000 garbage']]
        ...
       parameters length_label_sequence and labels_frame_rate are hardcoded into this function, because this database
       has exactly this 'frame rate' and it is convenient to read it with such parameters
       Function returns dictionary in the following format:
       key: name_of_filename  value: ndarray with int values of class (represented by Enum label_type)
       {'S0001': array([0, 0, 0, ..., 0, 0, 0], dtype=int16),
       'S0002': array([0, 0, 0, ..., 0, 0, 0], dtype=int16),
       'S0003': array([0, 0, 0, ..., 0, 0, 0], dtype=int16),
       ...


    :param parsed_list: list
    :param length_label_sequence: int, total length of labels sequence
    :return: dictionary, labels for each audiofile presented as int values
    """
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
        filenames_labels[parsed_list[idx_list][0]] =instance_labels.astype('int16')
    return filenames_labels, labels_frame_rate
#TODO: make comments
def load_labels_get_dict(path_to_labels):
    """

    :param path_to_labels:
    :return:
    """
    unparsed_labels=load_labels(path_to_labels)
    labels_dict, labels_frame_rate=convert_parsed_lines_to_num_classes(unparsed_labels)
    return labels_dict, labels_frame_rate

#TODO: make comments
class Database():

    def __init__(self, path_to_data, path_to_labels):
        self.path_to_data=path_to_data
        self.path_to_labels=path_to_labels
        self.data_frame_rate=None
        self.labels_frame_rate=None
        self.data_instances=[]

    def load_all_data_and_labels(self):
        dict_labels, self.labels_frame_rate = load_labels_get_dict(self.path_to_labels)
        for data_filename in dict_labels:
            instance = Database_instance()
            instance.load_data(self.path_to_data + data_filename+'.wav')
            instance.labels_frame_rate = self.labels_frame_rate
            instance.labels = dict_labels[data_filename.split('.')[0]]
            self.data_instances.append(instance)
        self.data_frame_rate=self.data_instances[0].data_frame_rate
        self.labels_frame_rate = self.data_instances[0].labels_frame_rate

    def cut_all_instances(self, window_size, window_step):
        for i in range(len(self.data_instances)):
            self.data_instances[i].cut_data_and_labels_on_windows(window_size, window_step)

    def get_all_concatenated_cutted_data_and_labels(self):
        data_window_size=self.data_instances[0].data_window_size
        labels_window_size=self.data_instances[0].labels_window_size
        result_data=np.zeros(shape=(0,data_window_size))
        result_labels=np.zeros(shape=(0, labels_window_size))
        tmp_data=[]
        tmp_labels=[]
        for i in range(len(self.data_instances)):
            tmp_data.append(self.data_instances[i].cutted_data)
            tmp_labels.append(self.data_instances[i].cutted_labels)

        result_data=np.vstack(tmp_data)
        result_labels = np.vstack(tmp_labels)
        return result_data, result_labels

    def reduce_labels_frame_rate(self, needed_frame_rate):
        ratio=int(self.labels_frame_rate/needed_frame_rate)
        self.labels_frame_rate=needed_frame_rate
        for i in range(len(self.data_instances)):
            self.data_instances[i].labels=self.data_instances[i].labels[::ratio]
            self.data_instances[i].labels_frame_rate=needed_frame_rate

    def get_predictions(self, model):
        #TODO: realize it (after realizing averaging function in Database_instance class)
        pass

    def shuffle_and_separate_cutted_data_on_train_and_val_sets(self, percent_of_validation):
        concatenated_data, concatenated_labels=self.get_all_concatenated_cutted_data_and_labels()
        permutation=np.random.permutation(concatenated_data.shape[0])
        concatenated_data, concatenated_labels= concatenated_data[permutation], concatenated_labels[permutation]
        sep_idx=int(concatenated_data.shape[0]*(100-percent_of_validation)/100.)
        train_part_data, train_part_labels= concatenated_data[:sep_idx], concatenated_labels[:sep_idx]
        val_part_data, val_part_labels= concatenated_data[sep_idx:], concatenated_labels[sep_idx:]
        return train_part_data, train_part_labels, val_part_data, val_part_labels






#TODO: make comments
class Database_instance():
    """This class represents one instance of database,
       including data and labels"""

    #TODO: реализовать функцию усреднения предсказаний, если окна пересекаются
    # для этого еще необходим массив таймстепов
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
        self.filename=path_to_data.split('\\')[-1].split('/')[-1].split('.')[0]


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

    def cut_sequence_on_windows(self, sequence, window_size, window_step):
        num_windows=how_many_windows_do_i_need(sequence.shape[0], window_size, window_step)
        cutted_data=np.zeros(shape=(num_windows, window_size))
        start_idx=0
        # start of cutting
        for idx_window in range(num_windows-1):
            end_idx=start_idx+window_size
            cutted_data[idx_window]=sequence[start_idx:end_idx]
            start_idx+=window_step
        # last window
        end_idx=sequence.shape[0]
        start_idx=end_idx-window_size
        cutted_data[num_windows-1]=sequence[start_idx:end_idx]
        return cutted_data

    def cut_data_and_labels_on_windows(self, window_size, window_step):
        # calculate params for cutting (size of window and step in index)
        self.data_window_size=int(window_size*self.data_frame_rate)
        self.data_window_step=int(window_step*self.data_frame_rate)
        self.labels_window_size=int(window_size*self.labels_frame_rate)
        self.labels_window_step=int(window_step*self.labels_frame_rate)

        self.cutted_data=self.cut_sequence_on_windows(self.data, self.data_window_size, self.data_window_step)
        self.cutted_labels=self.cut_sequence_on_windows(self.labels, self.labels_window_size, self.labels_window_step)
        self.cutted_data = self.cutted_data.astype('float32')
        self.cutted_labels = self.cutted_labels.astype('int32')
        return self.cutted_data, self.cutted_labels


    def load_labels(self, path_to_labels):
        dict_labels, self.labels_frame_rate=load_labels_get_dict(path_to_labels)
        self.labels=dict_labels[self.filename]

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

