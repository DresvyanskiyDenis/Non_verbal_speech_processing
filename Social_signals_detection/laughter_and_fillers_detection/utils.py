import os
import re
from enum import Enum

import numpy as np
import pandas as pd
from scipy.io import wavfile

class label_type(Enum):
    """This Enum represents type of classes in ComParE_2013_Vocalization Sub-challenge
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
             int, frame rate of labels
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

def load_labels_get_dict(path_to_labels):
    """This function exploits functions load_labels() and convert_parsed_lines_to_num_classes()
       to load and parse into dict labels from one file

    :param path_to_labels:String
    :return: dictionary, labels for each audiofile presented as int values
             int, frame rate of labels
    """
    unparsed_labels=load_labels(path_to_labels)
    labels_dict, labels_frame_rate=convert_parsed_lines_to_num_classes(unparsed_labels)
    return labels_dict, labels_frame_rate

class Database():

    def __init__(self, path_to_data, path_to_labels):

        self.path_to_data=path_to_data
        self.path_to_labels=path_to_labels
        self.data_frame_rate=None
        self.labels_frame_rate=None
        self.data_instances=[]

    def load_all_data_and_labels(self):
        """This function loads data and labels from folder self.path_to_data and file with path path_to_labels
           For computational efficiency the loading of labels is made as a separate function load_labels_get_dict()
           Every file is represented as instance of class Database_instance(). The data loading realized by Database_instance() class.
           Since all files have the same frame rates (as well as labels), data_frame_rate and labels_frame_rate will set
           to the same value taken from first element of list data_instances

        :return:None
        """
        # Since all labels are represented by only one file, for computational effeciency firstly we load all labels
        # and then give them to different loaded audiofiles
        dict_labels, self.labels_frame_rate = load_labels_get_dict(self.path_to_labels)
        for data_filename in dict_labels:
            instance = Database_instance()
            instance.load_data(self.path_to_data + data_filename+'.wav')
            instance.labels_frame_rate = self.labels_frame_rate
            instance.labels = dict_labels[data_filename.split('.')[0]]
            instance.generate_timesteps_for_labels()
            self.data_instances.append(instance)
        self.data_frame_rate=self.data_instances[0].data_frame_rate
        self.labels_frame_rate = self.data_instances[0].labels_frame_rate

    def cut_all_instances(self, window_size, window_step):
        """This function is cutting all instances of database (elements of list, which is Database_instance())
        It exploits included in Database_instance() class function for cutting.

        :param window_size: float, size of window in seconds
        :param window_step: float, step of window in seconds
        :return: None
        """
        for i in range(len(self.data_instances)):
            self.data_instances[i].cut_data_and_labels_on_windows(window_size, window_step)

    def get_all_concatenated_cutted_data_and_labels(self):
        """This function concatenates cutted data and labels of all elements of list self.data_instances
           Every element of list is Database_instance() class, which contains field cutted_data and cutted_labels

        :return: 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance, data_window_size),
                    concatenated cutted_data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance, labels_window_size),
                    concatenated cutted_labels of every element of list self.data_instances
        """
        tmp_data=[]
        tmp_labels=[]
        for i in range(len(self.data_instances)):
            tmp_data.append(self.data_instances[i].cutted_data)
            tmp_labels.append(self.data_instances[i].cutted_labels)
        result_data=np.vstack(tmp_data)
        result_labels = np.vstack(tmp_labels)
        return result_data, result_labels

    def reduce_labels_frame_rate(self, needed_frame_rate):
        """This function reduce labels frame rate to needed frame rate by taking every (second, thirs and so on) elements from
           based on calculated ratio.
           ratio calculates between current frame rate and needed frame rate

        :param needed_frame_rate: int, needed frame rate of labels per one second (e.g. 25 labels per second)
        :return:None
        """
        ratio=int(self.labels_frame_rate/needed_frame_rate)
        self.labels_frame_rate=needed_frame_rate
        for i in range(len(self.data_instances)):
            self.data_instances[i].labels=self.data_instances[i].labels[::ratio]
            self.data_instances[i].labels_frame_rate=needed_frame_rate

    def get_predictions(self, model):
        #TODO: realize it (after realizing averaging function in Database_instance class)
        pass

    def shuffle_and_separate_cutted_data_on_train_and_val_sets(self, percent_of_validation):
        """This function shuffle and then separate cutted data and labels by given percent_of_validation
           It exploits class function get_all_concatenated_cutted_data_and_labels() to get cutted data and labels from
           each database_instance and then concatenate it
           Then resulted arrays of get_all_concatenated_cutted_data_and_labels() function will be
           shuffled and then separated on train and validation parts

        :param percent_of_validation: float, percent of validation part in all data
        :return: 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*(100-percent_of_validation)/100, data_window_size),
                    train data - concatenated cutted_data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*(100-percent_of_validation)/100, labels_window_size),
                    train labels - concatenated cutted_labels of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*percent_of_validation/100, data_window_size),
                    validation data - concatenated cutted_data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*percent_of_validation/100, labels_window_size),
                    validation labels - concatenated cutted_data of every element of list self.data_instances
        """
        concatenated_data, concatenated_labels=self.get_all_concatenated_cutted_data_and_labels()
        permutation=np.random.permutation(concatenated_data.shape[0])
        concatenated_data, concatenated_labels= concatenated_data[permutation], concatenated_labels[permutation]
        sep_idx=int(concatenated_data.shape[0]*(100-percent_of_validation)/100.)
        train_part_data, train_part_labels= concatenated_data[:sep_idx], concatenated_labels[:sep_idx]
        val_part_data, val_part_labels= concatenated_data[sep_idx:], concatenated_labels[sep_idx:]
        return train_part_data, train_part_labels, val_part_data, val_part_labels






class Database_instance():
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
        self.labels_timesteps= None
        self.cutted_labels_timesteps=None
        self.cutted_predictions=None
        self.predictions=None

    def load_data(self, path_to_data):
        """ This function load data and corresponding frame rate from wav type file

        :param path_to_data: String
        :return: None
        """
        self.data, self.data_frame_rate=load_wav_file(path_to_data)
        self.filename=path_to_data.split('\\')[-1].split('/')[-1].split('.')[0]


    def pad_the_sequence(self, sequence, window_size,  mode, padding_value=0):
        """This fucntion pad sequence with corresponding padding_value to the given shape of window_size
        For example, if we have sequence with shape 4 and window_size=6, then
        it just concatenates 2 specified values like
        to the right, if padding_mode=='right'
            last_step   -> _ _ _ _ v v  where v is value (by default equals 0)
        to the left, if padding_mode=='left'
            last_step   -> v v _ _ _ _  where v is value (by default equals 0)
        to the center, if padding_mode=='center'
            last_step   -> v _ _ _ _ v  where v is value (by default equals 0)

        :param sequence: ndarray
        :param window_size: int
        :param mode: string, can be 'right', 'left' or 'center'
        :param padding_value: float
        :return: ndarray, padded to given window_size sequence
        """
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
        """This function cuts given sequence on windows with corresponding window_size and window_step
        for example, if we have sequence [1 2 3 4 5 6 7 8], window_size=4, window_step=3 then
        1st step: |1 2 3 4| 5 6 7 8
                  ......
        2nd step: 1 2 3 |4 5 6 7| 8
                        ..
        3rd step: 1 2 3 4 |5 6 7 8|

        Here, in the last step, if it is not enough space for window, we just take window, end of which is last element
        In given example for it we just shift window on one element
        In future version maybe padding will be added
        :param sequence: ndarray
        :param window_size: int, size of window
        :param window_step: int, step of window
        :return: 2D ndarray, shape=(num_windows, window_size)
        """
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
        """This function exploits function cut_sequence_on_windows() for cutting data and labels
           with corresponding window_size and window_step
           Window_size and window_step are calculating independently corresponding data and labels frame rate

        :param window_size: float, size of window in seconds
        :param window_step: float, step of window in seconds
        :return: 2D ndarray, shape=(num_windows, data_window_size), cutted data
                 2D ndarray, shape=(num_windows, labels_window_size), cutted labels
        """
        # calculate params for cutting (size of window and step in index)
        self.data_window_size=int(window_size*self.data_frame_rate)
        self.data_window_step=int(window_step*self.data_frame_rate)
        self.labels_window_size=int(window_size*self.labels_frame_rate)
        self.labels_window_step=int(window_step*self.labels_frame_rate)

        self.cutted_data=self.cut_sequence_on_windows(self.data, self.data_window_size, self.data_window_step)
        self.cutted_labels=self.cut_sequence_on_windows(self.labels, self.labels_window_size, self.labels_window_step)
        self.cutted_labels_timesteps=self.cut_sequence_on_windows(self.labels_timesteps, self.labels_window_size, self.labels_window_step)
        self.cutted_data = self.cutted_data.astype('float32')
        self.cutted_labels = self.cutted_labels.astype('int32')
        self.cutted_labels_timesteps= self.cutted_labels_timesteps.astype('float32')
        return self.cutted_data, self.cutted_labels, self.cutted_labels_timesteps


    def load_labels(self, path_to_labels):
        """This function loads labels for certain, concrete audiofile
        It exploits load_labels_get_dict() function, which loads and parses all labels from one label-file
        Then we just take from obtained dictionary labels by needed audio filename.
        Current solution is computational unefficient, but it is used very rarely

        :param path_to_labels:String
        :return:None
        """
        dict_labels, self.labels_frame_rate=load_labels_get_dict(path_to_labels)
        self.labels=dict_labels[self.filename]

    def load_and_preprocess_data_and_labels(self, path_to_data, path_to_labels):
        """This function loads data and labels from corresponding paths

        :param path_to_data: String
        :param path_to_labels: String
        :return: None
        """
        self.load_data(path_to_data)
        self.load_labels(path_to_labels)

    def generate_timesteps_for_labels(self):
        """This function generates timesteps for labels with corresponding labels_frame_rate
           After executing it will be saved in field self.labels_timesteps
        :return: None
        """
        label_timestep_in_sec=1./self.labels_timesteps
        timesteps=np.array([i for i in range(self.labels.shape[0])], dtype='float32')
        timesteps=timesteps*label_timestep_in_sec
        self.labels_timesteps=timesteps

#TODO: realize this class and make comments
class Metric_calculator():

    def __init__(self):
        self.ground_truth=None
        self.predictions=None
        self.cutted_predictions=None
        self.cutted_labels_timesteps=None

    def average_cutted_predictions_by_timestep(self):
        pass

    def calculate_AUC_ROC(self):
        pass






if __name__ == "__main__":
    path_to_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\ComParE_2013_Vocalization\\ComParE2013_Voc\\lab\\train.mlf'
    path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\ComParE_2013_Vocalization\\ComParE2013_Voc\\wav\\S0001.wav'
    window_size=1.5
    window_step=0.5
    #instance=database_instance()
    #instance.load_and_preprocess_data_and_labels(path_to_data, path_to_labels)
    #instance.cut_data_and_labels_on_windows(window_size, window_step)

