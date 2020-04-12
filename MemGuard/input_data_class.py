import numpy as np 
import ast
import configparser
np.random.seed(100)
import os


class InputData():
    def __init__(self, dataset):
        self.dataset=dataset

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')


        self.data_filepath=str(self.config[dataset]['all_data_path'])
        self.index_filepath=str(self.config[dataset]['shuffle_index'])

        self.user_training_data_range=self.config[dataset]['user_training_data_index_range']
        self.user_training_data_range=ast.literal_eval(self.user_training_data_range)

        self.user_testing_data_range=self.config[dataset]['user_testing_data_index_range']
        self.user_testing_data_range=ast.literal_eval(self.user_testing_data_range)

        self.defense_member_data_index_range=self.config[dataset]['defense_member_data_index_range']
        self.defense_member_data_index_range=ast.literal_eval(self.defense_member_data_index_range)

        self.defense_nonmember_data_index_range=self.config[dataset]['defense_nonmember_data_index_range']
        self.defense_nonmember_data_index_range=ast.literal_eval(self.defense_nonmember_data_index_range)


        self.attacker_train_member_data_range=self.config[dataset]['attacker_train_member_data_range']
        self.attacker_train_member_data_range=ast.literal_eval(self.attacker_train_member_data_range)

        self.attacker_train_nonmember_data_range=self.config[dataset]['attacker_train_nonmember_data_range']
        self.attacker_train_nonmember_data_range=ast.literal_eval(self.attacker_train_nonmember_data_range)

        self.attacker_evaluate_member_data_range=self.config[dataset]['attacker_evaluate_member_data_range']
        self.attacker_evaluate_member_data_range=ast.literal_eval(self.attacker_evaluate_member_data_range)

        self.attacker_evaluate_nonmember_data_range=self.config[dataset]['attacker_evaluate_nonmember_data_range']
        self.attacker_evaluate_nonmember_data_range=ast.literal_eval(self.attacker_evaluate_nonmember_data_range)


    def input_data_user(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']
        x_train_user=x_data[index_data[int(self.user_training_data_range["start"]):int(self.user_training_data_range["end"])],:]
        x_test_user=x_data[index_data[int(self.user_testing_data_range["start"]):int(self.user_testing_data_range["end"])],:]
        y_train_user=y_data[index_data[int(self.user_training_data_range["start"]):int(self.user_training_data_range["end"])]]
        y_test_user=y_data[index_data[int(self.user_testing_data_range["start"]):int(self.user_testing_data_range["end"])]]
        y_train_user=y_train_user-1.0
        y_test_user=y_test_user-1.0
        return (x_train_user,y_train_user),(x_test_user,y_test_user)

    def input_data_defender(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']
        x_train_user=x_data[index_data[int(self.defense_member_data_index_range["start"]):int(self.defense_member_data_index_range["end"])],:]
        x_nontrain_defender=x_data[index_data[int(self.defense_nonmember_data_index_range["start"]):int(self.defense_nonmember_data_index_range["end"])],:]
        y_train_user=y_data[index_data[int(self.defense_member_data_index_range["start"]):int(self.defense_member_data_index_range["end"])]]
        y_nontrain_defender=y_data[index_data[int(self.defense_nonmember_data_index_range["start"]):int(self.defense_nonmember_data_index_range["end"])]]

        x_train_defender=np.concatenate((x_train_user,x_nontrain_defender),axis=0)
        y_train_defender=np.concatenate((y_train_user,y_nontrain_defender),axis=0)
        y_train_defender=y_train_defender-1.0

        label_train_defender=np.zeros([x_train_defender.shape[0]],dtype=np.int)
        label_train_defender[0:x_train_user.shape[0]]=1
        return (x_train_defender,y_train_defender,label_train_defender)


    def input_data_attacker_adv1(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']

        x_train_member_attacker=x_data[index_data[int(self.attacker_train_member_data_range["start"]):int(self.attacker_train_member_data_range["end"])],:]
        x_train_nonmember_attacker=x_data[index_data[int(self.attacker_train_nonmember_data_range["start"]):int(self.attacker_train_nonmember_data_range["end"])],:]
        y_train_member_attacker=y_data[index_data[int(self.attacker_train_member_data_range["start"]):int(self.attacker_train_member_data_range["end"])]]
        y_train_nonmumber_attacker=y_data[index_data[int(self.attacker_train_nonmember_data_range["start"]):int(self.attacker_train_nonmember_data_range["end"])]]
        x_train_attacker=np.concatenate((x_train_member_attacker,x_train_nonmember_attacker),axis=0)
        y_train_attacker=np.concatenate((y_train_member_attacker,y_train_nonmumber_attacker),axis=0)
        y_train_attacker=y_train_attacker-1.0
        label_train_attacker=np.zeros([x_train_attacker.shape[0]],dtype=np.int)
        label_train_attacker[0:x_train_member_attacker.shape[0]]=1

        return (x_train_attacker,y_train_attacker,label_train_attacker)

    def input_data_attacker_evaluate(self):
        #print("Attacker evaluate member data range: {}".format(self.attacker_evaluate_member_data_range))
        #print("Attacker evaluate nonmember data range: {}".format(self.attacker_evaluate_nonmember_data_range))
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']

        x_evaluate_member_attacker=x_data[index_data[int(self.attacker_evaluate_member_data_range["start"]):int(self.attacker_evaluate_member_data_range["end"])],:]
        x_evaluate_nonmember_attacker=x_data[index_data[int(self.attacker_evaluate_nonmember_data_range["start"]):int(self.attacker_evaluate_nonmember_data_range["end"])],:]
        y_evaluate_member_attacker=y_data[index_data[int(self.attacker_evaluate_member_data_range["start"]):int(self.attacker_evaluate_member_data_range["end"])]]
        y_evaluate_nonmumber_attacker=y_data[index_data[int(self.attacker_evaluate_nonmember_data_range["start"]):int(self.attacker_evaluate_nonmember_data_range["end"])]]
        x_evaluate_attacker=np.concatenate((x_evaluate_member_attacker,x_evaluate_nonmember_attacker),axis=0)
        y_evaluate_attacker=np.concatenate((y_evaluate_member_attacker,y_evaluate_nonmumber_attacker),axis=0)
        y_evaluate_attacker=y_evaluate_attacker-1.0
        label_evaluate_attacker=np.zeros([x_evaluate_attacker.shape[0]],dtype=np.int)
        label_evaluate_attacker[0:x_evaluate_member_attacker.shape[0]]=1

        return (x_evaluate_attacker,y_evaluate_attacker,label_evaluate_attacker)


    def input_data_attacker_shallow_model_adv1(self):
        npzdata=np.load(self.data_filepath)
        x_data=npzdata['x'][:,:]
        y_data=npzdata['y'][:]
        npzdata_index=np.load(self.index_filepath)
        index_data=npzdata_index['x']

        x_train_member_attacker=x_data[index_data[int(self.attacker_train_member_data_range["start"]):int(self.attacker_train_member_data_range["end"])],:]
        x_train_nonmember_attacker=x_data[index_data[int(self.attacker_train_nonmember_data_range["start"]):int(self.attacker_train_nonmember_data_range["end"])],:]
        y_train_member_attacker=y_data[index_data[int(self.attacker_train_member_data_range["start"]):int(self.attacker_train_member_data_range["end"])]]
        y_train_nonmumber_attacker=y_data[index_data[int(self.attacker_train_nonmember_data_range["start"]):int(self.attacker_train_nonmember_data_range["end"])]]
        y_train_member_attacker=y_train_member_attacker-1.0
        y_train_nonmumber_attacker=y_train_nonmumber_attacker-1.0

        return (x_train_member_attacker,y_train_member_attacker),(x_train_nonmember_attacker,y_train_nonmumber_attacker)
