import os
import numpy as np
import math
import sys
import urllib
import pickle
import input_data_class
import argparse
sys.path.append('../')
from membership_inference_attacks import black_box_benchmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str, default='location', help='location or texas')
    parser.add_argument('--predictions-dir', type=str, default='./saved_predictions', help='directory of saved predictions')
    parser.add_argument('--defended', type=int, default=1, help='1 means defended; 0 means natural')
    args = parser.parse_args()
    
    dataset = args.dataset
    input_data=input_data_class.InputData(dataset=dataset)
    (x_target,y_target,l_target) =input_data.input_data_attacker_evaluate()
    npz_data = np.load('./saved_predictions/'+dataset+'_target_predictions.npz')
    if args.defended==1:
        target_predictions = npz_data['defense_output']
    else:
        target_predictions = npz_data['tc_output']
    target_train_performance = (target_predictions[l_target==1], y_target[l_target==1].astype('int32'))
    target_test_performance = (target_predictions[l_target==0], y_target[l_target==0].astype('int32'))

    (x_shadow,y_shadow,l_shadow) =input_data.input_data_attacker_adv1()
    npz_data = np.load('./saved_predictions/'+dataset+'_shadow_predictions.npz')
    if args.defended==1:
        shadow_predictions = npz_data['defense_output']
    else:
        shadow_predictions = npz_data['tc_output']

    shadow_train_performance = (shadow_predictions[l_shadow==1], y_shadow[l_shadow==1].astype('int32'))
    shadow_test_performance = (shadow_predictions[l_shadow==0], y_shadow[l_shadow==0].astype('int32'))
    
    print('Perform membership inference attacks!!!')
    if args.dataset=='location':
        num_classes = 30
    else:
        num_classes = 100
    MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                         target_train_performance,target_test_performance,num_classes=num_classes)
    MIA._mem_inf_benchmarks()