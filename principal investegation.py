import os
import numpy as np
import copy
import pickle
import math
import statistics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import parallel_processing as pp
from functools import partial
from scipy import stats
from scipy.optimize import minimize
from sklearn import linear_model as lm
from random import shuffle
import import_beh_data as di
import beh_analysis as pl
from collections import defaultdict

# Define a regular function as the default factory for the defaultdict




Dictionary = {}

master_data_folder = r"C:\Users\Test\OneDrive\Documents\FHS_project_data\Experiment_1"
for group_condition_folder in os.listdir(master_data_folder):
    if str(group_condition_folder) != '.DS_Store':# or str(group_condition_folder)[-3:] != 'rtf':
        print(group_condition_folder)
        group_condition_folder_path = os.path.join(master_data_folder, group_condition_folder)
        dir_folder_session = group_condition_folder_path
        experiment = di.Experiment(dir_folder_session)
        experiment.save()
        sessions = experiment.get_sessions(subject_IDs='all', when='all')

        if str(group_condition_folder)[9:11] != 'TR':
            if str(group_condition_folder)[9:17] == 'baseline':
                condition = 'baseline'
            else:
                condition = 'Drug'
            if str(group_condition_folder)[-1] == 'I':
                treatment = 'DOI'
            else:
                treatment = 'VEH'
            for session in sessions:
                subject_name = str(session.subject_ID)
                # Ensure treatment and condition keys exist
                Dictionary.setdefault(treatment, {})
                Dictionary[treatment].setdefault(condition, {})
                session_number = session.datetime
                a,data_piece = pl.Choice_latency_after_free_choice_ommision(session, split_transitions=True)

                Dictionary[treatment][condition].setdefault(subject_name, {})
                Dictionary[treatment][condition][subject_name][session_number] = data_piece
                #swap prior two lines to return
                r"""
                if subject_name in Dictionary[treatment][condition]:
                    Dictionary[treatment][condition][subject_name].append(data_piece)
                    np.concatenate([Dictionary[treatment][condition][subject_name]], axis=0, dtype=object)
                else:
                    Dictionary[treatment][condition][subject_name] = [data_piece]
                """
        elif str(group_condition_folder)[9:11] == 'TR':
            condition = 'TR'
            if str(group_condition_folder)[12:15] == 'DOI':
                treatment = 'DOI'
            else:
                treatment = 'VEH'
            for session in sessions:
                subject_name = str(session.subject_ID)
                # Ensure treatment and condition keys exist
                Dictionary.setdefault(treatment, {})
                Dictionary[treatment].setdefault(condition, {})


                session_number = session.datetime

                # Call the specified function
                a,data_piece = pl.Choice_latency_after_free_choice_ommision(session, split_transitions=True)

                Dictionary[treatment][condition].setdefault(subject_name, {})
                Dictionary[treatment][condition][subject_name][session_number] = data_piece

# remove non TR section to return
for category, inner_data in Dictionary['DOI']['TR'].items():
    sorted_inner_data = copy.deepcopy(inner_data)
    sorted_inner_data = {key: value for key, value in sorted(sorted_inner_data.items())}
    new_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    renamed_SID = {new_keys[i]: sorted_inner_data[key] for i, key in enumerate(sorted_inner_data)}
    Dictionary['DOI']['TR'][category] = renamed_SID

for category, inner_data in Dictionary['DOI']['baseline'].items():
    sorted_inner_data = copy.deepcopy(inner_data)
    sorted_inner_data = {key: value for key, value in sorted(sorted_inner_data.items())}
    new_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    renamed_SID = {new_keys[i]: sorted_inner_data[key] for i, key in enumerate(sorted_inner_data)}
    Dictionary['DOI']['baseline'][category] = renamed_SID

for category, inner_data in Dictionary['DOI']['Drug'].items():
    sorted_inner_data = copy.deepcopy(inner_data)
    sorted_inner_data = {key: value for key, value in sorted(sorted_inner_data.items())}
    new_keys = ['0', '1', '2', '3', '4', '5']
    renamed_SID = {new_keys[i]: sorted_inner_data[key] for i, key in enumerate(sorted_inner_data)}
    Dictionary['DOI']['Drug'][category] = renamed_SID

for category, inner_data in Dictionary['VEH']['TR'].items():
    sorted_inner_data = copy.deepcopy(inner_data)
    sorted_inner_data = {key: value for key, value in sorted(sorted_inner_data.items())}
    new_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    renamed_SID = {new_keys[i]: sorted_inner_data[key] for i, key in enumerate(sorted_inner_data)}
    Dictionary['VEH']['TR'][category] = renamed_SID

for category, inner_data in Dictionary['VEH']['baseline'].items():
    sorted_inner_data = copy.deepcopy(inner_data)
    sorted_inner_data = {key: value for key, value in sorted(sorted_inner_data.items())}
    new_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    renamed_SID = {new_keys[i]: sorted_inner_data[key] for i, key in enumerate(sorted_inner_data)}
    Dictionary['VEH']['baseline'][category] = renamed_SID

for category, inner_data in Dictionary['VEH']['Drug'].items():
    sorted_inner_data = copy.deepcopy(inner_data)
    sorted_inner_data = {key: value for key, value in sorted(sorted_inner_data.items())}
    new_keys = ['0', '1', '2', '3', '4', '5']
    renamed_SID = {new_keys[i]: sorted_inner_data[key] for i, key in enumerate(sorted_inner_data)}
    Dictionary['VEH']['Drug'][category] = renamed_SID

# Save the Dictionary to a file
with open('next_trial_choice_OR', 'wb') as file:
    pickle.dump(Dictionary, file)
