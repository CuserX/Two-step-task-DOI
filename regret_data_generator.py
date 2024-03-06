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
def default_factory():
    return defaultdict(dict)

def data_extraction(function):
    dictionary_name = defaultdict(default_factory)

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
                    dictionary_name.setdefault(treatment, {})
                    dictionary_name[treatment].setdefault(condition, {})

                    data_piece = function(session)
                    if subject_name in dictionary_name[treatment][condition]:
                        dictionary_name[treatment][condition][subject_name].append(data_piece)
                        np.concatenate([dictionary_name[treatment][condition][subject_name]], axis=0, dtype=object)
                    else:
                        dictionary_name[treatment][condition][subject_name] = [data_piece]

            elif str(group_condition_folder)[9:11] == 'TR':
                condition = 'TR'
                if str(group_condition_folder)[12:15] == 'DOI':
                    treatment = 'DOI'
                else:
                    treatment = 'VEH'
                for session in sessions:
                    subject_name = str(session.subject_ID)
                    # Ensure treatment and condition keys exist
                    dictionary_name.setdefault(treatment, {})
                    dictionary_name[treatment].setdefault(condition, {})


                    session_number = session.datetime

                    # Call the specified function
                    data_piece = function(session)

                    dictionary_name[treatment][condition].setdefault(subject_name, {})
                    dictionary_name[treatment][condition][subject_name][session_number] = data_piece

    for category, inner_data in dictionary_name['DOI']['TR'].items():
        sorted_inner_data = {key: value for key, value in sorted(inner_data.items())}
        dictionary_name['DOI']['TR'][category] = sorted_inner_data
    for category, inner_data in dictionary_name['VEH']['TR'].items():
        sorted_inner_data = {key: value for key, value in sorted(inner_data.items())}
        dictionary_name['VEH']['TR'][category] = sorted_inner_data

    for treatment in dictionary_name:
        for category in dictionary_name[treatment]['TR']:
            animal_ordered_sessions = []
            for date in dictionary_name[treatment]['TR'][category]:
                animal_ordered_sessions.append(dictionary_name[treatment]['TR'][category][date])
            dictionary_name[treatment]['TR'][category] = np.array(animal_ordered_sessions)
    return dictionary_name

# Define the function you want to apply to the data
def custom_function(session):
    return pl.regret_calculation(session)

# Call data_extraction with the custom function
regretted_trials = data_extraction(custom_function)
r"""
# Calculate the mean values for 'baseline' and 'Drug' conditions in regretted_trials
mean_values = {}
for treatment in regretted_trials:
    mean_values[treatment] = {}
    for condition in regretted_trials[treatment]:
        if condition in ['baseline', 'Drug']:
            mean_values[treatment][condition] = {}
            for animal in regretted_trials[treatment][condition]:
                mean_values[treatment][condition][animal] = np.mean(
                    np.array(regretted_trials[treatment][condition][animal])
                )

# Create a copy of regretted_trials
regret_change = copy.deepcopy(regretted_trials)

# Calculate the changes in the copied regret_change dictionary
for treatment in regret_change:
    for condition in regret_change[treatment]:
        if condition in ['Drug', 'TR']:
            for animal in regret_change[treatment][condition]:
                regret_change[treatment][condition][animal] = (
                    regret_change[treatment][condition][animal]
                    / mean_values[treatment]['baseline'][animal]
                )


for treatment in regretted_trials:
    for condition in regretted_trials[treatment]:
        if condition == 'baseline' or condition =='Drug':
            for animal in regretted_trials[treatment][condition]:
                regretted_trials[treatment][condition][animal] = np.mean(np.array(regretted_trials[treatment][condition][animal]))

regret_change = copy.copy(regretted_trials)
for treatment in regret_change:
    for condition in regret_change[treatment]:
        if condition =='Drug':
            for animal in regret_change[treatment][condition]:
                regret_change[treatment][condition][animal]= regretted_trials[treatment][condition][animal]-regretted_trials[treatment]['baseline'][animal]
        elif condition =='TR':
            for animal in regret_change[treatment][condition]:
                regret_change[treatment][condition][animal] = regretted_trials[treatment][condition][animal]-regretted_trials[treatment]['baseline'][animal]

regret_change_means = {
    'DOI': {'Drug': [], 'TR': []},
    'VEH': {'Drug': [], 'TR': []}
}
for treatment in regret_change:
    for condition in regret_change[treatment]:
        if condition == 'Drug':
            Drug = []
            for animal in regret_change[treatment][condition]:
                Drug.append(regret_change[treatment][condition][animal])
            mean_drug = np.mean(np.array(Drug))
            regret_change_means[treatment]['Drug'] = mean_drug
        elif condition == 'TR':
            TR_regret_change = []
            for animal in regret_change[treatment][condition]:
                TR_regret_change.append(regret_change[treatment][condition][animal])
            np.array(np.row_stack(TR_regret_change))
            # Split the matrix into column sections of 3
            num_sections = TR_regret_change.shape[1] // 3  # Calculate the number of sections
            sections = [TR_regret_change[:, i * 3:(i + 1) * 3] for i in range(num_sections)]
            for section in sections:
                section = np.mean(section, axis=1)
            TR_regret_change = np.column_stack(sections)

            TR_regret_change_mean = np.mean(TR_regret_change, axis=0)

            regret_change_means[treatment][condition]= TR_regret_change_mean



regret = {'trials': regretted_trials, 'change': regret_change}

# Define the filename for the .pkl file
filename = 'regret_storage.pkl'

# Save the dictionary to a .pkl file
with open(filename, 'wb') as file:
    pickle.dump(regret, file)
"""


