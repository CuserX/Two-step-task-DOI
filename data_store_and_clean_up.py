import import_beh_data as di
import beh_analysis as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sb
import os
import shutil
import natsort
from multiprocessing import Manager, Process
from datetime import datetime

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
#-----------------------------------------------------------------------------------------------------------------------
def custom_sort_key(string):
    fifth_digit = string[4]
    sixth_digit = string[5]
    eighth_digit = string[7]
    ninth_digit = string[8]

    if fifth_digit == '0':
        fifth_sort = 0
    else:
        try:
            fifth_sort = int(fifth_digit)
        except ValueError:
            fifth_sort = float('inf')  # Assign a large value for invalid digits

    try:
        sixth_sort = int(sixth_digit)
    except ValueError:
        sixth_sort = float('inf')

    try:
        eighth_sort = int(eighth_digit)
    except ValueError:
        eighth_sort = float('inf')

    try:
        ninth_sort = int(ninth_digit)
    except ValueError:
        ninth_sort = float('inf')

    return (fifth_sort, sixth_sort, eighth_sort, ninth_sort)


dir_experiment = "C:/Users/Test/OneDrive/Documents/FHS_project_data/Experiment_1_clean"
experiment_folders = os.listdir(dir_experiment)
sorted_folders = natsort.natsorted(experiment_folders)
ms_sessions = {}
# Iterate over the experiment folders
for folder in sorted_folders:
    if folder == ".DS_Store":  # Skip the .DS_Store file
        continue
    dir_folder_session = os.path.join(dir_experiment, folder)
    if not os.path.isdir(dir_folder_session):  # Skip non-directory items
        continue
    experiment = di.Experiment(dir_folder_session)
    experiment.save()
    sessions = experiment.get_sessions(subject_IDs='all', when='all')  # selects sessions from all subjects for all days
    # Iterate over the sessions
    for session in sessions:
        subject_id = session.subject_ID
        category_name = f'ms{subject_id}'
        # Check if the category exists in the dictionary
        if category_name not in ms_sessions:
            ms_sessions[category_name] = []
        # Append the session to the corresponding category
        ms_sessions[category_name].append(session)
ordered_ms_sessions = {}
keys = list(ms_sessions.keys())

for animal in keys:
    sessions = ms_sessions[animal]
    dateses = np.empty((len(sessions), 2), dtype=object)  # Use object dtype for storing session objects

    for i, session_object in enumerate(sessions):
        date = session_object.datetime_string
        dateses[i, 0] = session_object
        dateses[i, 1] = date

    #datetime_col = dateses[:, 1]
    #datetime_col = np.expand_dims(datetime_col, axis=1)
    sort_keys = np.zeros(dateses.shape[0], dtype=[('key', 'U19')])  # Assuming the second column has string values
    sort_keys['key'] = dateses[:, 1]
    sorted_indices = np.lexsort((sort_keys['key'],))
    sorted_array = dateses[sorted_indices]
    new_ordered_sesions_list = sorted_array[:, 0].tolist()
    #sort_order = natsort.natsorted(datetime_col)
    #sorted_indices = np.argsort(sort_order)
    #sorted_sessions = dateses[sorted_indices, 0]
    ordered_ms_sessions[animal] = new_ordered_sesions_list

#-----------------------------------------------------------------------------------------------------------------------
#poke board not responding clean
def animal_data(subject_number):
    return np.array(ordered_ms_sessions[f'ms{subject_number}'])
r"""
ordered_ms_sessions ={}
keys = list(ms_sessions.keys())
for i, animal in enumerate(ms_sessions):
    dates = np.zeros((24, 2))
    for i, session in enumerate(animal):
        date = session.datetime
        row_index = i
        session_object = ms_sessions[animal][i]
        dates[row_index] = [session_object,date]
    datetime_col = dates[:, 1]
    sorted_indices = np.argsort(datetime_col)
    sorted_sessions = dates[sorted_indices]
    sorted_just_sessions = sorted_sessions[:,0]
    ordered_ms_sessions[keys[i]]= []
    ordered_ms_sessions.keys[i].append(sorted_just_sessions)
"""


ms1 = ordered_ms_sessions['ms1']
ms2 = ordered_ms_sessions['ms2']
ms3 = ordered_ms_sessions['ms3']
ms4 = ordered_ms_sessions['ms4']
ms5 = ordered_ms_sessions['ms5']
ms6 = ordered_ms_sessions['ms6']
ms7 = ordered_ms_sessions['ms7']
ms8 = ordered_ms_sessions['ms8']
ms9 = ordered_ms_sessions['ms9']
ms10 = ordered_ms_sessions['ms10']
ms11 = ordered_ms_sessions['ms11']
ms12 = ordered_ms_sessions['ms12']
ms13 = ordered_ms_sessions['ms13']
ms14 = ordered_ms_sessions['ms14']
ms15 = ordered_ms_sessions['ms15']
ms16 = ordered_ms_sessions['ms16']
ms17 = ordered_ms_sessions['ms17']
ms18 = ordered_ms_sessions['ms18']
ms19 = ordered_ms_sessions['ms19']
ms20 = ordered_ms_sessions['ms20']
ms21 = ordered_ms_sessions['ms21']
ms22 = ordered_ms_sessions['ms22']
ms23 = ordered_ms_sessions['ms23']
ms24 = ordered_ms_sessions['ms24']
ms25 = ordered_ms_sessions['ms25']
ms26 = ordered_ms_sessions['ms26']

def is_even(num):
    return num % 2 == 0







