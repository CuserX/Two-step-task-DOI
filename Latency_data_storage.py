import os
import numpy as np
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

all_second_step_latency = defaultdict(lambda: defaultdict(dict))

master_data_folder = r"C:\Users\Test\OneDrive\Documents\FHS_project_data\Experiment_1"
for group_condition_folder in os.listdir(master_data_folder):
    if str(group_condition_folder) != '.DS_Store':
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
                all_second_step_latency.setdefault(treatment, {})
                all_second_step_latency[treatment].setdefault(condition, {})

                data_piece = pl.session_latencies(session, 'second_step', FsplitF=False, )

                if subject_name in all_second_step_latency[treatment][condition]:
                    all_second_step_latency[treatment][condition][subject_name].append(data_piece)
                    np.concatenate([all_second_step_latency[treatment][condition][subject_name]], axis=0, dtype=object)
                else:
                    all_second_step_latency[treatment][condition][subject_name] = [data_piece]

        elif str(group_condition_folder)[9:11] == 'TR':
            condition = 'TR'
            if str(group_condition_folder)[12:15] == 'DOI':
                treatment = 'DOI'
            else:
                treatment = 'VEH'
            for session in sessions:
                subject_name = str(session.subject_ID)
                # Ensure treatment and condition keys exist
                all_second_step_latency.setdefault(treatment, {})
                all_second_step_latency[treatment].setdefault(condition, {})

                if subject_name in all_second_step_latency[treatment][condition]:
                    if len(all_second_step_latency[treatment][condition][subject_name]) == 3:
                        session_number = 10
                    elif len(all_second_step_latency[treatment][condition][subject_name]) == 4:
                        session_number = 11
                    elif len(all_second_step_latency[treatment][condition][subject_name]) == 3:
                        session_number = 12
                    else:
                        session_number = str(1 + len(all_second_step_latency[treatment][condition][subject_name]))
                else:
                    session_number = '1'


                data_piece = pl.session_latencies(session, 'second_step', FsplitF=False)
                all_second_step_latency[treatment][condition].setdefault(subject_name, {})

                all_second_step_latency[treatment][condition][subject_name][session_number] = data_piece
