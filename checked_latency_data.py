import numpy as np
import extra_plots as ep

import pickle
import copy

# Load the Dictionary from a file
with open('stay_probability_NR.pkl', 'rb') as file:
    free_choice_after_omission_latency = pickle.load(file)

free_choice_after_omission_latency_combined_arrays = copy.deepcopy(free_choice_after_omission_latency)
r"""
# Assuming free_choice_after_omission_latency is a dictionary with arrays
for treatment in free_choice_after_omission_latency:
    for condition in free_choice_after_omission_latency[treatment]:
        if condition != 'TR':
            for animal in free_choice_after_omission_latency[treatment][condition]:
                if str(type(free_choice_after_omission_latency[treatment][condition][animal][0])) == "<class 'numpy.float64'>":
                    concatenated_array = []
                    for float in free_choice_after_omission_latency[treatment][condition][animal]:
                        concatenated_array.extend(float)
                else:
                    concatenated_array = np.concatenate(free_choice_after_omission_latency[treatment][condition][animal])
                free_choice_after_omission_latency_combined_arrays[treatment][condition][animal] = concatenated_array
"""
free_choice_after_omission_latency_animal_medians = copy.copy(free_choice_after_omission_latency_combined_arrays)


for treatment in free_choice_after_omission_latency_animal_medians:
    for condition in free_choice_after_omission_latency_animal_medians[treatment]:
        if condition != 'TR':
            median_condition_array =[]
            for animal in free_choice_after_omission_latency_animal_medians[treatment][condition]:
                median_condition_array.append(np.median(free_choice_after_omission_latency_combined_arrays[treatment][condition][animal]))
            median_condition_array = np.hstack((median_condition_array))
            free_choice_after_omission_latency_animal_medians[treatment][condition] = median_condition_array
        elif condition == 'TR':
            for animal in free_choice_after_omission_latency_animal_medians[treatment][condition]:
                animal_median_array = []
                for i, session in enumerate(free_choice_after_omission_latency_animal_medians[treatment][condition][animal]):
                    animal_median_array.append(np.median(free_choice_after_omission_latency_animal_medians[treatment][condition][animal][session]))
                animal_median_array = np.hstack((animal_median_array))
                free_choice_after_omission_latency_animal_medians[treatment][condition][animal] = animal_median_array
            sorted_dict = {k: free_choice_after_omission_latency_animal_medians[treatment][condition][k] for k in sorted(free_choice_after_omission_latency_animal_medians[treatment][condition], key=lambda x: int(x))}
            stacked_array = np.vstack(list(sorted_dict.values()))
            free_choice_after_omission_latency_animal_medians[treatment][condition] = stacked_array
r"""

free_choice_after_omission_latency_ILOR = copy.copy(free_choice_after_omission_latency_combined_arrays)
for treatment in free_choice_after_omission_latency_combined_arrays:
    for condition in free_choice_after_omission_latency_combined_arrays[treatment]:
        if condition != 'TR':
            for animal in free_choice_after_omission_latency_combined_arrays[treatment][condition]:
                free_choice_after_omission_latency_ILOR[treatment][condition][animal] = ep.remove_outliers_by_z_score(
                    free_choice_after_omission_latency_combined_arrays[treatment][condition][animal])
        elif condition == 'TR':
            for animal in free_choice_after_omission_latency_combined_arrays[treatment][condition]:
                for session in free_choice_after_omission_latency_combined_arrays[treatment][condition][animal]:
                    free_choice_after_omission_latency_ILOR[treatment][condition][
                        animal][session] = ep.remove_outliers_by_z_score(
                        free_choice_after_omission_latency_combined_arrays[treatment][condition][animal][session])
"""