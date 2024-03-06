import numpy as np
import extra_plots as ep

import pickle
import copy

def mean_without_outliers(data, z_threshold=2.0):
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    filtered_data = data[z_scores < z_threshold]
    return np.mean(filtered_data)
def mean_without_outliers_specifiedt(data, threshold):
    flat_data = [item for sublist in data for item in sublist]  # Flatten the nested list
    filtered_data = [value for value in flat_data if value <= threshold]
    return np.mean(filtered_data)

# Load the Dictionary from a file
with open('next_trial_choice_OR', 'rb') as file:
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

            for animal in free_choice_after_omission_latency_animal_medians[treatment][condition]:
                animal_median_array = []
                for i, session in enumerate(free_choice_after_omission_latency_animal_medians[treatment][condition][animal]):
                    animal_median_array.append(mean_without_outliers_specifiedt([free_choice_after_omission_latency_animal_medians[treatment][condition][animal][session]], 5000))
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