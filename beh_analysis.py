import os

import numpy
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


#-------------------------------------------------------------------------------------------------------------
# Utility
#-------------------------------------------------------------------------------------------------------------

def savefig(dir_folder, figname, pdf=True, svg=False, *args, **kwargs):
    ''' Saving a figure in a .pdf and vector (.svg) format '''
    #.tiff very low quality so not used
    if pdf == True:
        plt.savefig(os.path.join(dir_folder, figname + '.pdf'), *args, **kwargs)
      # if figname includes forbidden characters (: or .) the function will return an error
    if svg == True:
        plt.savefig(os.path.join(dir_folder, figname + '.svg'), *args, **kwargs)
    #plt.close(plt.gcf())
        

def _nans(shape, dtype=float):
    ''' Returns an array of nans of specified shape.'''
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def _double_exp_fit_error(params, mean_traj, p_0, p_e, t):
    ''' This function is called in _fit_exp_to_choice_traj '''
    return np.sum((mean_traj-_double_exp_choice_traj(params, p_0, p_e, t))**2)


def _double_exp_choice_traj(params, p_0, p_e, t):
    ''' This function is called in _double_exp_fit_error '''
    tau_f, tau_ratio, fs_mix = params
    tau_s = tau_f * tau_ratio
    return (1. - p_e) + (p_0 + p_e - 1.) * (fs_mix * np.exp(-t / tau_f) + (1 - fs_mix) * np.exp(-t / tau_s))


def _lag(x, i):
    ''' Applies lag of i trials to array x. '''
    x_lag = np.zeros(x.shape, x.dtype)
    if i > 0:
        x_lag[i:] = x[:-i]
    else:
        x_lag[:i] = x[-i:]
    return x_lag


def _get_data_to_analyse(session, transform_rew_state=True):
    ''' This function extract the necessary variables for various analyses. '''
    
    if hasattr(session, 'unpack_trial_data'):
        choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data(dtype=bool)
    else:
        choices = session.trial_data['choices'].astype(bool) # 1: left, 0: right
        transitions_AB = session.trial_data['transitions'].astype(bool) #1 for left-->up & right-->down (Type A), 0 for left-->down & right-->up (Type B)
        second_steps = session.trial_data['second_steps'].astype(bool) # 1: up, 0: down
        outcomes = session.trial_data['outcomes'].astype(bool) # 1: reward, 0: no reward
    
    trans_state = session.blocks['trial_trans_state']  # Trial by trial state of the transition matrix (A vs B); constant for our version of the task
    transitions_CR = transitions_AB == trans_state  # Trial by trial common or rare transitions (True: common; False: rare)
    transition_CR_x_outcome = transitions_CR == outcomes  # True: common and rewarded / rare and unrewarded
                                                            # False: common and unrewarded / rare and rewarded
    forced_choice_trials = ~session.trial_data['free_choice'] # tilde is a complement operator; any non-free choice = forced choice 


    # Transform session.blocks['trial_rew_state'] (coded as # 1: up, 0: down, 2: neutral) to new variable rew_state coded as 2: up, 1: neutral, 0: down
    if transform_rew_state == True: #default
        rew_state = np.array([session.blocks['trial_rew_state'][i] + 1 if session.blocks['trial_rew_state'][i] == 1 # up is 2 instead of 1
                              else session.blocks['trial_rew_state'][i] - 1 if session.blocks['trial_rew_state'][i] == 2 # neutral is 1 instead of 2 
                              else session.blocks['trial_rew_state'][i] for i in range(len(session.blocks['trial_rew_state']))]) # down stays the same (=0)
    else:
        rew_state = session.blocks['trial_rew_state']

    return (choices, transitions_AB, second_steps, outcomes, trans_state, transitions_CR, transition_CR_x_outcome, forced_choice_trials, rew_state)


def get_event_id(session, event):
  ''' Return the position in an array (NOT in time!) of a particular event on a particular session.
    Note: session.times can give you the time stamps of all events in a session
    Events:
    init_trial, 
    choice_state, choose_down, choose_left, choose_up, choose_right
    down_state, up_state, cue_up_state, cue_down_state,
    reward_cue, no_reward_cue, reward_down, reward_up
    reward_consumption, time_out,
    inter_trial_interval,
    poke_1 (=up), poke_1_out, poke_4 (=left), poke_4_out, poke_5 (=center), poke_5_out, poke_6 (=right), poke_6_out, poke_9 (=down), poke_9_out'''
  return np.where(np.asarray([session.events[i].name for i in range(len(session.events))]) == event)[0]



#-------------------------------------------------------------------------------------------------------------
# Session plot
#-------------------------------------------------------------------------------------------------------------

class exp_mov_ave:
    ''' Exponential moving average of choices class.
        This moving average is an average of choices (right/left) NOT a moving average of correct choices (CA calculated in task code). '''
    
    def __init__(self, tau, init_value=0):
        self.tau = tau
        self.init_value = init_value
        self.reset()

    def reset(self, init_value=None, tau=None):
        if tau:
            self.tau = tau
        if init_value:
            self.init_value = init_value
        self.value = self.init_value
        self._m = math.exp(-1./self.tau)
        self._i = 1 - self._m

    def update(self, sample):
        self.value = (self.value * self._m) + (self._i * sample)
        

def trials_plot(session):
    ''' Plots the choice moving average and correct blocks for a single session. '''
    
    # get trial data
    choices, transitions, second_steps, outcomes = session.unpack_trial_data(dtype=bool)
    
    # calculate choice moving average
    moving_average = exp_mov_ave(tau=8, init_value=0.5)
    moving_average_session = []
    for x in choices:
        moving_average.update(x)
        moving_average_session.append(moving_average.value)
    
    # get block info
    if hasattr(session, 'blocks'):
        for i in range(len(session.blocks['start_trials'])):
            # x position corresponding to times of blocks
            x = [session.blocks['start_trials'][i], session.blocks['end_trials'][i]]
            # y position coresponding to what choice was the correct one in a specific block
            if session.blocks['transition_states'][i] == 1:
                #for Type A transitions: right is correct for down blocks, left is correct for up blocks
                y = [0.25,0.75,0.5][session.blocks['reward_states'][i]]  
            else:
                #for Type B transitions: right is correct for up blocks, left is correct for down blocks
                y = [0.75,0.25,0.5][session.blocks['reward_states'][i]]
            # plot block lines 
            plt.plot(x, [y,y], 'gray', linewidth=3, alpha=0.6)
    
    # plot choice moving average 
    plt.plot(moving_average_session,'black', linewidth=2)    
    
    # formatting
    plt.xlabel('Trial number', fontsize=28, fontweight='bold')
    plt.yticks([0,0.5,1], [' R ', '', ' L '], fontsize=25)
    plt.ylim(0, 1)
    plt.xlim(0,len(choices))
    plt.xticks(fontsize=25)
    plt.ylabel('Choice \nmoving average', fontsize=28, fontweight='bold')
    

def session_structure_plot(session, title):
    ''' Plots the transition and reward block structure for a single session. '''
    
    # Setting the colour scheme
    col_transA='orange'
    col_transB='#6FC8CE'
    col_rewU='#1DC6FE'
    col_rewD='#A589D3'
    
    # reward states
    x = np.vstack([session.blocks['start_trials'], 
                   session.blocks['end_trials']]).T.reshape([1,-1])[0]
    rs = np.vstack([session.blocks['reward_states'], 
                   session.blocks['reward_states']]).T.reshape([1,-1])[0]
    rew_up = np.array([0.2,0.8,0.5])[rs] # up port reward probability
    rew_down = np.array([0.8,0.2,0.5])[rs] # down port reward probability 
    
    # transition states
    ts = np.vstack([session.blocks['transition_states'], 
                    session.blocks['transition_states']]).T.reshape([1,-1])[0]
    trans_prob = np.array([0.2,0.8])[ts] # probability of type A transition
    
    # plot init
    plt.figure(figsize=[6.0, 6.6]).clf()
    gs = GridSpec(3, 1, height_ratios=[0.5, 0.5, 1.2]) # creates a figure with 3 subplots in 1 column with the first two subplots of equal size and third subplot approx. 2x the size
    
    
    # Transition probabilities plot
    plt.subplot(gs[0]) 
    if title != '': # if empty title not requsted, set title as...
        plt.title('Subject: ms{}, session_type: {}, date: {}'.format(session.subject_ID, session.datetime_string[:10]), fontsize=9, fontweight='bold') # where title input to the function specifies session type
    plt.plot(x,  trans_prob,col_transA, linewidth=3, alpha=0.8)
    plt.plot(x,1-trans_prob,col_transB, linewidth=3, alpha=0.8) # Type B transiiton probability is 1-p(Type A)
    plt.ylim(0,1)
    plt.xlim(x[0],x[-1])
    plt.ylabel('Transition \nprobabilities', fontsize=28, fontweight='bold')
    #plt.text(1,0.25, r"Type A", fontsize=7, fontweight='bold', color=col_transA, style='italic') #location may be diff depending on the session and common transition
    #plt.text(1,0.85, r"Type B", fontsize=7, fontweight='bold', color=col_transB, style='italic') #location may be diff depending on the session and common transition
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.xticks(fontsize=25)
    plt.yticks([0,0.5,1],  fontsize=25)
    plt.axhline(linewidth=3, color='k')
    plt.axvline(x=0, linewidth=3, color='k')
    plt.tick_params(length=16, width=2)

    
    # Reward probabilities plot
    plt.subplot(gs[1])
    plt.plot(x,rew_down,col_rewD, linewidth=3, alpha=0.8)
    plt.plot(x,rew_up,col_rewU, linewidth=3, alpha=0.8)
    plt.ylim(0,1)
    plt.xlim(x[0],x[-1])
    plt.ylabel('Reward \nprobabilities',  fontsize=28, fontweight='bold')
    #plt.text(1,0.25, r"Up", fontsize=7, fontweight='bold', color=col_rewU, style='italic') #location may be diff depending on the session
    #plt.text(1,0.85, r"Down", fontsize=7, fontweight='bold', color=col_rewD, style='italic') #location may be diff depending on the session
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.xticks(fontsize=25)
    plt.yticks([0,0.5,1],  fontsize=25)
    plt.axhline(linewidth=3, color='k')
    plt.axvline(x=0, linewidth=3, color='k')
    plt.tick_params(length=16, width=2)
    
    # Choices plot
    plt.subplot(gs[2])
    trials_plot(session)
    plt.xlim(x[0],x[-1])
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.xlabel('Trial number',  fontsize=28, fontweight='bold')
    plt.axhline(linewidth=3, color='k')
    plt.axvline(x=0, linewidth=3, color='k')
    plt.tick_params(length=16, width=2)

    
    plt.subplots_adjust( #left=0.2,
                    #bottom=0.2, 
                    #right=0.95, 
                    #top=0.95, 
                    wspace=0.4, 
                    hspace=0.5)
    
    
#-------------------------------------------------------------------------------------------------------------
# Stay probability
#-------------------------------------------------------------------------------------------------------------
def my_stay_prob(session, block_type='all', selection_type='all', select_n=0, forced_choice=False):
    stay_probability = np.array([])
    if len(session.blocks[
               'trial_rew_state']) > 11:  # Analyse if there were more than 10 trials in a block (excludes the final block if not enough trials availbale for adaptation)
        positions = session.select_trials(selection_type=selection_type, select_n=select_n,
                                          block_type=block_type)  # select only the trials specified by block_type and and selection_type
        choices = session.trial_data['choices']
        outcomes = session.trial_data['outcomes']
        transitions = session.trial_data['transitions']
        transition_type = session.blocks['trial_trans_state']
        free_choice_trials = session.trial_data['free_choice']
        rew_state = session.blocks['trial_rew_state']

        # Positions of trial type
        rew_common = np.where((transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
        rew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
        nonrew_common = np.where((transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
        nonrew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]

        if forced_choice == False:  # eliminate forced choice trials
            rew_common = [x for x in rew_common if
                          ((x + 1 in np.where(free_choice_trials == True)[0]) and (
                                      x in np.where(positions == True)[0]))]
            rew_rare = [x for x in rew_rare if
                        ((x + 1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            nonrew_common = [x for x in nonrew_common if
                             ((x + 1 in np.where(free_choice_trials == True)[0]) and (
                                         x in np.where(positions == True)[0]))]
            nonrew_rare = [x for x in nonrew_rare if
                           ((x + 1 in np.where(free_choice_trials == True)[0]) and (
                                       x in np.where(positions == True)[0]))]

        else:
            forced_before = [i + 1 for i, (f, f1) in enumerate(zip(free_choice_trials, free_choice_trials[1:]))
                             if (f == False and f1 == True)]
            rew_common = [x for x in rew_common if
                          ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (
                                      x in np.where(positions == True)[0]))]
            rew_rare = [x for x in rew_rare if
                        ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (
                                    x in np.where(positions == True)[0]))]
            nonrew_common = [x for x in nonrew_common if
                             ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (
                                         x in np.where(positions == True)[0]))]
            nonrew_rare = [x for x in nonrew_rare if
                           ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (
                                       x in np.where(positions == True)[0]))]

        # Get probability of stay (averaged across all sessions)
        stay = (choices[1:] == choices[:-1]).astype(int)  # 1: stay, 0: switch
        stay_probs = np.zeros(4)
        stay_probsRC = np.nanmean(stay[rew_common])  # Rewarded, common transition.
        stay_probsRR = np.nanmean(stay[rew_rare])  # Rewarded, rare transition.
        stay_probsNC = np.nanmean(stay[nonrew_common])  # Non-rewarded, common transition.
        stay_probsNR = np.nanmean(stay[nonrew_rare])  # Non-rewarded, rare transition.

    return stay_probsRC, stay_probsRR, stay_probsNC, stay_probsNR


def _compute_stay_probability_sessions(sessions_sub, selection_type, select_n, block_type, forced_choice=False):
    ''' This function is called in compute_stay_probability.
        It calculates the probability of repeating a choice depending on the transition type and outcome of the previous choice.
        block type: 'all' / 'neutral' / 'non_neutral'
        selection_type: where to select trials; 'start' (after block transition) / 'start_1' (eliminates first trial) /'end' (before block transition) 
                                                / 'all' (=DEFAULT)
                                                / 'xmid' (middle of the block)
        select_n: number of trials to select if selection_type != 'all'; set to 0, not needed for this task set up '''
    
    stay_probability = np.array([])

    for session in sessions_sub:
        if len(session.blocks['trial_rew_state']) > 11:  # Analyse if there were more than 10 trials in a block (excludes the final block if not enough trials availbale for adaptation)
            positions = session.select_trials(selection_type=selection_type, select_n=select_n, block_type=block_type) # select only the trials specified by block_type and and selection_type
            choices = session.trial_data['choices']
            outcomes = session.trial_data['outcomes']
            transitions = session.trial_data['transitions']
            transition_type = session.blocks['trial_trans_state']
            free_choice_trials = session.trial_data['free_choice']
            rew_state = session.blocks['trial_rew_state']

            # Positions of trial type
            rew_common = np.where((transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
            rew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
            nonrew_common = np.where((transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
            nonrew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]

            if forced_choice == False: # eliminate forced choice trials
                rew_common = [x for x in rew_common if
                              ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
                rew_rare = [x for x in rew_rare if
                            ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
                nonrew_common = [x for x in nonrew_common if
                                 ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
                nonrew_rare = [x for x in nonrew_rare if
                               ((x+1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]

            else:
                forced_before = [i+1 for i, (f, f1) in enumerate(zip(free_choice_trials, free_choice_trials[1:]))
                                 if (f==False and f1 == True)]
                rew_common = [x for x in rew_common if
                              ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (x in np.where(positions == True)[0]))]
                rew_rare = [x for x in rew_rare if
                            ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (x in np.where(positions == True)[0]))]
                nonrew_common = [x for x in nonrew_common if
                                 ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (x in np.where(positions == True)[0]))]
                nonrew_rare = [x for x in nonrew_rare if
                               ((x in np.where(free_choice_trials == True)[0]) and (x in forced_before) and (x in np.where(positions == True)[0]))]

            # Get probability of stay (averaged across all sessions)
            stay = (choices[1:] == choices[:-1]).astype(int) # 1: stay, 0: switch
            stay_probs = np.zeros(4)
            stay_probs[0] = np.nanmean(stay[rew_common])  # Rewarded, common transition.
            stay_probs[1] = np.nanmean(stay[rew_rare])  # Rewarded, rare transition.
            stay_probs[2] = np.nanmean(stay[nonrew_common])  # Non-rewarded, common transition.
            stay_probs[3] = np.nanmean(stay[nonrew_rare])  # Non-rewarded, rare transition.
            stay_probability = np.vstack([stay_probability, stay_probs]) if stay_probability.size else stay_probs

    # get the stay probaility means per subject
    stay_probability_mean_sub = np.nanmean(stay_probability, axis=0) if len(stay_probability.shape) > 1 else stay_probability

    return stay_probability_mean_sub, stay_probability


def compute_stay_probability(sessions, block_type='all', selection_type='all', select_n=0, forced_choice=False, subjects=[]):
    ''' This function is called in plot_stay_probability.
        It calculates the mean and sem of stay probabilities calculated by calling the function _compute_stay_probability_sessions '''
    
    stay_probability_all_means = np.array([])
    sessions_sub = []
    if subjects == []: # get subject IDs
        subjects = list(set([sessions[i].subject_ID for i in range(len(sessions))]))
    for sub in subjects:
        idx_sub = np.where([sessions[i].subject_ID == sub for i in range(len(sessions))])[0]
        sessions_sub.append([sessions[x] for x in idx_sub])
    
    # calculate stay probability per subject
    stay_prob_compute = pp.map(partial(_compute_stay_probability_sessions,selection_type=selection_type,
                                     block_type=block_type, select_n=select_n, forced_choice=forced_choice), sessions_sub)
    stay_probability_all_means, stay_prob_per_session_all = zip(*stay_prob_compute)
    stay_probability_all_means = np.vstack(stay_probability_all_means)
    
    # calculate stay probability average across all subjects
    stay_probability_mean = np.nanmean(stay_probability_all_means, axis=0) \
        if len(stay_probability_all_means.shape) > 1 else stay_probability_all_means
    # get standard error of mean 
    stay_probability_sem = stats.sem(stay_probability_all_means, axis=0, nan_policy='omit') \
        if len(stay_probability_all_means.shape) > 1 else 0

    return (stay_probability_all_means, stay_probability_mean, stay_probability_sem, stay_prob_per_session_all)


def plot_stay_probability(stay_probability_all, stay_probability_mean, stay_probability_sem, scatter=True,):
    ''' Plot of stay probabilities averaged across all sessions per animal, then across animals.
        if scatter=True subject means are plotted'''
    
    plt.figure(figsize=[1.8, 2.2]).clf()
    colors = ['orange', '#6FC8CE', '#1DC6FE', '#A589D3']
    plt.bar(np.arange(1, 5), stay_probability_mean, yerr=stay_probability_sem,
          error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color = colors, alpha=0.8, fill=True, zorder=-1) 
    if scatter == True:
        y = stay_probability_all
        x = np.random.normal(0, 0.12, size=len(stay_probability_all)) # to distribute the dots randomly across the length of a bar 
        for i in np.arange(1, 5):
            plt.scatter(x+i, y.T[i-1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4, zorder=1) # zorder=1 to bring to front
    plt.xlim(0.75, 5)
    plt.xticks([-0.25, 1, 2, 3, 4], ['\nTrans.\nRew.', '\nC\n+', '\nR\n+', '\nC\n-', '\nR\n-'],fontsize=8, fontweight='bold')
    plt.ylabel('Stay probability', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0 ,0.2, 0.4, 0.6, 0.8, 1.0],  fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)
#--------------------------------------------------------------------------------------------------------------
# regret investigation

def regret_calculation(session):
    sublists = []
    current_sublist = []
    for event in session.events:
        if event.name == 'choice_state':
            if current_sublist:
                sublists.append(current_sublist)
                current_sublist = []
        else:
            # Add the event to the current sublist
            current_sublist.append(event)
    # Add the last sublist if it's not empty
    if current_sublist:
        sublists.append(current_sublist)
    sublists = sublists[1:]
    regret_trials = 0
    for sublist in sublists:
        sublist_events = []
        for i,event in enumerate(sublist):
            sublist_events.append(event.name)
            if event.name == 'choose_left':
                regret_poke = 'poke_6'
            elif event.name == 'choose_right':
                regret_poke = 'poke_4'
        if regret_poke in sublist_events:
            regret_trials = regret_trials + 1

    fraction_regretted = regret_trials/len(sublists)

    return float(fraction_regretted)



#events after choice availibility
def pokes_after_structure(session, transition, array=True):
    sublists = []
    current_sublist = []
    for event in session.events:
        if event.name == 'choice_state':
            if current_sublist:
                sublists.append(current_sublist)
                current_sublist = []
        else:
            # Add the event to the current sublist
            current_sublist.append(event)
    # Add the last sublist if it's not empty
    if current_sublist:
        sublists.append(current_sublist)
    sublists = sublists[1:]

    transition_types = session.trial_data['transitions']

    common_trans_sublists = []
    rare_trans_sublists =[]
    for lst, mask in zip(sublists, transition_types):
        if mask:
            common_trans_sublists.append(lst)
        else:
            rare_trans_sublists.append(lst)

    poke_5_times = []
    chosen_side_poke = []
    unchosen_side_poke = []
    outcome_port_poke = []
    nonoutcome_port_poke = []
    if transition == 'all':
        for i, sublist in enumerate(sublists):
            for event in sublist:  # assigns poke 5 times and defines paradigme structure
                if event.name == 'poke_5':
                    relative_event_time = event.time - sublist[0].time
                    poke_5_times.append(relative_event_time)
                elif event.name == 'choose_left':
                    chosen_side = 'poke_4'
                    unchosen_side = 'poke_6'
                elif event.name == 'choose_right':
                    chosen_side = 'poke_6'
                    unchosen_side = 'poke_4'
                elif event.name == 'cue_up_state':
                    outcome_port = 'poke_1'
                    nonoutcome_port = 'poke_9'
                elif event.name == 'cue_down_state':
                    outcome_port = 'poke_9'
                    nonoutcome_port = 'poke_1'

            for event in sublist:
                if event.name == chosen_side:
                    relative_event_time = event.time - sublist[0].time
                    chosen_side_poke.append(relative_event_time)
                elif event.name == unchosen_side:
                    relative_event_time = event.time - sublist[0].time
                    unchosen_side_poke.append(relative_event_time)
                elif event.name == outcome_port:
                    relative_event_time = event.time - sublist[0].time
                    outcome_port_poke.append(relative_event_time)
                elif event.name == nonoutcome_port:
                    relative_event_time = event.time - sublist[0].time
                    nonoutcome_port_poke.append(relative_event_time)
        if array == True:
            return np.array(poke_5_times), np.array(chosen_side_poke), np.array(unchosen_side_poke), np.array(
                outcome_port_poke), np.array(nonoutcome_port_poke), len(sublists)
        else:
            return poke_5_times, chosen_side_poke, unchosen_side_poke, outcome_port_poke, nonoutcome_port_poke, len(sublists)

    if transition == 'common':
        for i, sublist in enumerate(common_trans_sublists):
            for event in sublist: # assigns poke 5 times and defines paradigme structure
                if event.name == 'poke_5':
                    relative_event_time = event.time - sublist[0].time
                    poke_5_times.append(relative_event_time)
                elif event.name == 'choose_left':
                    chosen_side = 'poke_4'
                    unchosen_side = 'poke_6'
                elif event.name == 'choose_right':
                    chosen_side = 'poke_6'
                    unchosen_side = 'poke_4'
                elif event.name == 'cue_up_state':
                    outcome_port = 'poke_1'
                    nonoutcome_port = 'poke_9'
                elif event.name == 'cue_down_state':
                    outcome_port = 'poke_9'
                    nonoutcome_port = 'poke_1'

            for event in sublist:
                if event.name == chosen_side:
                    relative_event_time = event.time - sublist[0].time
                    chosen_side_poke.append(relative_event_time)
                elif event.name == unchosen_side:
                    relative_event_time = event.time - sublist[0].time
                    unchosen_side_poke.append(relative_event_time)
                elif event.name == outcome_port:
                    relative_event_time = event.time - sublist[0].time
                    outcome_port_poke.append(relative_event_time)
                elif event.name == nonoutcome_port:
                    relative_event_time = event.time - sublist[0].time
                    nonoutcome_port_poke.append(relative_event_time)
        if array == True:
            return np.array(poke_5_times), np.array(chosen_side_poke), np.array(unchosen_side_poke), np.array(outcome_port_poke), np.array(nonoutcome_port_poke), len(common_trans_sublists)
        else:
            return poke_5_times, chosen_side_poke, unchosen_side_poke, outcome_port_poke, nonoutcome_port_poke, len(common_trans_sublists)
    elif transition == 'rare':
        for i, sublist in enumerate(rare_trans_sublists):
            for event in sublist:  # assigns poke 5 times and defines paradigme structure
                if event.name == 'poke_5':
                    relative_event_time = event.time - sublist[0].time
                    poke_5_times.append(relative_event_time)
                elif event.name == 'choose_left':
                    chosen_side = 'poke_4'
                    unchosen_side = 'poke_6'
                elif event.name == 'choose_right':
                    chosen_side = 'poke_6'
                    unchosen_side = 'poke_4'
                elif event.name == 'cue_up_state':
                    outcome_port = 'poke_1'
                    nonoutcome_port = 'poke_9'
                elif event.name == 'cue_down_state':
                    outcome_port = 'poke_9'
                    nonoutcome_port = 'poke_1'

            for event in sublist:
                if event.name == chosen_side:
                    relative_event_time = event.time - sublist[0].time
                    chosen_side_poke.append(relative_event_time)
                elif event.name == unchosen_side:
                    relative_event_time = event.time - sublist[0].time
                    unchosen_side_poke.append(relative_event_time)
                elif event.name == outcome_port:
                    relative_event_time = event.time - sublist[0].time
                    outcome_port_poke.append(relative_event_time)
                elif event.name == nonoutcome_port:
                    relative_event_time = event.time - sublist[0].time
                    nonoutcome_port_poke.append(relative_event_time)
        if array == True:
            return np.array(poke_5_times), np.array(chosen_side_poke), np.array(unchosen_side_poke), np.array(
                outcome_port_poke), np.array(nonoutcome_port_poke), np.array(len(rare_trans_sublists))
        else:
            return poke_5_times, chosen_side_poke, unchosen_side_poke, outcome_port_poke, nonoutcome_port_poke, len(rare_trans_sublists)
def pokes_after_structure_composite(sessions, transition):
    poke_5_times = []
    chosen_side_poke = []
    unchosen_side_poke = []
    outcome_port_poke = []
    nonoutcome_port_poke = []
    trials = 0

    for session in sessions:
            a, b, c, d, e, f = pokes_after_structure(session, transition)
            poke_5_times.extend(a)
            chosen_side_poke.extend(b)
            unchosen_side_poke.extend(c)
            outcome_port_poke.extend(d)
            nonoutcome_port_poke.extend(e)
            trials = trials + f
    return np.array(poke_5_times), np.array(chosen_side_poke), np.array(unchosen_side_poke), np.array(outcome_port_poke), np.array(nonoutcome_port_poke), trials

#-------------------------------------------------------------------------------------------------------------
# Latency
#-------------------------------------------------------------------------------------------------------------

def session_latencies(session, type, FsplitF=False):
  '''
  return latency between events
  type: 'start', 'choice', 'second-step', 'ITI'
  FsplitF if true removes forced trials
  '''
  """if type == 'start': # time between initiation and poking the center port 
    times_e1, times_e2 = plp.get_times_consecutive_events(session, 'init_trial', 'poke_5', ['init_trial', 'poke_5'])"""
  if FsplitF == False:
    if type == 'start':  # time between centre port availiblity and centre press
        events = ['init_trial', 'poke_5']
        all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                               if session.events[i].name in events])
        times_e1 = [session.events[x][0] for x in all_id[::2]]
        times_e2 = [session.events[x][0] for x in all_id[1::2]]
    elif type == 'choice':  # time between choice state and correct choice
        events = ['choice_state', 'choose_right', 'choose_left']
        all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                 if session.events[i].name in events])
        times_e1 = [session.events[x][0] for x in all_id[::2]]
        times_e2 = [session.events[x][0] for x in all_id[1::2]]
    elif type == 'second_step':  # time between step 2 and step 2 state selection
        events = ['up_state', 'down_state', 'choose_up', 'choose_down']
        all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                 if session.events[i].name in events])
        times_e1 = [session.events[x][0] for x in all_id[::2]]
        times_e2 = [session.events[x][0] for x in all_id[1::2]]
    elif type == 'ITI':
        times_e1, times_e2 = plp.get_times_consecutive_events(session, 'inter_trial_interval', 'init_trial',
                                                          ['inter_trial_interval', 'init_trial'])
    elif type == 'ITI-choice':  # time between initiation and step 1 choice
        events = ['inter_trial_interval', 'choose_right', 'choose_left']
        all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                 if session.events[i].name in events])
        times_e1 = [session.events[x][0] for x in all_id[::2]]
        times_e2 = [session.events[x][0] for x in all_id[1::2]]
    elif type == 'ITI-start':  # time between initiation and step 1
        times_e1, times_e2 = plp.get_times_consecutive_events(session, 'inter_trial_interval', 'choice_state',
                                                          ['inter_trial_interval', 'choice_state'])

    latency = [e2 - e1 for e1, e2 in zip(times_e1, times_e2)][:len(times_e1)]

    return np.asarray(latency)

  elif FsplitF == True:
      if type == 'start':  # time between centre port availiblity and centre press
          events = ['init_trial', 'poke_5']
          all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
          times_e1 = [session.events[x][0] for x in all_id[::2]]
          times_e2 = [session.events[x][0] for x in all_id[1::2]]

          latency = [e2 - e1 for e1, e2 in zip(times_e1, times_e2)][:len(times_e1)]

          return np.asarray(latency)

      elif type == 'choice':  # time between choice state and correct choice
          freeC_latency = []
          forcedC_latency = []

          events = ['choice_state', 'choose_right', 'choose_left']
          all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
          times_e1 = [session.events[x][0] for x in all_id[::2]]
          times_e2 = [session.events[x][0] for x in all_id[1::2]]

          latency_choice = [e2 - e1 for e1, e2 in zip(times_e1, times_e2)][:len(times_e1)]

          step1freeC = np.array(session.trial_data['free_choice'], dtype=np.bool)  # Convert to boolean array

          Lstep1_completed_trials = latency_choice[:len(step1freeC)]  # removes unfinished trials

          step1freeC_indices = np.where(step1freeC)[0] #converts from a boolean array
          freeC_latency.extend([Lstep1_completed_trials[i] for i in step1freeC_indices])

          forcedC_indices = np.where(~step1freeC)[0] #converts from a boolean array
          forcedC_latency.extend([Lstep1_completed_trials[i] for i in forcedC_indices])

          return np.array(freeC_latency), np.array(forcedC_latency)
      elif type == 'second_step': # time between step 2 and step 2 state selection
          freeC_latency = []
          forcedC_latency = []

          events = ['up_state', 'down_state', 'choose_up', 'choose_down']
          all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                       if session.events[i].name in events])
          times_e1 = [session.events[x][0] for x in all_id[::2]]
          times_e2 = [session.events[x][0] for x in all_id[1::2]]

          latency_second_step = [e2 - e1 for e1, e2 in zip(times_e1, times_e2)][:len(times_e1)]

          second_step_freeC = np.array(session.trial_data['free_choice'], dtype=bool) # generate and convert to bool array

          Second_step_completed_trials = latency_second_step[:len(second_step_freeC)]  # removes unfinished trials

          second_step_freeC_indices = np.where(second_step_freeC)[0]  # converts from a boolean array
          freeC_latency.extend([Second_step_completed_trials[i] for i in second_step_freeC_indices]) #extends aray with value from second step completed trials with index of second step free choice

          second_step_forcedC_indices = np.where(~second_step_freeC)[0]  # converts from a boolean array
          forcedC_latency.extend([Second_step_completed_trials[i] for i in second_step_forcedC_indices]) #extends aray with value from second step completed trials with index of not second step free choice

          return np.array(freeC_latency), np.array(forcedC_latency)
      elif type == 'ITI-choice':
          freeC_latency = []
          forcedC_latency = []

          events = ['inter_trial_interval', 'choose_right', 'choose_left']
          all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                                        if session.events[i].name in events])
          times_e1 = [session.events[x][0] for x in all_id[::2]]
          times_e2 = [session.events[x][0] for x in all_id[1::2]]

          latency_ITI_to_choice = [e2 - e1 for e1, e2 in zip(times_e1, times_e2)][:len(times_e1)]

          free_choice = np.array(session.trial_data['free_choice'], dtype=bool) #generates an array of free choices

          LITI_choice_completed_trials = latency_ITI_to_choice[:len(free_choice)] #removes unfinished trials

          ITIC_freeC_indicies = np.where(free_choice)[0] #converts bool to anarray
          freeC_latency.extend([LITI_choice_completed_trials[i] for i in ITIC_freeC_indicies])

          ITIC_forcedC_indicies = np.where(~free_choice)[0]  # converts bool to anarray
          forcedC_latency.extend([LITI_choice_completed_trials[i] for i in ITIC_forcedC_indicies])

          return np.array(freeC_latency), np.array(forcedC_latency)

def Choice_latency_after_free_choice_reward(session, split_transitions=False):
    transitions = session.trial_data['transitions']
    free_choice = session.trial_data['free_choice']
    outcomes = session.trial_data['outcomes']
    choice_latency = session_latencies(session, 'choice')
    choice_latency = choice_latency[0:len(outcomes)]
    free_choice_after_free_choice = np.copy(free_choice)
    for i, bool in enumerate(free_choice):
        if i != len(free_choice)-1:
            if bool == False:
                free_choice_after_free_choice[i+1] = False
    outcomes_bool = outcomes.astype(bool)
    rewarded_trials = outcomes_bool
    free_choice_and_rewarded_trials = free_choice_after_free_choice & rewarded_trials

    if split_transitions ==True:
        common_transitions = transitions.astype(bool)
        rare_transitions = ~common_transitions
        free_choice_and_rewarded_common_trials = free_choice_and_rewarded_trials & common_transitions
        free_choice_and_rewarded_rare_trials = free_choice_and_rewarded_trials & rare_transitions
        free_choice_and_rewarded_common_latency = choice_latency[free_choice_and_rewarded_common_trials]
        free_choice_and_rewarded_rare_latency = choice_latency[free_choice_and_rewarded_rare_trials]

        return free_choice_and_rewarded_common_latency, free_choice_and_rewarded_rare_latency
    else:
        free_choice_and_nonrewarded_latency = choice_latency[free_choice_and_rewarded_trials]
        return free_choice_and_nonrewarded_latency
def Choice_latency_after_free_choice_ommision(session, split_transitions=False):
    transitions = session.trial_data['transitions']
    free_choice = session.trial_data['free_choice']
    outcomes = session.trial_data['outcomes']
    choice_latency = session_latencies(session, 'choice')
    choice_latency = choice_latency[0:len(outcomes)]
    free_choice_after_free_choice = np.copy(free_choice)
    for i, bool in enumerate(free_choice):
        if i != len(free_choice)-1:
            if bool == False:
                free_choice_after_free_choice[i+1] = False
    outcomes_bool = outcomes.astype(bool)
    nonrewarded_trials = ~outcomes_bool
    free_choice_and_nonrewarded_trials = free_choice_after_free_choice & nonrewarded_trials

    if split_transitions ==True:
        common_transitions = transitions.astype(bool)
        rare_transitions = ~common_transitions
        free_choice_and_nonrewarded_common_trials = free_choice_and_nonrewarded_trials & common_transitions
        free_choice_and_nonrewarded_rare_trials = free_choice_and_nonrewarded_trials & rare_transitions
        free_choice_and_nonrewarded_common_latency = choice_latency[free_choice_and_nonrewarded_common_trials]
        free_choice_and_nonrewarded_rare_latency = choice_latency[free_choice_and_nonrewarded_rare_trials]

        return free_choice_and_nonrewarded_common_latency, free_choice_and_nonrewarded_rare_latency
    else:
        free_choice_and_nonrewarded_latency = choice_latency[free_choice_and_nonrewarded_trials]
        return free_choice_and_nonrewarded_latency


def second_step_latency_after_transition_type(type, session):
    transitions = session.trial_data['transitions']
    free_choice = session.trial_data['free_choice']
    second_step_latency = session_latencies(session, 'second_step')
    if type == 'rare':
        common_transitions = transitions.astype(bool)
        transitions = ~common_transitions
    transitions = transitions.astype(bool)
    transitions = transitions[0:len(second_step_latency)]
    second_step_latency = second_step_latency[0:len(transitions)]
    transition_second_step_latency = second_step_latency[transitions]
    return np.array(transition_second_step_latency)

def all_folder_latency(data, type):
    """returns median latencies for each subject acording to stated latency type """
    dir_folder_session = data
    experiment = di.Experiment(dir_folder_session)
    experiment.save()
    sessions = experiment.get_sessions(subject_IDs='all', when='all')

    IDSpre = []
    for Session in sessions:
        sessionID = Session.subject_ID
        IDSpre.append(sessionID)
    IDSpre = np.array(list(set(IDSpre)))
    IDS = IDSpre.astype(int)
    subject_no = len(IDS)
    if type == 'start':
        all_medians = np.zeros(subject_no)
        for i, subject_id in enumerate(IDS):
            sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
            all_subjectL = []
            for session in sessions:
                latency = session_latencies(session, 'start', FsplitF=False)
                all_subjectL.append(latency)
            all_subjectLcon =np.concatenate(all_subjectL).tolist()
            sub_med = np.median(all_subjectLcon)
            all_medians[i] =sub_med
        return np.array(all_medians), np.mean(all_medians)

    elif type != 'start':
        all_free_medians = np.zeros(subject_no)  # Array to store free choice medians for each subject
        all_forced_medians = np.zeros(subject_no)  # Array to store forced choice medians for each subject
        for i, subject_id in enumerate(IDS):  # sets up iteration per subject
            sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
            all_subject_freeL = []
            all_subject_forcedL = []
            for session in sessions:  # extracts free and forced latencies per session
                if type == 'choice':
                    free_latency, forced_latency = session_latencies(session,'choice', FsplitF =True,)
                elif type == 'second_step':
                    free_latency, forced_latency = session_latencies(session,'second_step', FsplitF=True,)
                elif type == 'ITI-Choice':
                   free_latency, forced_latency = session_latencies(session, 'ITI-choice', FsplitF=True,)
                all_subject_freeL.append(free_latency)
                all_subject_forcedL.append(forced_latency)

            all_subject_freeLcon = np.concatenate(all_subject_freeL).tolist()  # consentrates list of arrays into one useable list
            all_subject_forcedLcon = np.concatenate(all_subject_forcedL).tolist()  # consentrates list of arrays into one useable list

            free_sub_med = np.median(all_subject_freeLcon)  # calculates subject median and stores it to list value
            forced_sub_med = np.median(all_subject_forcedLcon)  # calculates subject median and stores it to list value
            all_free_medians[i] = free_sub_med  # adds subject median to all medains list
            all_forced_medians[i] = forced_sub_med  # adds subject median all median lsit

        return np.array(all_free_medians), np.mean(all_free_medians), np.array(all_forced_medians), np.mean(all_forced_medians)

def plot_comparitive_latency(VEH_free_medians, VEH_forced_medians, DOI_free_medians, DOI_forced_medians, scatter = True, title = {}):
    #adjusting the inputs-----------------------------------------------------------------------------------------------
    R_VEH_free_medians = np.transpose(VEH_free_medians)
    R_VEH_forced_medians = np.transpose(VEH_forced_medians)
    R_DOI_free_medians = np.transpose(DOI_free_medians)
    R_DOI_forced_medians = np.transpose(DOI_forced_medians)
    MedianL_all = np.column_stack((R_VEH_free_medians, R_VEH_forced_medians, R_DOI_free_medians, R_DOI_forced_medians))
    MedianL_sem = stats.sem(MedianL_all, axis=0, nan_policy='omit') \
        if len(MedianL_all.shape) > 1 else 0
    MedianL_mean = np.nanmean(MedianL_all, axis=0) \
        if len(MedianL_all.shape) > 1 else MedianL_all
    #generating the plot -----------------------------------------------------------------------------------------------
    plt.figure(figsize=[3.6, 4.4]).clf()
    colors = ['orange', '#6FC8CE', '#1DC6FE', '#A589D3']
    plt.bar(np.arange(1, 5), MedianL_mean, yerr=MedianL_sem,
            error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color=colors, alpha=0.8, fill=True, zorder=-1)
    if scatter == True:
        y = MedianL_all
        x = np.random.normal(0, 0.12, size=len(MedianL_all))  # to distribute the dots randomly across the length of a bar
        for i in np.arange(1, 5):
            plt.scatter(x + i, y.T[i - 1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4,
                        zorder=1)  # zorder=1 to bring to front
    plt.title(title)
    plt.xlim(0.75, 5)
    plt.xticks([-0.25, 1, 2, 3, 4], ['\nGroup.\nchoice state', '\nVEH\nfree', '\nVEH\nforced', '\nDOI\nfree', '\nDOI\nforced'], fontsize=2,
               fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0, 100, 200, 300], fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)


def session_transition_types_latency(session,block_type, forced_choice=False,):
    positions = session.select_trials('all', 'all', first_n_mins=False,block_type=block_type, )  # select only the trials specified by block_type and and selection_type
    choices = session.trial_data['choices']
    outcomes = session.trial_data['outcomes']
    transitions = session.trial_data['transitions']
    transition_type = session.blocks['trial_trans_state']
    free_choice_trials = session.trial_data['free_choice']
    rew_state = session.blocks['trial_rew_state']

    # Positions of trial type
    rew_common = np.where((transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
    rew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
    nonrew_common = np.where((transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
    nonrew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
    #
    if forced_choice == False:  # eliminate forced choice trials
        rew_common = [x for x in rew_common if
                      ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        rew_rare = [x for x in rew_rare if
                    ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_common = [x for x in nonrew_common if
                         ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        nonrew_rare = [x for x in nonrew_rare if
                       ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]

    Latency_second_step = session_latencies(session, 'second_step', FsplitF=False, )
    latency_rew_common = Latency_second_step[rew_common]
    latency_rew_rare = Latency_second_step[rew_rare]
    latency_nonrew_common = Latency_second_step[nonrew_common]
    latency_nonrew_rare = Latency_second_step[nonrew_rare]

    Latency_commonT = np.concatenate((latency_rew_common, latency_nonrew_common))#maybe
    Latency_rareT = np.concatenate((latency_rew_rare, latency_nonrew_rare))

    median_commonT = np.median(Latency_commonT)
    median_rareT = np.median(Latency_rareT)
    medianL_CT_RT = [median_commonT, median_rareT]

    return np.array(medianL_CT_RT), np.array(Latency_commonT), np.array(Latency_rareT)

def folder_transition_types_latency(folder,block_type, breakdown, forced_choice=False, ):

    if breakdown == 'folder':
        dir_folder_session = folder
        experiment = di.Experiment(dir_folder_session)
        experiment.save()
        sessions = experiment.get_sessions(subject_IDs='all', when='all')
        # defines subject ids and numbers acording to the file
        IDSpre = []
        for Session in sessions:
            sessionID = Session.subject_ID
            IDSpre.append(sessionID)
        IDSpre = np.array(list(set(IDSpre)))
        IDS = IDSpre.astype(int)
        subject_no = len(IDS)

        all_LC_medians = np.zeros(subject_no)
        all_LR_medians = np.zeros(subject_no)


        for i, subject_id in enumerate(IDS):  # sets up iteration per subject
            sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
            all_subject_LC =[]
            all_subject_LR = []

            for session in sessions:
                unused_meds, latency_commonT, latency_rareT = session_transition_types_latency(session,forced_choice=forced_choice, block_type=block_type,)
                all_subject_LC.extend(latency_commonT)
                all_subject_LR.extend(latency_rareT)


            LC_animal_median = np.median(all_subject_LC)
            LR_animal_median = np.median(all_subject_LR)

            all_LC_medians[i] = LC_animal_median
            all_LR_medians[i] = LR_animal_median


        return np.array(all_LC_medians), np.array(all_LR_medians)

    elif breakdown == 'session':
        dir_folder_session = folder
        experiment = di.Experiment(dir_folder_session)
        experiment.save()
        sessions = experiment.get_sessions(subject_IDs='all', when='all')
        # defines subject no and ids by the folder
        IDSpre = []
        for Session in sessions:
            sessionID = Session.subject_ID
            IDSpre.append(sessionID)
        IDSpre = np.array(list(set(IDSpre)))
        IDS = IDSpre.astype(int)
        subject_no = len(IDS)

        s1_all = []
        s2_all = []
        s3_all = []

        # do it by getting the sessions with the subject ids in subject id then going through the sessions calculating the medain latnecies and returning it to the relivant 's' array

        for i, subject_id in enumerate(IDS):  # sets up iteration per subject
            sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
            s1 = sessions[0]
            s2 = sessions[1]
            s3 = sessions[2]

            medaians_row_s1, a, b = session_transition_types_latency(s1, block_type=block_type, forced_choice=forced_choice)
            medaians_row_s2, a, b = session_transition_types_latency(s2, block_type=block_type, forced_choice=forced_choice)
            medaians_row_s3, a, b = session_transition_types_latency(s3, block_type=block_type, forced_choice=forced_choice)

        # lines to test if code is dividing data correctly:
            #medaians_row_s1 = (s1.subject_ID, s1.datetime_string, 1, 1)
            #medaians_row_s2 = (s2.subject_ID, s2.datetime_string, 1, 1)
            #medaians_row_s3 = (s3.subject_ID, s3.datetime_string, 1, 1)
            s1_all.append(medaians_row_s1)
            s2_all.append(medaians_row_s2)
            s3_all.append(medaians_row_s3)

        return np.array(s1_all), np.array(s2_all), np.array(s3_all)

#def get_TT_frequency(folder, breakdown, forced_choice = bool):


def plot_transition_type_latency(data_type, all_data_VEH, all_data_DOI, LC_VEH, LR_VEH, LC_DOI, LR_DOI, Title ={},):
    """data_type can be 'folder' or 'session' 
            if data_type is session all_data should = data, other values = 0 (irrelivant)
            if data_type is folder all_data =0 (is irrelivant) and enter corretly other values"""
    if data_type == 'folder':
        #convert inputs
        R_LC_VEH = LC_VEH[:, np.newaxis]
        R_LR_VEH = LR_VEH[:, np.newaxis]
        R_LC_DOI = LC_DOI[:, np.newaxis]
        R_LR_DOI = LR_DOI[:, np.newaxis]


        Median_all_Ttypes = np.concatenate((R_LC_VEH, R_LR_VEH, R_LC_DOI, R_LR_DOI), axis=1)
        MedianL_all_Ttypes_sem = stats.sem(Median_all_Ttypes, axis=0, nan_policy='omit') \
            if len(Median_all_Ttypes.shape) > 1 else 0
        MedianL_all_Ttypes_mean = np.nanmean(Median_all_Ttypes, axis=0) \
            if len(Median_all_Ttypes.shape) > 1 else Median_all_Ttypes
    elif data_type == 'session':
        Median_all_Ttypes = np.hstack((all_data_VEH,all_data_DOI))
        MedianL_all_Ttypes_sem = stats.sem(Median_all_Ttypes, axis=0, nan_policy='omit') \
            if len(Median_all_Ttypes.shape) > 1 else 0
        MedianL_all_Ttypes_mean = np.nanmean(Median_all_Ttypes, axis=0) \
            if len(Median_all_Ttypes.shape) > 1 else Median_all_Ttypes
    # generate the plot--------------------------------------------------------------------
    plt.figure(figsize=[3.6, 4.4]).clf()
    colors = ['orangered', 'coral', 'limegreen', 'lime']
    plt.bar(np.arange(1, 5), MedianL_all_Ttypes_mean, yerr=MedianL_all_Ttypes_sem,
            error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color=colors, alpha=0.8, fill=True, zorder=-1)

    y = Median_all_Ttypes
    x = np.random.normal(0, 0.12,size=len(Median_all_Ttypes))  # to distribute the dots randomly across the length of a bar
    for i in np.arange(1, 5):
        plt.scatter(x + i, y.T[i - 1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4,zorder=1)  # zorder=1 to bring to front
    plt.title(Title)
    plt.xlim(0.75, 5)
    plt.xticks([-0.25, 1, 2, 3, 4],
               ['\nTreatment.\ntype', '\nVEH\ncommon', '\nVEH\nrare', '\nVEH\ncommon', '\nVEH\nrare'], fontsize=2,fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0,25,50,75,100,125, 150, 175, 200,225,250,275,300,325,350,375,400], fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)

def session_trial_interaction_latencies(session,split, forced_choice=False):
    positions = session.select_trials(selection_type='all', select_n='all',
                                      block_type='non_neutral', )  # select only the trials specified by block_type and and selection_type
    choices = session.trial_data['choices']
    outcomes = session.trial_data['outcomes']
    transitions = session.trial_data['transitions']
    transition_type = session.blocks['trial_trans_state']
    free_choice_trials = session.trial_data['free_choice']
    rew_state = session.blocks['trial_rew_state']
    all_choice_latency = session_latencies(session=session, type='choice', FsplitF=False, )

    rew_ommision = np.where(outcomes[:-1] != 1)[0]
    rew = np.where(outcomes[:-1] == 1)[0]
    if forced_choice == False:  # eliminate forced choice trials
        rew_ommision = [x for x in rew_ommision if
                        ((x + 1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
        rew = [x for x in rew if
               ((x + 1 in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]

    rew_ommision_arr = np.array(rew_ommision) + 1
    rew_arr = np.array(rew) + 1

    if split == 'outcome':

        subsiquentL_after_ommision =all_choice_latency[rew_ommision_arr]
        subsiquentL_after_rew =all_choice_latency[rew_arr]
        subsiquentL_after_ommision_med = np.median(subsiquentL_after_ommision)
        subsiquentL_after_rew_med = np.median(subsiquentL_after_rew)
        subsiquentL_med_Ommis_rew = np.array([subsiquentL_after_ommision_med,subsiquentL_after_rew_med])

        return np.array(subsiquentL_med_Ommis_rew), np.array(subsiquentL_after_ommision), np.array(subsiquentL_after_rew)

    elif split == 'by_stay':

        # trying to get stay subsiquent latency depending on reward or not
        stay = tuple((choices[1:] == choices[:-1]).astype(int))  # 1: stay, 0: switch, position represents second trial in computation
        non_stay = tuple((choices[1:] != choices[:-1]).astype(int)) # 1: switch, 0: stay, position represents second trial in computation

        all_choice_latency_trim = all_choice_latency[1:len(stay)+1] #trims latencies so it can be indexed by stay
        stay_rew = stay
        stay_omm = stay
        nonstay_rew = non_stay
        nonstay_omm = non_stay

        mask_rewS = np.isin(np.arange(len(stay_rew)), rew_arr)
        stay_rew = np.where(~mask_rewS, 0, stay_rew)
        mask_ommS = np.isin(np.arange(len(stay_omm)), rew_ommision_arr)
        stay_omm = np.where(~mask_ommS, 0, stay_omm)
        mask_NSrew = np.isin(np.arange(len(nonstay_rew)), rew_arr)
        nonstay_rew = np.where(~mask_NSrew, 0, nonstay_rew)
        mask_NSomm = np.isin(np.arange(len(nonstay_omm)), rew_ommision_arr)
        nonstay_omm = np.where(~mask_NSomm, 0, nonstay_omm)

        stay_rewL = all_choice_latency_trim[np.array(stay_rew, dtype=bool)]
        stay_ommL = all_choice_latency_trim[np.array(stay_omm, dtype=bool)]
        nonstay_rewL = all_choice_latency_trim[np.array(nonstay_rew, dtype=bool)]
        nonstay_ommL = all_choice_latency_trim[np.array(nonstay_omm, dtype=bool)]

        med_stay_rewL = np.median(stay_rewL)
        med_stay_ommL = np.median(stay_ommL)
        med_nonstay_rewL =np.median(nonstay_rewL)
        med_nonstay_ommL =np.median(nonstay_ommL)
        medaian_SommL_SrewL_NSommL_NSrewL = [med_stay_ommL,med_stay_rewL,med_nonstay_ommL,med_nonstay_rewL]

        return np.array(medaian_SommL_SrewL_NSommL_NSrewL), np.array(stay_ommL), np.array(stay_rewL), np.array(nonstay_ommL), np.array(nonstay_rewL)

    elif split == 'transition_type':
        rew_common = np.where((transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
        rew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
        nonrew_common = np.where((transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
        nonrew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
        #
        if forced_choice == False:  # eliminate forced choice trials
            rew_common = [x for x in rew_common if
                          ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            rew_rare = [x for x in rew_rare if
                        ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            nonrew_common = [x for x in nonrew_common if
                             ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            nonrew_rare = [x for x in nonrew_rare if
                           ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]

        commonT = np.concatenate((rew_common,nonrew_common))
        rareT = np.concatenate((rew_rare, nonrew_rare))

        all_step1_latency = session_latencies(session, 'choice', FsplitF=False)

        commonT_nextT = commonT + 1
        commonT_nextL = all_step1_latency[commonT_nextT]
        commonT_nextLmed = np.median(commonT_nextL)

        rareT_nextT = rareT + 1
        rareT_nextL = all_step1_latency[rareT_nextT]
        rareT_nextLmed = np.median(rareT_nextL)

        medians = np.hstack((commonT_nextLmed, rareT_nextLmed))

        return np.array(medians), np.array(commonT_nextL), np.array(rareT_nextL)

    elif split == 'transXoutcome':
        rew_common = np.where((transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
        rew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & outcomes[:-1])[0]
        nonrew_common = np.where((transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
        nonrew_rare = np.where(~(transitions[:-1] == transition_type[:-1]) & ~outcomes[:-1])[0]
        #
        if forced_choice == False:  # eliminate forced choice trials
            rew_common = [x for x in rew_common if
                          ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            rew_rare = [x for x in rew_rare if
                        ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            nonrew_common = [x for x in nonrew_common if
                             ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]
            nonrew_rare = [x for x in nonrew_rare if
                           ((x in np.where(free_choice_trials == True)[0]) and (x in np.where(positions == True)[0]))]

        rew_commonNT = np.array(rew_common) + 1
        rew_rareNT = np.array(rew_rare) + 1
        nonrew_commonNT = np.array(nonrew_common) + 1
        nonrew_rareNT = np.array(nonrew_rare) + 1

        rew_commonNTL = all_choice_latency[rew_commonNT]
        rew_rareNTL = all_choice_latency[rew_rareNT]
        nonrew_commonNTL = all_choice_latency[nonrew_commonNT]
        nonrew_rareNTL = all_choice_latency[nonrew_rareNT]
        tXo_medians = [np.median(rew_commonNTL), np.median(rew_rareNTL), np.median(nonrew_commonNTL), np.median(nonrew_rareNTL)]

        return np.array(tXo_medians), np.array(rew_commonNTL), np.array(rew_rareNTL), np.array(nonrew_commonNTL), np.array(nonrew_rareNTL)

def folder_trial_interaction_latency(folder, breakdown, split):
    if split == 'outcome' or split == 'transition_type':
        if breakdown == 'folder':
            dir_folder_session = folder
            experiment = di.Experiment(dir_folder_session)
            experiment.save()
            sessions = experiment.get_sessions(subject_IDs='all', when='all')
            # defines subject ids and numbers acording to the file
            IDSpre = []
            for Session in sessions:
                sessionID = Session.subject_ID
                IDSpre.append(sessionID)
            IDSpre = np.array(list(set(IDSpre)))
            IDS = IDSpre.astype(int)
            subject_no = len(IDS)

            all_0medians = np.zeros(subject_no)
            all_1medians = np.zeros(subject_no)

            for i, subject_id in enumerate(IDS):  # sets up iteration per subject
                sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
                all_subject_Lomm = []
                all_subject_Lrew = []

                for session in sessions:
                    unused_meds, latency_afterO, latency_afterR = session_trial_interaction_latencies(session, split=split, forced_choice=False)
                    all_subject_Lomm.extend(latency_afterO)
                    all_subject_Lrew.extend(latency_afterR)

                Lomm_animal_median = np.median(all_subject_Lomm)
                Lrew_animal_median = np.median(all_subject_Lrew)

                all_0medians[i] = Lomm_animal_median #ommision or common t type
                all_1medians[i] = Lrew_animal_median # reward or rare t type

            return np.array(all_0medians), np.array(all_1medians)


        elif breakdown == 'session':
            dir_folder_session = folder
            experiment = di.Experiment(dir_folder_session)
            experiment.save()
            sessions = experiment.get_sessions(subject_IDs='all', when='all')
            # defines subject no and ids by the folder
            IDSpre = []
            for Session in sessions:
                sessionID = Session.subject_ID
                IDSpre.append(sessionID)
            IDSpre = np.array(list(set(IDSpre)))
            IDS = IDSpre.astype(int)
            subject_no = len(IDS)

            s1_all = []
            s2_all = []
            s3_all = []

            # do it by getting the sessions with the subject ids in subject id then going through the sessions calculating the medain latnecies and returning it to the relivant 's' array

            for i, subject_id in enumerate(IDS):  # sets up iteration per subject
                sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
                s1 = sessions[0]
                s2 = sessions[1]
                s3 = sessions[2]

                medaians_row_s1, a, b = session_trial_interaction_latencies(s1,split=split, forced_choice=False)
                medaians_row_s2, a, b = session_trial_interaction_latencies(s2,split=split, forced_choice=False)
                medaians_row_s3, a, b = session_trial_interaction_latencies(s3,split=split, forced_choice=False)
                # lines to test if code is dividing data correctly:
                # medaians_row_s1 = (s1.subject_ID, s1.datetime_string, 1, 1)
                # medaians_row_s2 = (s2.subject_ID, s2.datetime_string, 1, 1)
                # medaians_row_s3 = (s3.subject_ID, s3.datetime_string, 1, 1)
                s1_all.append(medaians_row_s1)
                s2_all.append(medaians_row_s2)
                s3_all.append(medaians_row_s3)

            return np.array(s1_all), np.array(s2_all), np.array(s3_all)

    elif split == 'by_stay' or 'transXoutcome':
        if breakdown == 'folder':
            dir_folder_session = folder
            experiment = di.Experiment(dir_folder_session)
            experiment.save()
            sessions = experiment.get_sessions(subject_IDs='all', when='all')
            IDSpre = []
            for Session in sessions:
                sessionID = Session.subject_ID
                IDSpre.append(sessionID)
            IDSpre = np.array(list(set(IDSpre)))
            IDS = IDSpre.astype(int)
            subject_no = len(IDS)
            all_SommL_medians = np.zeros(subject_no) # SommL or reward_common if split == trasXoutcome
            all_SrewL_medians = np.zeros(subject_no) # SrewL or reward_rare if split == trasXoutcome
            all_NSommL_medians = np.zeros(subject_no)# NSommL or nonrew_common if split == trasXoutcome
            all_NSrewL_medians = np.zeros(subject_no)# NSrewL or nonrew_rare if split == trasXoutcome

            for i, subject_id in enumerate(IDS):  # sets up iteration per subject
                sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
                all_subject_SommL = []
                all_subject_SrewL = []
                all_subject_NSommL = []
                all_subject_NSrewL = []
                for session in sessions:
                    unused_meds, SommL, SrewL, NSommL, NSrewL = session_trial_interaction_latencies(session, split=split, forced_choice=False)
                    all_subject_SommL.extend(SommL)
                    all_subject_SrewL.extend(SrewL)
                    all_subject_NSommL.extend(NSommL)
                    all_subject_NSrewL.extend(NSrewL)

                SommL_animal_median = np.median(all_subject_SommL)
                SrewL_animal_median = np.median(all_subject_SrewL)
                NSommL_animal_median = np.median(all_subject_NSommL)
                NSrewL_animal_median = np.median(all_subject_NSrewL)

                all_SommL_medians[i] = SommL_animal_median
                all_SrewL_medians[i] = SrewL_animal_median
                all_NSommL_medians[i] = NSommL_animal_median
                all_NSrewL_medians[i] = NSrewL_animal_median
            return np.array(all_SommL_medians), np.array(all_SrewL_medians), np.array(all_NSommL_medians), np.array(all_NSrewL_medians)

        elif breakdown == 'session':
            dir_folder_session = folder
            experiment = di.Experiment(dir_folder_session)
            experiment.save()
            sessions = experiment.get_sessions(subject_IDs='all', when='all')
            IDSpre = []
            for Session in sessions:
                sessionID = Session.subject_ID
                IDSpre.append(sessionID)
            IDSpre = np.array(list(set(IDSpre)))
            IDS = IDSpre.astype(int)

            s1_all = []
            s2_all = []
            s3_all = []

            # do it by getting the sessions with the subject ids in subject id then going through the sessions calculating the medain latnecies and returning it to the relivant 's' array

            for i, subject_id in enumerate(IDS):  # sets up iteration per subject
                sessions = experiment.get_sessions(subject_IDs=[subject_id], when='all')
                s1 = sessions[0]
                s2 = sessions[1]
                s3 = sessions[2]

                medaians_row_s1, a, b, c, d = session_trial_interaction_latencies(s1, split=split, forced_choice=False)
                medaians_row_s2, a, b, c, d = session_trial_interaction_latencies(s2, split=split, forced_choice=False)
                medaians_row_s3, a, b, c, d = session_trial_interaction_latencies(s3, split=split, forced_choice=False)

                s1_all.append(medaians_row_s1)
                s2_all.append(medaians_row_s2)
                s3_all.append(medaians_row_s3)

        return np.array(s1_all), np.array(s2_all), np.array(s3_all)


def plot_trial_interaction_latency(data_type, all_data_VEH, all_data_DOI, Lomm_VEH, Lrew_VEH, Lomm_DOI, Lrew_DOI, Title ={},):
    """data_type can be 'folder' or 'session'
            if data_type is session all_data should = data, other values = 0 (irrelivant)
            if data_type is folder all_data =0 (is irrelivant) and enter corretly other values"""
    if data_type == 'folder':
        #convert inputs
        R_Lomm_VEH = Lomm_VEH[:, np.newaxis]
        R_Lrew_VEH = Lrew_VEH[:, np.newaxis]
        R_Lomm_DOI = Lomm_DOI[:, np.newaxis]
        R_Lrew_DOI = Lrew_DOI[:, np.newaxis]


        Median_all_types = np.concatenate((R_Lomm_VEH, R_Lrew_VEH, R_Lomm_DOI, R_Lrew_DOI), axis=1)
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    elif data_type == 'session':
        Median_all_types = np.concatenate(all_data_VEH,all_data_DOI)
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    # generate the plot--------------------------------------------------------------------
    plt.figure(figsize=[3.6, 4.4]).clf()
    colors = ['orangered', 'coral', 'limegreen', 'lime']
    plt.bar(np.arange(1, 5), MedianL_all_types_mean, yerr=MedianL_all_types_sem,
            error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color=colors, alpha=0.8, fill=True, zorder=-1)

    y = Median_all_types
    x = np.random.normal(0, 0.12,size=len(Median_all_types))  # to distribute the dots randomly across the length of a bar
    for i in np.arange(1, 5):
        plt.scatter(x + i, y.T[i - 1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4,zorder=1)  # zorder=1 to bring to front
    plt.title(Title)
    plt.xlim(0.75, 5)
    plt.xticks([-0.25, 1, 2, 3, 4],
               ['\nTreatment.\np.Trew', '\nVEH\n-', '\nVEH\n+', '\nDOI\n-', '\nDOI\n+'], fontsize=2,fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0,100,200,300,400,500,600,700,800], fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)

def plot_trial_stayornonstay_interaction_latency(data_type, all_data_VEH, all_data_DOI, SommL_VEH, SrewL_VEH,NSommL_VEH, NSrewL_VEH, SommL_DOI, SrewL_DOI, NSommL_DOI, NSrewL_DOI, Title ={},):
    """data_type can be 'folder' or 'session'
            if data_type is session all_data should = data, other values = 0 (irrelivant)
            if data_type is folder all_data =0 (is irrelivant) and enter corretly other values"""
    if data_type == 'folder':
        #convert inputs
        R_SommL_VEH = SommL_VEH[:, np.newaxis]
        R_SrewL_VEH = SrewL_VEH[:, np.newaxis]
        R_NSommL_VEH = NSommL_VEH[:, np.newaxis]
        R_NSrewL_VEH = NSrewL_VEH[:, np.newaxis]

        R_SommL_DOI = SommL_DOI[:, np.newaxis]
        R_SrewL_DOI = SrewL_DOI[:, np.newaxis]
        R_NSommL_DOI = NSommL_DOI[:, np.newaxis]
        R_NSrewL_DOI = NSrewL_DOI[:, np.newaxis]

        Median_all_types = np.concatenate((R_SommL_VEH, R_SrewL_VEH, R_NSommL_VEH, R_NSrewL_VEH, R_SommL_DOI, R_SrewL_DOI, R_NSommL_DOI,R_NSrewL_DOI), axis=1)
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    elif data_type == 'session':
        Median_all_types = np.hstack((all_data_VEH,all_data_DOI))
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    # generate the plot--------------------------------------------------------------------
    plt.figure(figsize=[7.2, 6]).clf()
    colors = ['orangered', 'coral', 'chocolate', 'sandybrown', 'limegreen', 'lime','lightseagreen', 'turquoise']
    plt.bar(np.arange(1, 9), MedianL_all_types_mean, yerr=MedianL_all_types_sem,
            error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color=colors, alpha=0.8, fill=True, zorder=-1)

    y = Median_all_types
    x = np.random.normal(0, 0.12,size=len(Median_all_types))  # to distribute the dots randomly across the length of a bar
    for i in np.arange(1, 9):
        plt.scatter(x + i, y.T[i - 1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4,zorder=1)  # zorder=1 to bring to front
    plt.title(Title)
    plt.xlim(0.75, 9)
    plt.xticks([-0.25, 1, 2, 3, 4, 5, 6, 7, 8],
               ['\nTreatment\np.Trew\nAction', '\nVEH\n-\nstay', '\nVEH\n+\nstay', '\nVEH\n-\nchange', '\nVEH\n+\nchnage', '\nDOI\n-\nstay', '\nDOI\n+\nstay', '\nDOI\n-\nchange', '\nDOI\n+\nchnage'], fontsize=2,fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200], fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)

def plot_trial_transT_interaction_latency(data_type, all_data_VEH, all_data_DOI, CTnL_VEH, RTnL_VEH, CTnL_DOI, RTnL_DOI, Title ={},):
    """data_type can be 'folder' or 'session'
            if data_type is session all_data should = data, other values = 0 (irrelivant)
            if data_type is folder all_data =0 (is irrelivant) and enter corretly other values"""
    if data_type == 'folder':
        #convert inputs
        R_CTnL_VEH = CTnL_VEH[:, np.newaxis]
        R_RTnL_VEH = RTnL_VEH[:, np.newaxis]
        R_CTnL_DOI = CTnL_DOI[:, np.newaxis]
        R_RTnL_DOI = RTnL_DOI[:, np.newaxis]


        Median_all_types = np.concatenate((R_CTnL_VEH, R_RTnL_VEH, R_CTnL_DOI, R_RTnL_DOI), axis=1)
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    elif data_type == 'session':
        Median_all_types = np.hstack((all_data_VEH,all_data_DOI))
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    # generate the plot--------------------------------------------------------------------
    plt.figure(figsize=[3.6, 4.4]).clf()
    colors = ['orangered', 'coral', 'limegreen', 'lime']
    plt.bar(np.arange(1, 5), MedianL_all_types_mean, yerr=MedianL_all_types_sem,
            error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color=colors, alpha=0.8, fill=True, zorder=-1)

    y = Median_all_types
    x = np.random.normal(0, 0.12,size=len(Median_all_types))  # to distribute the dots randomly across the length of a bar
    for i in np.arange(1, 5):
        plt.scatter(x + i, y.T[i - 1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4,zorder=1)  # zorder=1 to bring to front
    plt.title(Title)
    plt.xlim(0.75, 5)
    plt.xticks([-0.25, 1, 2, 3, 4],
               ['\nTreatment.\np.Ttype', '\nVEH\ncommon', '\nVEH\nrare', '\nDOI\ncommon', '\nDOI\nrare'], fontsize=2,fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0,100,200,300,400,500,600,700,800], fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)

def plot_transXoutcome_trial_interaction_latency(data_type, all_data_VEH, all_data_DOI, rew_common_VEH, rew_rare_VEH,nonrew_common_VEH, nonrew_rare_VEH, rew_common_DOI, rew_rare_DOI, nonrew_common_DOI, nonrew_rare_DOI, Title ={},):
    """data_type can be 'folder' or 'session'
            if data_type is session all_data should = data, other values = 0 (irrelivant)
            if data_type is folder all_data =0 (is irrelivant) and enter corretly other values"""
    if data_type == 'folder':
        #convert inputs
        R_RC_VEH = rew_common_VEH[:, np.newaxis]
        R_RR_VEH = rew_rare_VEH[:, np.newaxis]
        R_NRC_VEH = nonrew_common_VEH[:, np.newaxis]
        R_NRR_VEH = nonrew_rare_VEH[:, np.newaxis]

        R_RC_DOI = rew_common_DOI[:, np.newaxis]
        R_RR_DOI = rew_rare_DOI[:, np.newaxis]
        R_NRC_DOI = nonrew_common_DOI[:, np.newaxis]
        R_NRR_DOI = nonrew_rare_DOI[:, np.newaxis]

        Median_all_types = np.concatenate((R_RC_VEH, R_RR_VEH, R_NRC_VEH, R_NRR_VEH, R_RC_DOI, R_RR_DOI, R_NRC_DOI,R_NRR_DOI), axis=1)
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    elif data_type == 'session':
        Median_all_types = np.hstack((all_data_VEH,all_data_DOI))
        MedianL_all_types_sem = stats.sem(Median_all_types, axis=0, nan_policy='omit') \
            if len(Median_all_types.shape) > 1 else 0
        MedianL_all_types_mean = np.nanmean(Median_all_types, axis=0) \
            if len(Median_all_types.shape) > 1 else Median_all_types
    # generate the plot--------------------------------------------------------------------
    plt.figure(figsize=[7.2, 6]).clf()
    colors = ['orangered', 'coral', 'chocolate', 'sandybrown', 'limegreen', 'lime','lightseagreen', 'turquoise']
    plt.bar(np.arange(1, 9), MedianL_all_types_mean, yerr=MedianL_all_types_sem,
            error_kw={'ecolor': 'k', 'capsize': 4, 'elinewidth': 1, 'markeredgewidth': 1},
            edgecolor='k', linewidth=1, color=colors, alpha=0.8, fill=True, zorder=-1)

    y = Median_all_types
    x = np.random.normal(0, 0.12,size=len(Median_all_types))  # to distribute the dots randomly across the length of a bar
    for i in np.arange(1, 9):
        plt.scatter(x + i, y.T[i - 1], color='gray', marker="o", s=10, alpha=0.4, edgecolors='k', lw=0.4,zorder=1)  # zorder=1 to bring to front
    plt.title(Title)
    plt.xlim(0.75, 9)
    plt.xticks([-0.25, 1, 2, 3, 4, 5, 6, 7, 8],
               ['\nTreatment\np.Trew\np.Ttrans', '\nVEH\n+\ncommon', '\nVEH\n+\nrare', '\nVEH\n-\ncommon', '\nVEH\n-\nrare', '\nDOI\n+\ncommon', '\nDOI\n+\nrare', '\nDOI\n-\ncommon', '\nDOI\n-\nrare'], fontsize=2,fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=9, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks([0,100,200,300,400,500,600,700,800], fontsize=8)
    plt.gca().spines["top"].set_color("None")
    plt.gca().spines["right"].set_color("None")
    plt.gcf().set_tight_layout(True)


def plot_line_s2L_transition_types(baselineC, baselineR, DrugC, DrugR, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,Title={}):
    #adjusting the inputs
    session_list = [(a1),(a2),(a3),(a4),(a5),(a6),(a7),(a8),(a9),(a10),(a11),(a12)]
    commonTL = np.hstack([arr[:, 0].reshape(-1, 1) for arr in session_list])
    rareTL = np.hstack([arr[:, 1].reshape(-1, 1) for arr in session_list])
    #calculating means and SEM----------------------------------
    #common transition latency
    SEM_per_session_CTL = stats.sem(commonTL, axis=0, nan_policy='omit') \
        if len(commonTL.shape) > 1 else 0
    Mean_per_session_CTL = np.nanmean(commonTL, axis=0) \
        if len(commonTL.shape) > 1 else commonTL
    # rare transition latency
    SEM_per_session_RTL = stats.sem(rareTL, axis=0, nan_policy='omit') \
        if len(rareTL.shape) > 1 else 0
    Mean_per_session_RTL = np.nanmean(rareTL, axis=0) \
        if len(rareTL.shape) > 1 else rareTL
    # for basline
    SEM_baseline_CTL = stats.sem(baselineC)
    Mean_baseline_CTL = np.mean(baselineC)
    SEM_baseline_RTL = stats.sem(baselineR)
    Mean_baseline_RTL = np.mean(baselineR)
    # for Drug
    SEM_Drug_CTL = stats.sem(DrugC)
    Mean_Drug_CTL = np.mean(DrugC)
    SEM_Drug_RTL = stats.sem(DrugR)
    Mean_Drug_RTL = np.mean(DrugR)
    #-------------------------------------------------------------------------------------------------
    # X-axis labels
    x_labels = ['Baseline', 'Drug'] + ['Post TR Day: 1'] + ['{}'.format(i) for i in range(2, 13)]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Set the plot limits and axis labels
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, 400)
    ax.set_xlabel('trial stage')
    ax.set_ylabel('Latency (ms)')

    # Define colors for each data set
    common_colour = 'tab:orange'
    rare_colour = 'limegreen'

    # Plot the baseline data points with error bars
    ax.errorbar([0], Mean_baseline_CTL, yerr=SEM_baseline_CTL, fmt='o', color=common_colour, label='common transition')
    ax.errorbar([0], Mean_baseline_RTL, yerr=SEM_baseline_RTL, fmt='o', color=rare_colour, label='rare transition')

    # Plot the drug data points with error bars
    ax.errorbar([1], Mean_Drug_CTL, yerr=SEM_Drug_CTL, fmt='o', color=common_colour)
    ax.errorbar([1], Mean_Drug_RTL, yerr=SEM_Drug_RTL, fmt='o', color=rare_colour)

    # Plot the lines for post-TR data with error bars
    ax.errorbar(range(2, 14), Mean_per_session_CTL, yerr=SEM_per_session_CTL, fmt='-o', color=common_colour)
    ax.errorbar(range(2, 14), Mean_per_session_RTL, yerr=SEM_per_session_RTL, fmt='-o', color=rare_colour)

    # Set the x-axis tick locations and labels
    x_ticks = np.arange(len(x_labels))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    plt.title(Title)
    # Add legend
    ax.legend(title='step 2 latency after:')

    plt.gcf().set_tight_layout(True)

def plot_line_compare_single_typeL_VEHvsDOI(baselineVEH,baselineDOI,DrugVEH,DrugDOI, TRVEH, TRDOI, title, ylim, label1, label2):

    SEM_per_session_VEH = stats.sem(TRVEH, axis=0, nan_policy='omit') \
        if len(TRVEH.shape) > 1 else 0
    Mean_per_session_VEH = np.nanmean(TRVEH, axis=0) \
        if len(TRVEH.shape) > 1 else TRVEH
    # rare transition latency
    SEM_per_session_DOI = stats.sem(TRDOI, axis=0, nan_policy='omit') \
        if len(TRDOI.shape) > 1 else 0
    Mean_per_session_DOI = np.nanmean(TRDOI, axis=0) \
        if len(TRDOI.shape) > 1 else TRDOI
    # for basline
    SEM_baseline_VEH = stats.sem(baselineVEH)
    Mean_baseline_VEH = np.mean(baselineVEH)
    SEM_baseline_DOI = stats.sem(baselineDOI)
    Mean_baseline_DOI = np.mean(baselineDOI)
    # for Drug
    SEM_Drug_VEH = stats.sem(DrugVEH)
    Mean_Drug_VEH = np.mean(DrugVEH)
    SEM_Drug_DOI = stats.sem(DrugDOI)
    Mean_Drug_DOI = np.mean(DrugDOI)
    # -------------------------------------------------------------------------------------------------
    # X-axis labels
    x_labels = ['Baseline', 'Drug'] + ['Post TR Day: 1'] + ['{}'.format(i) for i in range(2, 13)]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Set the plot limits and axis labels
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('trial stage')
    ax.set_ylabel('Latency (ms)')

    # Define colors for each data set
    VEH_colour = 'coral'
    DOI_colour = 'limegreen'

    # Plot the baseline data points with error bars
    ax.errorbar([0], Mean_baseline_VEH, yerr=SEM_baseline_VEH, fmt='o', color=VEH_colour, label=label1)
    ax.errorbar([0], Mean_baseline_DOI, yerr=SEM_baseline_DOI, fmt='o', color=DOI_colour, label=label2)

    # Plot the drug data points with error bars
    ax.errorbar([1], Mean_Drug_VEH, yerr=SEM_Drug_VEH, fmt='o', color=VEH_colour)
    ax.errorbar([1], Mean_Drug_DOI, yerr=SEM_Drug_DOI, fmt='o', color=DOI_colour)

    # Plot the lines for post-TR data with error bars
    ax.errorbar(range(2, 14), Mean_per_session_VEH, yerr=SEM_per_session_VEH, fmt='-o', color=VEH_colour)
    ax.errorbar(range(2, 14), Mean_per_session_DOI, yerr=SEM_per_session_DOI, fmt='-o', color=DOI_colour)

    # Set the x-axis tick locations and labels
    x_ticks = np.arange(len(x_labels))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    plt.title(title)
    # Add legend
    ax.legend(title='conditions:')

    plt.gcf().set_tight_layout(True)
r"""
def event_couter(event, breakdown):
    if event == 'step2_transition_type':

    elif event == 'next tri"""
def process_by_animal(sessions,process, medians=True):
    baseline = sessions[:6]
    Drug = sessions[6:12]
    TR = sessions[12:]

    if process == 'non_reward_CT_latency':
        baselinevar =[]
        Drugvar =[]

        for session in baseline:
            a,b,c,session_data,d = session_trial_interaction_latencies(session, 'transXoutcome', forced_choice=False)
            baselinevar.extend(session_data)
        for session in Drug:
            a, b, c, session_data, d = session_trial_interaction_latencies(session, 'transXoutcome', forced_choice=False)
            Drugvar.extend(session_data)

        medbaseline = np.median(baselinevar)
        medDrug = np.median(Drugvar)


        meds1, a, b, TR1, c = session_trial_interaction_latencies(TR[0], 'transXoutcome', forced_choice=False)
        med1 = np.median(TR1)

        meds2, a, b, TR2, c = session_trial_interaction_latencies(TR[1], 'transXoutcome', forced_choice=False)
        med2 = np.median(TR2)

        meds3, a, b, TR3, c = session_trial_interaction_latencies(TR[2], 'transXoutcome', forced_choice=False)
        med3 = np.median(TR3)

        meds4, a, b, TR4, c = session_trial_interaction_latencies(TR[3], 'transXoutcome', forced_choice=False)
        med4 = np.median(TR4)

        meds5, a, b, TR5, c = session_trial_interaction_latencies(TR[4], 'transXoutcome', forced_choice=False)
        med5 = np.median(TR5)

        meds6, a, b, TR6, c = session_trial_interaction_latencies(TR[5], 'transXoutcome', forced_choice=False)
        med6 = np.median(TR6)

        meds7, a, b, TR7, c = session_trial_interaction_latencies(TR[6], 'transXoutcome', forced_choice=False)
        med7 = np.median(TR7)

        meds8, a, b, TR8, c = session_trial_interaction_latencies(TR[7], 'transXoutcome', forced_choice=False)
        med8 = np.median(TR8)

        meds9, a, b, TR9, c = session_trial_interaction_latencies(TR[8], 'transXoutcome', forced_choice=False)
        med9 = np.median(TR9)

        meds10, a, b, TR10, c = session_trial_interaction_latencies(TR[9], 'transXoutcome', forced_choice=False)
        med10 = np.median(TR10)

        meds11, a, b, TR11, c = session_trial_interaction_latencies(TR[10], 'transXoutcome', forced_choice=False)
        med11 = np.median(TR11)

        meds12, a, b, TR12, c = session_trial_interaction_latencies(TR[11], 'transXoutcome', forced_choice=False)
        med12 = np.median(TR12)

        TRnon_reward_CT_latency_meddata = [(med1),(med2),(med3),(med4),(med5),(med6),(med7),(med8),(med9),(med10),(med11),(med12)]
        TRnon_reward_CT_latency_data = [(TR1), (TR2), (TR3), (TR4), (TR5), (TR6), (TR7), (TR8), (TR9), (TR10), (TR11), (TR12)]

        if medians ==True:
            return np.array(medbaseline), np.array(medDrug), np.array(TRnon_reward_CT_latency_meddata)
        elif medians ==False:
            return np.array(baselinevar), np.array(Drugvar), np.array(TRnon_reward_CT_latency_data)


    elif process == 'step2_transition_Latency':
        baselinevar = []
        Drugvar = []

        for session in baseline:
            a, b, session_data = session_transition_types_latency(session, 'non_neutral', forced_choice=False)
            baselinevar.extend(session_data)
        for session in Drug:
            a, b, c, session_data, d = session_transition_types_latency(session, 'non_neutral', forced_choice=False)
            Drugvar.extend(session_data)

        medbaseline = np.median(baselinevar)
        medDrug = np.median(Drugvar)

        meds1, a, TR1, = session_transition_types_latency(session, 'non_neutral', forced_choice=False)
        med1 = np.median(TR1)

        meds2, a, b, TR2, c = session_trial_interaction_latencies(TR[1], 'transXoutcome', forced_choice=False)
        med2 = np.median(TR2)

        meds3, a, b, TR3, c = session_trial_interaction_latencies(TR[2], 'transXoutcome', forced_choice=False)
        med3 = np.median(TR3)

        meds4, a, b, TR4, c = session_trial_interaction_latencies(TR[3], 'transXoutcome', forced_choice=False)
        med4 = np.median(TR4)

        meds5, a, b, TR5, c = session_trial_interaction_latencies(TR[4], 'transXoutcome', forced_choice=False)
        med5 = np.median(TR5)

        meds6, a, b, TR6, c = session_trial_interaction_latencies(TR[5], 'transXoutcome', forced_choice=False)
        med6 = np.median(TR6)

        meds7, a, b, TR7, c = session_trial_interaction_latencies(TR[6], 'transXoutcome', forced_choice=False)
        med7 = np.median(TR7)

        meds8, a, b, TR8, c = session_trial_interaction_latencies(TR[7], 'transXoutcome', forced_choice=False)
        med8 = np.median(TR8)

        meds9, a, b, TR9, c = session_trial_interaction_latencies(TR[8], 'transXoutcome', forced_choice=False)
        med9 = np.median(TR9)

        meds10, a, b, TR10, c = session_trial_interaction_latencies(TR[9], 'transXoutcome', forced_choice=False)
        med10 = np.median(TR10)

        meds11, a, b, TR11, c = session_trial_interaction_latencies(TR[10], 'transXoutcome', forced_choice=False)
        med11 = np.median(TR11)

        meds12, a, b, TR12, c = session_trial_interaction_latencies(TR[11], 'transXoutcome', forced_choice=False)
        med12 = np.median(TR12)

        TRnon_reward_CT_latency_meddata = [(med1), (med2), (med3), (med4), (med5), (med6), (med7), (med8), (med9),
                                           (med10), (med11), (med12)]
        TRnon_reward_CT_latency_data = [(TR1), (TR2), (TR3), (TR4), (TR5), (TR6), (TR7), (TR8), (TR9), (TR10), (TR11),
                                        (TR12)]

        if medians == True:
            return np.array(medbaseline), np.array(medDrug), np.array(TRnon_reward_CT_latency_meddata)
        elif medians == False:
            return np.array(baselinevar), np.array(Drugvar), np.array(TRnon_reward_CT_latency_data)


def remove_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    mask = (data >= lower_bound) & (data <= upper_bound)
    cleaned_data = data[mask]
    removed_indices = np.where(~mask)[0]
    return cleaned_data, removed_indices

def plot_individual_animal_latency(baseline, Drug, TR, ylim, title):


    x_labels = ['Baseline', 'Drug'] + ['Post TR Day: 1'] + ['{}'.format(i) for i in range(2, 13)]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Set the plot limits and axis labels
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('trial stage')
    ax.set_ylabel('Latency (ms)')

    # Define colors for each data set
    colour = 'firebrick'


    # Plot the baseline data points with error bars
    ax.errorbar([0], baseline,  fmt='o', color=colour)


    # Plot the drug data points with error bars
    ax.errorbar([1], Drug, fmt='o', color=colour)

    # Plot the lines for post-TR data with error bars
    ax.errorbar(range(2, 14), TR,  fmt='-o', color=colour)

    # Set the x-axis tick locations and labels
    x_ticks = np.arange(len(x_labels))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    plt.title(title)


    plt.gcf().set_tight_layout(True)