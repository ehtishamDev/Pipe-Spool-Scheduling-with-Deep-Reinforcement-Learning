import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import random
import operator
import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
import copy
import pandas as pd

import tensorflow as tf


# Define the spool class to store id, start date, processing time, due date and required resources information
# Define the __init__ method for class instantiation, then we passed different arguments to the instantiation operator
class Spool:
    def __init__(self, idx, id, start_date, processing_time, due_date, required_resources):
        self.idx = idx
        self.id = id
        self.start_date = start_date
        # self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        # self.start_date = datetime.strptime(start_date, '%m/%d/%Y')
        self.processing_time = processing_time
        # the finish date is calculated by a function that takes the start date as a parameter
        self.finish_date = self.calculate_finish_date(self.start_date)
        # print("finish date", self.finish_date)
        self.due_date = due_date  # datetime.strptime(due_date, '%m/%d/%Y')
        # the slack is calculated by a function
        self.slack = self.calculate_slack()
        # print("slack", self.slack)
        self.required_resources = required_resources
        self.start_date_agent = copy.deepcopy(self.start_date)
        self.finish_date_agent = copy.deepcopy(self.finish_date)

    # the function calculates the finish date
    def calculate_finish_date(self, start_date):
        processing_time = self.processing_time
        finish_date = start_date + timedelta(days=processing_time)
        finish_date = finish_date.ceil("D")
        # self.finish_date = self.finish_date.strftime('%Y-%m-%d')
        # self.finish_date = datetime.strptime(self.finish_date, '%Y-%m-%d')
        return finish_date

    def calculate_slack(self):
        due_date = self.due_date
        finish_date = self.finish_date
        # self.slack_datetime = self.due_date + timedelta(hours=self.processing_time)
        # Since due_date and finish_date do not have time information, the slack is purely the difference in days
        slack = due_date - finish_date
        return slack.days


# Define the dispatching rules
def FCFS(spools):
    schedule = []
    for spool in spools:
        schedule.append(spool.idx)
    return schedule


def EDD(spools):
    spools.sort(key=operator.attrgetter("due_date")) # this "key=operator.attrgetter("due_date")" is used as a key to the sort function
    schedule = []
    for spool in spools:
        schedule.append(spool.idx)
    return schedule


# 14-3-2023 Mohamed: Added SPT and STR and CR functions
def SPT(spools):
    spools.sort(key=operator.attrgetter("processing_time"))
    schedule = []
    for spool in spools:
        schedule.append(spool.idx)
    return schedule


# def STR(spools):
#     spools.sort(key=lambda x: x.finish_date - datetime.now())
#     schedule = []
#     for spool in spools:
#         schedule.append(spool.idx)
#     return schedule


def CR(spools):
    spools.sort(key=lambda x: (x.due_date - datetime.now()).total_seconds() / x.processing_time)
    schedule = []
    for spool in spools:
        schedule.append(spool.idx)
    return schedule


# Define the custom environment
class PipeSpoolFabricationEnv(gym.Env):
    def __init__(self, spool_ids, start_dates, processing_times, due_dates, required_resources):
        self.spool_ids = spool_ids
        self.start_dates = start_dates
        # print(start_dates)
        self.processing_times = processing_times
        self.due_dates = due_dates
        self.required_resources = required_resources
        self.num_spools = len(start_dates) # count the number of spools we have
        self.num_methods = 4 # the count of dispatching rules we have

        self.spools = [] # initialize the an empty list of spools to be filled later
        # The for loop appends the spools list with spool number, its start date, processign time, due date and required resources
        for i in range(self.num_spools):
            this_spool = Spool(i, spool_ids[i], start_dates[i], processing_times[i], due_dates[i],required_resources[i])
            self.spools.append(this_spool) # we append the spools list we initialized before
        self.spools = pd.Series(self.spools) # check why this is needed? it is the writing style, like lsit or array and so on

        # Action we can take
        # Either FCFS or EDD
        # self.action_space = spaces.Discrete(2)  # results in values either 0 or 1 which reflect the action of FCFS or EDD Song: uniformly sample from available spoons #There is random.choice and random.sample
        # self.action_space = np.random.choice(spaces.Discrete(spools).n) #Mohamed: random.choice sample without replacement, meaning that that spool that is selected won't be selected again

        # the dispatching rules are represented as a vector of 4 binary values
        # Mohamed 7 June 2023: I changed it to Discrete instead of multi-binary
        self.action_space = spaces.Discrete(self.num_methods)
        # Start with the number of spools as function parameter

        # check the observation space? this is no longer used
        # self.observation_space = spaces.Discrete(2)  # todo make observation space be binary as action space # Song: s1=remaining resource (continuous); s2=remaining spools (this is actually the search space of action, how can we update action_state everytime? Read paper); s3=read paper
        # self.reset()

        # Mohamed 7 June 2023: I changed it to Box instead of discrete
        # assuming that:
        # - delta_time can vary between -100 and 100
        # - mean_remain_process_time between -100 and 100
        # - mean_slack between -100 and 100
        # - max_slack between -100 and 100
        # - min_slack between -100 and 100
        # - resources can vary between -100 and 100
        # - length of remaining_spools_list can vary between -100 and 100

        low_values = np.array([-100, -100, -100, -100, -100, -100, -100])
        high_values = np.array([100, 100, 100, 100, 100, 100, 100])

        self.observation_space = spaces.Box(low=low_values, high=high_values, dtype=np.float32)

        # Mohamed 7 June 2023: I defined the initial state of the environment
        self.state_list = [0, np.mean(processing_times), np.mean(due_dates - start_dates), np.max(due_dates - start_dates),
                      np.min(due_dates - start_dates), np.mean(required_resources), self.num_spools]

    def step(self, action, first_selection):
        """1. Propagate state from time instant k to k+1
        2. Calculate the reward (delay) based on states and actions
        """
        # # Apply action
        # if action == 0:
        #     schedule = FCFS(self.spools)
        # elif action == 1:
        #     schedule = EDD(self.spools)
        # Song: calculate sk+1
        previous_state = copy.deepcopy(self.state_list) # independent copy of the state_list is created so any changes in it will not impact the previous state
        state = self.calculate_state(action, first_selection)
        # Song: calculate reward
        reward, done = self.calculate_reward(previous_state, state, action)
        # reward = 0
        # for i in range(self.state, self.num_spools): # Song: removed -1
        #     next_state = i + 1
        #     if self.resources >= self.spools[schedule[next_state]].required_resources: # Song: should we use current state not next state
        #         self.resources -= self.spools[schedule[next_state]].required_resources
        #         reward += abs(schedule[next_state] - schedule[i]) # Song: add due date (delay) as the reward instead of id. Also, if one spool is failed, we still need to finish it
        #         self.state = next_state
        #     else:
        #         reward -= 100  # If we do not have resources, how should we penalize it? Need to understand the magniture of your reward so this penalty should be strong enough.
        #         break
        if len(self.action_history) == self.num_spools:  # All spools are processed
            done = True
        return self.state_list, reward, done, {}

    def calculate_state(self, action, first_selection):

        # this gives a list of spools that have not yet been completed
        remaining_spools = []
        # iterate over each item in the self.remaining_spools_list
        for item in self.remaining_spools_list:
            # append the corresponding spool to the remaining_spools list
            remaining_spools.append(self.spools.iloc[item])


        if action == 0:
            schedule = FCFS(remaining_spools)
        elif action == 1:
            schedule = EDD(remaining_spools)
        elif action == 2:
            schedule = SPT(remaining_spools)
        elif action == 3:
            schedule = CR(remaining_spools)
        selected_spool = schedule[0] # this mean that the selected spool is assigned the first spool in the schedule list
        self.selected_spools = [selected_spool]


        # Mohamed: states --> s0 = current time and date s1: remaining available resoruces s2: remaining spools, s3: in-process spools, s4: completed spools, s5: remaining processing time
        if first_selection == True:
            self.current_date = self.last_date + timedelta(days=1)  # s0. Song: need to convert it to day
        for item in self.selected_spools:
            self.spools.iloc[item].start_date_agent = self.current_date
            # self.spools.iloc[selected_spools].finish_date_agent = self.spools[action].start_date_agent + self.spools[action].processing_time
            self.spools.iloc[item].finish_date_agent = self.spools.iloc[item].calculate_finish_date(self.spools.iloc[item].start_date_agent) # caluclate the finish date agent based on the start date agent
            self.resources -= self.spools.iloc[item].required_resources  # s1 # decreases the number of available resources in the shop
        self.action_history.extend(self.selected_spools) # add the selected spools to the action history
        self.remaining_spools_list = [i for i in self.spools_idx if i not in self.action_history]  # s2

        # Mohamed 7 June 2023: I commented the lines from 203 to 225
    # def split_spools(self, remaining_spools_list, current_date, available_resources):
    #     # Initialize lists for spools that can and cannot be started
    #     can_start = []
    #     cannot_start = []
    #
    #     # Loop through remaining spools
    #     for spool_idx in remaining_spools_list:
    #         spool = env.spools[spool_idx]
    #
    #         # Check if start date is earlier than current date and there are enough resources
    #         if spool.start_date_agent <= current_date and spool.required_resources <= available_resources:
    #             can_start.append(spool_idx)
    #             available_resources -= spool.required_resources
    #             print("split function", can_start, cannot_start)
    #         else:
    #             cannot_start.append(spool_idx)
    #             print("split function", can_start, cannot_start)
    #
    #
    #     print("split function", can_start, cannot_start)
    #     return can_start, cannot_start
    #
    #     print("split function", can_start, cannot_start)

        # [0, 1, 2, 3, 5, 6, 7, 8]
        # print("remaining_spools_list", self.remaining_spools_list)
        # examples?

        # check what does the next lines do? one hot is no longer used
        # self.remaining_spools_list_onehot = []
        # for spool in self.remaining_spools_list:
        #     tmp = np.zeros(self.num_spools)
        #     tmp[spool] = 1
        #     self.remaining_spools_list_onehot.append(tmp)
        # [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), ...]
        # print("remaining_spools_list_onehot", self.remaining_spools_list_onehot)

        def check_status(action_history):
            complete = [spool for spool in action_history if self.spools[spool].finish_date_agent < self.current_date]  # complete
            in_process = [spool for spool in action_history if spool not in complete]
            if first_selection == True:
                for spool in complete:
                    self.resources += self.spools.iloc[spool].required_resources # this line we are adding the resources back to the main pool
            return in_process, complete

        self.in_process, self.complete = check_status(self.action_history)  # s3 and s4
        self.mean_remain_process_time = self.calculate_average_remaining_processing_time(self.in_process)
        self.mean_slack, self.max_slack, self.min_slack = self.calculate_average_slack_of_remaining_spools(self.remaining_spools_list)

        # Finalize the state list
        delta_time = (self.current_date - self.last_date).days
        print("date", self.current_date, self.last_date)
        self.last_date = self.current_date
        self.state_list = [
            # self.current_date,
            delta_time,
            self.mean_remain_process_time,
            self.mean_slack,
            self.max_slack,
            self.min_slack,
            self.resources,
            len(self.remaining_spools_list),
        ]
        print("state", self.state_list)
        return self.state_list

    def calculate_reward(self, previous_state, state, action):
        # 2. Need to modify the method "calculate_reward". Now our actions are
        # which method (FCFS, EDD, SPT, or CR) we are going to follow to select the spool.
        # The following scenario could happen: the agent picked EDD,
        # then EDD picked the one with the earliest deadline, then what if we do
        # not have enough resource to build this spool? or what if we do not have
        # enough materials to build this spool yet? ------
        # The possible solution is that we do not penalize RL based on
        # "resource" and "start_date" anymore. When a method (EDD) is
        # picking a spool, the available spools are the ones with the
        # sufficient resources and available materials.

        self.reward = 0
        # The following shows the condition of item in check
        # item = 0: 1-1=0 (first time picked); 0-0=0 (picked already, was not picked again)
        # item = 1: 1-0=1 (has not been picked yet)
        # item = -1: 0-1=-1 (picked before, now picked again)


        # check = previous_state[: self.num_spools] - action # Check this, i don't understand the slicing part? this was used previously (when RL picks a spool not method) so it is no longer used

        # print("check", check)

        # penalize agent if it picks the one whose start_date is after the current time. Then terminate the episode
        for spool in self.selected_spools:
            if (self.spools.iloc[spool].start_date_agent < self.spools.iloc[spool].start_date):  # todo: only penalize the failed state not all states
                self.reward = -10000
                self.terminate = True
                break
            # elif -1 in check:
            #     self.reward = -10000
            #     self.terminate = True
            #     break
            else:
                if (self.spools.iloc[spool].finish_date_agent <= self.spools.iloc[spool].due_date):
                    self.reward = self.reward + abs((self.spools.iloc[spool].finish_date_agent - self.spools.iloc[spool].due_date).days)
                    self.terminate = False
                else:
                    self.reward = (self.reward + abs((self.spools.iloc[spool].finish_date_agent - self.spools.iloc[spool].due_date).days)* -100)
                    self.terminate = False
        return self.reward, self.terminate

    def calculate_average_slack_of_remaining_spools(self, remaining_spools_list):
        slack_of_remaining_spools = [
            (self.spools[spool].due_date - self.spools[spool].finish_date_agent).days
            for spool in remaining_spools_list
        ]
        total_slack = sum(slack_of_remaining_spools)
        average_slack_of_remaining_spools = np.ceil(
            total_slack / len(remaining_spools_list)
        )
        max_slack_of_remaining_spools = max(slack_of_remaining_spools)
        min_slack_of_remaining_spools = min(slack_of_remaining_spools)
        return (
            average_slack_of_remaining_spools,
            max_slack_of_remaining_spools,
            min_slack_of_remaining_spools,
        )

    def calculate_average_remaining_processing_time(self, in_process):
        remaining_processing_time = [
            (self.spools[spool].finish_date_agent - self.current_date).days
            for spool in in_process
        ]
        total_remaining_processing_time = sum(remaining_processing_time)
        average_remaining_processing_time = total_remaining_processing_time / max(
            len(in_process), 1
        )
        return average_remaining_processing_time



    def reset(self, seed=None, options=None):
        # main states
        self.last_date = self.start_dates.min()  # s0. Song: google and check syntax
        delta_time = 0 # we added this in order to avoid using time stamp that caused several errors
        # print(self.current_date)
        self.mean_remain_process_time = (self.calculate_average_remaining_processing_time([]))
        self.remaining_spools_list = np.arange(self.num_spools)  # This list is updated after RL make an action
        self.mean_slack, self.max_slack, self.min_slack = self.calculate_average_slack_of_remaining_spools(self.remaining_spools_list) # this function returns the 3 expressions
        self.remain_num_spools = self.num_spools
        self.resources = 100  # s1
        self.state_list = [
            delta_time,  # calculate delta of time, for example the length of hours or days, in our case it should be days
            self.mean_remain_process_time,
            self.mean_slack,
            self.max_slack,
            self.min_slack,
            self.resources,
            self.remain_num_spools,
        ] # self.remain_num_spools is a count

        print("state", self.state_list)
        # extra information for updating the main states
        self.spools_idx = np.arange(self.num_spools)  # This list never going to change.
        self.in_progress = []  # s3
        self.complete = []  # s4
        self.remaining_processing_time = []  # now we have mean_process_time, so this is no longer needed s5 check should this list be empty or it should have the time since it is all remaining
        self.action_history = []
        self.terminate = False
        return self.state_list




# if __name__ == "__main__":
#     """Main"""
#     df = pd.read_csv("Test.csv", parse_dates=["Start Date", "Due Date"])
#     spool_id = df["SpoolID"]
#     start_dates = df["Start Date"]
#     processing_times = df["Processing Time in Days"]
#     due_dates = df["Due Date"]
#     required_resources = df["Required Resources"]
#
#     env = PipeSpoolFabricationEnv(
#         spool_id, start_dates, processing_times, due_dates, required_resources
#     )
#
#     episodes = 10
#     for episode in range(1, episodes + 1):
#         print("Episode", episode)
#         state = env.reset()
#         done = False
#         score = 0
#         i = 1
#         while not done:
#             print("Step", i)
#             # env.render()
#             action = env.action_space.sample()
#             n_state, reward, done, info = env.step(action, True)
#             score += reward
#             i += 1
#         print("Episode:{} Score:{}".format(episode, score))
#     print("done")
