# Pipe-Spool-Scheduling-with-Deep-Reinforcement-Learning
This project aims to optimize the scheduling of pipe spool fabrication tasks using deep reinforcement learning techniques. A custom simulation environment is created to model the pipe spool scheduling problem. A DQN agent is then trained using this environment to learn effective scheduling policies.

## Problem Description
The problem involves scheduling a set of pipe spool fabrication jobs with given start dates, processing times, due dates and resource requirements. The objective is to find a scheduling/dispatching rule that minimizes delays in completing jobs on time.

## Key challenges include:

Limited fabrication resources that need to be allocated efficiently

Uncertain processing times that can affect due dates

Interdependent jobs that need to be scheduled sequentially

Conventional scheduling rules like FCFS, EDD etc. may not optimize for delays in all cases. There is a need for a learning-based approach.

## Environment
The PipeSpoolFabricationEnv class models the scheduling problem as a Markov Decision Process. It contains:

Data on jobs (id, start date, processing time, due date, resources)

State representation capturing time elapsed, mean slack, resources etc.

Actions to select next job based on FCFS, EDD, SPT or CR rules

Step function to simulate state transition and calculate rewards

Reset function to restart the episode

## Model
A DQN agent is used to learn optimal scheduling policies. It consists of:

Q-Networks to estimate Q-values for each (state, action) pair

Experience replay with prioritization to train on transitions

epsilon-greedy policy for exploration

Target network for stability

## Training
The agent is trained using the environment for many episodes. Each episode consists of sequential steps of observing state, selecting action, transitioning to next state and receiving rewards. The Q-Networks are updated using experience replay to minimize temporal difference errors.

## Results
The trained agent achieves significantly lower average delays compared to baseline rules, demonstrating that it has learned effective scheduling strategies through trial-and-error interactions with the environment.
