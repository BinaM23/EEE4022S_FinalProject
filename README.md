# EEE4022S_FinalProject
Repo for MATLAB code for Final year project
NB: methodology was adapted from [7], formerly referred to as [11] in the code comments
## File descriptions
- Biased_weights.m: implements part 2 of experiment 2 (varying service priority but users have specific preferences for criteria)
- FAHP_EAM.m: uses the extent analysis method to calculate Ws
- Gain_calculator_all_combinations.m: implements the last test of experiment 1 (tries all parameter combinations to 1dp)
- Gain_calculator_manyUsers.m: implements the scenarios in experiment 1 for 1000 users (weight parameter determination)
- Handover_simulation.m: implements the first 2 parts of experiment 3 (investigating threshold on handover)
- service_priority_exp_V2.m: implements experiment 2 (effect of varying service priority on selection)
- Single_user_algorithm.m: implements the worked example in the report/selects optimal RAT for a single user
- unnecessary_handover_simulation.m: implements the last part of experiment 3
## Things to note
- All programs used the same functions which I created.
- Some programs have the algorithm implementation encapsulated in a run_experiment function
- All details of how to run the programs are commented on in their respective files
- FAHP_EAM.m uses the extent analysis method to calculate Ws, this was changed from another method, therefore you must run it with a priority vector of your choice e.g. Pu=[3,3,3] to get the correct weights
