# A new family of Deep Reinforcement Learning Algorithms: DQV, Dueling-DQV and DQV-Max Learning
  
  This repo contains the code that releases a new family of Deep Reinforcement Learning (DRL) algorithms.
  The aim of these algorithms is to learn an approximation of the state-value V(s) function alongside an approximation of
  the state-action value Q(s,a) function. Both approximations learn from each-others estimates, therefore 
  yielding faster and more robust training. This work is an in-depth extension of our original [DQV-Learning](https://arxiv.org/abs/1810.00368)
  paper.
  
  An in depth presentation of the several benefits that these algorithms provide are discussed in our **new paper**: 
  ['Approximating  two value functions instead of one: towards characterizing a new family of Deep Reinforcement 
  Learning Algorithms'](https://arxiv.org/submit/2825529/view). 
   
  The main algorithms presented in this repo are:
  
   * Dueling Deep Quality-Value (Dueling-DQV) Learning: **This Repo** 
   * Deep Quality-Value-Max (DQV-Max) Learning: **This Repo**
   * Deep Quality-Value (DQV) Learning: originally presented in ['DQV-Learning''](https://github.com/paintception/Deep-Quality-Value-DQV-Learning-),
    is now properly refactored.
   
   while we also release implementations of:
   
   * Deep Q-Learning: [DQN](https://arxiv.org/abs/1312.5602)
   * Double Deep Q-Learning: [DDQN](https://arxiv.org/abs/1509.06461) 
   
   which have been used for all the experimental comparisons presented in our work.
      
  ![alt text](https://github.com/paintception/Deep-Quality-Value-Family-/blob/master/figures/dqv_max_pong.jpg)![alt text](https://github.com/paintception/Deep-Quality-Value-Family-/blob/master/figures/dqv_max_enduro.jpg)
   
   If you aim to train an agent from scratch on a game of the Atari Arcade Learning benchmark (ALE) run the 
   `training_job.sh` script: it allows you to choose which type of agent to train according to the type of policy 
   learning it uses (online for DQV and Dueling-DQV, while offline for all other algorithms).
   
   In `./models` we release the trained models obtained on  the three main games of the ALE which 
   have been presented in our paper. We release weights for both DQV and DQV-Max. 
   
   You can use these models to explore the behavior of the learned value functions with the `./src/test_value_functions.py`
   script. The script will compute the averaged expected return of all visited states and show that the algorithms of the
   DQV-family suffer less from the overestimation bias of the Q function. The script will
   also show that our algorithms do not overestimate the V function instead of the Q function.
   
   ![alt text](https://github.com/paintception/Deep-Quality-Value-Family-/blob/master/figures/DQV-Max_estimates.png)
   
   We are currently benchmarking our algorithms on as many games of the Atari benchmark as possible: `./src/DQV_FULL_ATARI.sh`.