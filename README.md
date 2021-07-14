# A new family of Deep Reinforcement Learning algorithms: DQV, Dueling-DQV and DQV-Max Learning
  
  This repo contains the code that releases a new family of Deep Reinforcement Learning (DRL) algorithms.
  The aim of these algorithms is to learn an approximation of the state-value V(s) function alongside an approximation of
  the state-action value Q(s,a) function. Both approximations learn from each-others estimates, therefore 
  yielding faster and more robust training. This work is an in-depth extension of our original [DQV-Learning](https://arxiv.org/abs/1810.00368)
  paper and will be presented in December at the coming **NeurIPS Deep Reinforcement Learning (DRLW) Workshop** in Vancouver (Canada).
  
  An in depth presentation of the several benefits that these algorithms provide are discussed in our new paper: 
  **'Approximating  two value functions instead of one: towards characterizing a new family of Deep Reinforcement 
  Learning Algorithms'**.
  
  Be sure to check out Arxiv for a [pre-print](https://arxiv.org/abs/1909.01779) of our work!
  
  The main algorithms presented in this repo are:
  
   * Dueling Deep Quality-Value (Dueling-DQV) Learning: **This Repo** 
   * Deep Quality-Value-Max (DQV-Max) Learning: **This Repo**
   * Deep Quality-Value (DQV) Learning: originally presented in ['DQV-Learning''](https://github.com/paintception/Deep-Quality-Value-DQV-Learning-),
    is now properly refactored.
   
   while we also release implementations of:
   
   * Deep Q-Learning: [DQN](https://arxiv.org/abs/1312.5602)
   * Double Deep Q-Learning: [DDQN](https://arxiv.org/abs/1509.06461) 
   
   which have been used for some of the comparisons presented in our work.
      
  ![alt text](https://github.com/paintception/Deep-Quality-Value-Family-/blob/master/figures/dqv_max_pong.jpg)![alt text](https://github.com/paintception/Deep-Quality-Value-Family-/blob/master/figures/dqv_max_enduro.jpg)
   
   If you aim to train an agent from scratch on a game of the Atari Arcade Learning benchmark (ALE) run the 
   `training_job.sh` script: it allows you to choose which type of agent to train according to the type of policy 
   learning it uses (online for DQV and Dueling-DQV, while offline for all other algorithms).
   Note that based on which game you choose, some modifications to the code might be required.
   
   In `./models` we release a trained model obtained on Pong both for DQV and for DQV-Max. 
   
   You can use these models to explore the behavior of the learned value functions with the `./src/test_value_functions.py`
   script. The script will compute the averaged expected return of all visited states and show that the algorithms of the
   DQV-family suffer less from the overestimation bias of the Q function. The script will
   also show that our algorithms do not overestimate the V function instead of the Q function.
  
  
