source activate Rl
KERAS_BACKEND=tensorflow

policy_mode="online"
agent="dqv"
source_game="PongDeterministic-v4"
episodes=1500

python choose_rl_ensemble.py --policy_mode $policy_mode --agent $agent --source_game $source_game --episodes $episodes
