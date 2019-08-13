source activate Rl
KERAS_BACKEND=tensorflow

policy_mode="offline"
agent="ddqn"
source_game="PongDeterministic-v4"
episodes=1

python choose_rl_ensemble.py --policy_mode $policy_mode --agent $agent --source_game $source_game --episodes $episodes
