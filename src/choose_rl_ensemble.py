import argparse
from agent import RlAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_mode', type=str)
    parser.add_argument('--agent', type=str)
    parser.add_argument('--source_game', type=str)
    parser.add_argument('--episodes', type=int)

    args = parser.parse_args()
    policy_mode = args.policy_mode
    agent = args.agent
    source_game = args.source_game
    episodes = args.episodes

    RlAgent(source_game, policy_mode, agent, episodes)