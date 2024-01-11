from model import Model, RLModel
from player import Player, RandomPlayer, HumanPlayer, RLPlayer, MinMaxPlayer
import numpy as np

if __name__ == '__main__':
    # Initialize players
    human = HumanPlayer(name='figimodi')
    min_max_p = MinMaxPlayer(name='min_max_player', max_depth=3)
    # rl_p1 = RLPlayer(name='p1_20k', lr=0.7)
    # rl_p2 = RLPlayer(name='rl_player2_lr05', lr=0.8)

    # Initialize the model
    model = Model()
    # model.set_players(player1=min_max_p)
    # rl_model = RLModel()
    # rl_model.set_players(player1=rl_p1)
    # rl_model.set_rewards(reward_winning=5., reward_losing=0.)

    # rl_p1.load_policy('../policies/policy_rl_player1')
    # rl_p2.load_policy('../policies/policy_rl_player2')

    # Train the model
    # epochs = 40
    # wrs = np.zeros(40)

    # for epoch in range(epochs):
    #     rl_p1.name = f'p1_epoch={epoch}_20k'
    #     rl_model.training(rounds=500)
    #     rl_p1.load_policy(f'../policies/policy_p1_epoch={epoch}_20k')
    #     wrs[epoch] = rl_model.testing(500)

    # np.save('statistics', wrs)

    # Setting depth to min max player
    # min_max_p.set_max_depth(3)

    # Setting the policies to the agent player. The policy will be used to play in the test phase
    # rl_p1.load_policy('../policies/policy_p1_500_07_5_0')
    # rl_p2.load_policy('../policies/policy_rl_player2_lr05')
    
    # Start testing 
    # model.testing(rounds=100)
    # rl_model.testing(rounds=500)

    # Play against bots human
    model.set_players(player1=min_max_p)
    model.single_match()
