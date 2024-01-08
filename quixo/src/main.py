from model import Model, RLModel
from player import Player, RandomPlayer, HumanPlayer, RLPlayer, MinMaxPlayer

if __name__ == '__main__':
    # Initialize players
    human = HumanPlayer(name='figimodi')
    min_max_p = MinMaxPlayer(name='min_max_player')
    rl_p1 = RLPlayer(name='rl_player1_advance')
    rl_p2 = RLPlayer(name='rl_player2_advance')

    # Initialize the model
    # model = Model()
    # model.set_players(player1=min_max_p)
    rl_model = RLModel()
    rl_model.set_players(player1=rl_p1, player2=rl_p2)
    rl_model.set_rewards(reward_winning=1., reward_losing=0.)

    # Setting the policies to the agent player. The policy will be used to play in the test phase
    rl_p1.load_policy('../policies/policy_rl_player')
    rl_p2.load_policy('../policies/policy_rl_player2')

    # Train the model
    rl_model.training(rounds=500)

    # Setting depth to min max player
    # min_max_p.set_max_depth(3)

    # Setting the policies to the agent player. The policy will be used to play in the test phase
    rl_p1.load_policy('../policies/policy_rl_player1_advance')
    rl_p2.load_policy('../policies/policy_rl_player2_advance')
    
    # Start testing 
    # model.testing(rounds=10)
    rl_model.testing(rounds=250)

    # Play against bots human
    # model.set_players(player1=human, player2=min_max_p)
    # model.single_match()