from model import Model
from player import Player, RandomPlayer, HumanPlayer, RLPlayer, MinMaxPlayer

if __name__ == '__main__':
    # Initialize players
    human = HumanPlayer(name='figimodi')
    min_max_p = MinMaxPlayer(name='min_max_player')
    rl_p = RLPlayer(name='rl_player')

    # Initialize the model
    model = Model()
    model.set_players(player1=min_max_p)
    # rl_model = RLModel()
    # rl_model.set_players(player1=rl_p)
    # rl_model.set_rewards(reward_winning=0., reward_losing=1.)

    # Train the model
    # rl_model.training(rounds=2000)

    # Setting depth to min max player
    min_max_p.set_max_depth(3)

    # Setting the policies to the agent player. The policy will be used to play in the test phase
    rl_p.load_policy('../policies/policy_rl_player')
    
    # Start testing 
    model.testing(rounds=10)
    # rl_model.testing(rounds=500)

    # Play against bots human
    # model.set_players(player1=human, player2=min_max_p)
    # model.single_match()