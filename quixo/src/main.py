from model import Model
from player import Player, RandomPlayer, HumanPlayer, RLPlayer, MinMaxPlayer

if __name__ == '__main__':
    # Initialize players
    human = HumanPlayer(name='figimodi')
    min_max_p = MinMaxPlayer(name='min_max_player')
    rl_p = RLPlayer(name='rl_player')

    # Initialize the model
    model = Model()
    model.set_players(player1=rl_p, player2=min_max_p)
    # model.set_players(player1=rl_p)
    # model.set_rewards(reward_winning=0., reward_losing=1.)

    # Train the model
    # model.training()

    # Setting the policies to the agent player. The policy will be used to play in the test phase
    rl_p.load_policy('../policies/policy_50kgames_p1')
    
    # Start testing 
    # model.testing(rounds=50)

    # Play against bots human
    # model.set_players(player1=human, player2=min_max_p)
    model.single_match()