from model import Model

if __name__ == '__main__':
    # Initialize the model: selecting players, names, policies and eventually rewards
    model = Model(player1='MinMaxPlayer', player2='RandomPlayer', name1='min_max_p1')
    
    # Train the model
    # model.training()

    # Setting the policies to the agent player. The policy will be used to play in the test phase
    # model.set_policies(policy1='../policy_min_max_p1')
    
    # Start testing 
    model.testing(rounds=100)
