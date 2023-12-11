from model import Model

if __name__ == '__main__':
    model = Model(player1='MyAgent', player2='RandomPlayer')
    model.training(rounds=5000)
    # model.testing(rounds=1000, policy='policy_myAgent')
    # model.single_match()
