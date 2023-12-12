from model import Model

if __name__ == '__main__':
    model = Model(player1='MyAgent', player2='RandomPlayer', name1='50kgames_p1', name2='50kgames_p2')
    model.training()
    model.set_policies(policy1='policy_50kgames_p1')
    model.testing()
    # model.single_match()
