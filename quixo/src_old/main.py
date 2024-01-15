# from model import Model, RLModel
# from player import Player, RandomPlayer, HumanPlayer, RLPlayer, MinMaxPlayer
from game import Game, MyGame, RandomPlayer

if __name__ == '__main__':
    random1 = RandomPlayer("p1")
    random2 = RandomPlayer("p2")

    game = Game()

    game.play()