# Lab 10 - Tic Tac Toe with RL

# Skeleton of the code
The skeleton of the code was taken from the template of Quixo given by the professors. Only `check_winner` function was changed according to the Tic Tac Toe rules.

## MyAgent player
MyAgent player is a class that represents a player that adopts an RL strategy.
The attributes of this player that were added from the standard Player class are:
1. **_states**: contains the list of states that occured during the game from the RL player prospective. In this way, when the match is over, starting from the last state that occured, we can backpropagate the rewards.
2. **_state_value**: it's a dictionary of (state, value) pairs, that gives to each state a certain score, which will be then utilized by the agent to make descions about the moves to play. The value of a specific state is continuously updated game by game. <u>This dictionary is what all the agents need to represent the policy it adpots</u>
3. **_lr**: The learning rate is a value that represents how strong the rewards should be. The higher the learning rate, the bigger/lower the rewards are respectively to winner/loser.
4. **_exp_rate**: the exploration rate represents the probability of the agent doing a random move rather than state-value driven one. This helps looking for new moves that were not known before, increasing exploration.
During test phase the exp_rate is set to 0, since we want to make decisions only with the current knowledge.
5. **_decay_gamma**: acts as a sort of discount, which reduce the effect of the rewards itslef, in a different way wrt learning rate. 

## Results
The results shows that the agent playing with a RL policy trained on 100k games will win 97.20% of the times and draw 2.80% of the 5k games played against a random player starting first. On the other hand, when the agent starts second, it loses only 0.16% of the times and draws 18.72% of the games.
```
MY AGENT IS PLAYER1
100%|██████████| 5000/5000 [00:10<00:00, 487.80it/s]
The results of the match [<class '__main__.MyAgent'> vs <class '__main__.RandomPlayer'>] are shown here:
The win rate for the player1 is 97.20% on a total of 5000 matches
The two players drew 2.80% of the games

MY AGENT IS PLAYER2
100%|██████████| 5000/5000 [00:09<00:00, 540.89it/s]
The results of the match [<class '__main__.RandomPlayer'> vs <class '__main__.MyAgent'>] are shown here:
The win rate for the player1 is 0.16% on a total of 5000 matches
The two players drew 18.72% of the games
```

I tried also increasing the number of training games (200k) and these are the corresponding results:
```
MY AGENT IS PLAYER1
100%|██████████| 5000/5000 [00:09<00:00, 516.69it/s]
The results of the match [<class '__main__.MyAgent'> vs <class '__main__.RandomPlayer'>] are shown here:
The win rate for the player1 is 97.86% on a total of 5000 matches
Thw two players drew 2.14% of the games

MY AGENT IS PLAYER2
100%|██████████| 5000/5000 [00:08<00:00, 557.44it/s]
The results of the match [<class '__main__.RandomPlayer'> vs <class '__main__.MyAgent'>] are shown here:
The win rate for the player1 is 0.00% on a total of 5000 matches
Thw two players drew 20.28% of the games
```