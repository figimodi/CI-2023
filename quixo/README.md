## Setup Libraries
```
pip install -r requirements.txt
```

## Literature
The following references helped me out building the RL agent and the MinMax agent.
RL was adapted from Tic-Tac-Toe to Quixo.
MinMax agent was translated from C++ to python, with some bugs correction made in the process (i.e. check_sequence is bugged in original code).
RL Q-learning gives another point of view to Tic-Tac-Toe with RL. I didn't implement Q-learning beacuse my friend, colleague and collaborator for the project [Stiven Hidri](https://github.com/stiven-hidri/) tried this approach with scarce results (similar to the one i get with standard RL).
1. [RL Tic-Tac-Toe](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)
2. [Quixo Bot C++](https://github.com/DobrinTs/Quixo-Bot)
3. [RL Q-learning Tic-Tac-Toe](https://towardsdatascience.com/an-ai-agent-learns-to-play-tic-tac-toe-part-3-training-a-q-learning-rl-agent-2871cef2faf0)

## Results

### **RL agents**
RL agents were trained on at most 100k games. The two agents (one starting first, and the other starting second) trained themselves but some matches with RandomPlayer were as well added to the train phase of the agents to give an higher diversity of strength of the opponents.
I noticed that the amount of games used for training doesn't variate at all the performances of the agents. This could maybe be caused by the huge dimension of the tree of possibilities (20 moves per each turn on average). That could be also the reason why the performance of the agents will sometimes goes under 50% (worst wrt a Random), since they trust moves that were maybe only encountered very few times and don't deserve the outcome of the overall game.

Here are some variables and parameters that characterizes the agent:
1. **_states**: contains the list of states that occured during the game from the RL player prospective. In this way, when the match is over, starting from the last state that occured, we can backpropagate the rewards.
2. **_state_value**: it's a dictionary of (state, value) pairs, that gives to each state a certain score, which will be then utilized by the agent to make descions about the moves to play. The value of a specific state is continuously updated game by game. <u>This dictionary is what all the agents need to represent the policy it adpots</u>
3. **_lr**: The learning rate is a value that represents how strong the rewards should be. The higher the learning rate, the bigger/lower the rewards are respectively to winner/loser.
4. **_exp_rate**: the exploration rate represents the probability of the agent doing a random move rather than state-value driven one. This helps looking for new moves that were not known before, increasing exploration.
During test phase the exp_rate is set to 0, since we want to make decisions only with the current knowledge.
5. **_decay_gamma**: acts as a sort of discount, which reduce the effect of the rewards itslef, in a different way wrt learning rate. 

Some of the results are shown here:
```
FIRST TRAINING:
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [13:30<00:00, 12.34it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [12:16<00:00, 13.57it/s]

FIRST TESTING:
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:23<00:00, 12.03it/s]
The results of the match <class '__main__.RLPlayer'> vs <class '__main__.RandomPlayer'> are shown here:
The win rate for the player1 is 30.30% on a total of 1000 matches
Player1 was trained on 10k games against a Random
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:26<00:00, 11.60it/s]
The results of the match <class '__main__.RandomPlayer'> vs <class '__main__.RLPlayer'> are shown here:
The win rate for the player1 is 73.80% on a total of 1000 matches
Player2 was trained on 10k games against a Random

SECOND TRAINING:
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [45:23<00:00,  3.67it/s] 

SECOND TESTING:
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:15<00:00, 13.26it/s]
The results of the match <class '__main__.RLPlayer'> vs <class '__main__.RLPlayer'> are shown here:
The win rate for the player1 is 100.00% on a total of 1000 matches
Player1 and Player2 were trained (on top of the first training) on 10k games against a each other
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:31<00:00, 10.99it/s]
The results of the match <class '__main__.RLPlayer'> vs <class '__main__.RandomPlayer'> are shown here:
The win rate for the player1 is 24.40% on a total of 1000 matches
Player1 was trained (on top of the first training) on 10k games against another agent
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:29<00:00, 11.22it/s]
The results of the match <class '__main__.RandomPlayer'> vs <class '__main__.RLPlayer'> are shown here:
The win rate for the player1 is 78.70% on a total of 1000 matches
Player2 was trained (on top of the first training) on 10k games against another agent
```

As we can see, adding some additional training seems to worsen the performances, even if this is an unexpected behavior
The 100% win rate of RL agent 1 vs RL agent 2 is the result of the same game over and over, since RL agents will fall into the same path all the times (the path that simulates the sequence of action that gives the best chance to bot of them). This result could be 100% 0r 0% depending on the moves that were discovered by each of them.

Policies for player 1 and 2 are provided in the `/policies` folder.

### **MinMax agents**

My Min Max agent was built taking inspiration from a github repo that showed a C++ bot playing with min max (I corrected the relative bugs of that version and changed the functions due to the mismatches between C++ and python).

MinMax agents give optimal performances with depth 3 (5 is a little bit slow): both 100% win rate staring first and starting second, both against random and RL agents.
I also tried to challenge one of my friend Stiven Hidri (who built his own heurisitc of a Min Max agent) and it won every gamee we analyzed. I hope this helped Stiven to better build his agent.

The heuristic used for the MinMax agent is the following:
1. It accounts on how many tiles the bot owns on the border (corner have a little bit lower value than edges), while penalizing tiles that the opponent owns.
2. It accounts for threes-in-a-row and fours-in-a-row owned by the bot (and penalize as well the one that the opponent owns).

Some considerations:
I would have liked to try to test these techniques which would have likely improved RL/MinMax, but i had short time getting these new versions:
- Mix MinMax to RL to improve speed of the agent, but maybe better if: build a table that would remember the very first moves of the game so that every new game the agent doesn't need to start from scratch the computation of very simlpe move. Maybe apply a threshold of moves `tm` after which the agent starts computing in depth. 
- Prevent the MinMax agent to go into an infinte loop, how? increasing depth when repetition at first. Secondly inform also on console.
- Implement specular positions (and relatives concepts). Although i belive this would lead to a smaller size of our policy file and a faster convergence to optimal play by the RL agent, i think that practically speaking, with the resources i had to train (hardware and time), this wouldn't have changed the results at all. I'm pretty sure that the number of games of training should be in the order of millions.