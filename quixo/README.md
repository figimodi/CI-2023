## Setup Libraries
```py
pip install -r requirements.txt
```

## Literature
1. [RL Tic-Tac-Toe](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)
2. [Quixo Bot C++](https://github.com/DobrinTs/Quixo-Bot)
3. [RL Q-learning Tic-Tac-Toe](https://towardsdatascience.com/an-ai-agent-learns-to-play-tic-tac-toe-part-3-training-a-q-learning-rl-agent-2871cef2faf0)

## Results
Below are shown the results for v1.0 of the project.

The tests have been done on a Reinforcement Learning Agent and a MinMax Agent.

RL Agents were trained on 500 games that they played against each other.

### <u>RL Agent vs Random</u>
```
The results of the match <class 'player.RLPlayer'> vs <class 'player.RandomPlayer'> are shown here:
The win rate for the player1 is 65.00% on a total of 100 matches
```

### <u>Random vs RL Agent</u>
```
The results of the match <class 'player.RLPlayer'> vs <class 'player.RandomPlayer'> are shown here:
The win rate for the player1 is 65.00% on a total of 100 matches
```

### <u>RL Agent vs RL Agent</u>
```
The results of the match <class 'player.RandomPlayer'> vs <class 'player.RLPlayer'> are shown here:
The win rate for the player1 is 35.00% on a total of 100 matches
```

### <u>MinMax vs Random</u>
```
The results of the match <class 'player.MinMaxPlayer'> vs <class 'player.RandomPlayer'> are shown here:
The win rate for the player1 is 100.00% on a total of 100 matches
```
