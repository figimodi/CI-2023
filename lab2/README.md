## Lab 2 - nim/misère game
I created two optimal players:
1. one for the game of nim (the one who picks the last element(s) win)
2. one for the variation of the game of nim called "misère" (the one who picks the last element(s) lose)

### Nim "normal" game
The optimal strategy for the game is to leave the opponent in a state with `sum=0`:
what we mean is that, if the maximum number of objects that each player can take is k, then `sum=[mex(r_1) xor mex(r_2) xor ... xor mex(r_n)] = 0`,
where `mex(r_i)` is equal to `mex(r_i)=n_i mod (k + 1)` and it's called the Minimum Excluded Value, and it comes from the Sprague-Grundy Theorem.
`n_i` is the number of objects in the i_th row. Whenever a player is in a state sum=0 the we call the position a P position, which means the Previous player is winning, thus the one who has to play lose. 
N position on the other hand are the one in which the Next opponent is winning, i.e. the one who's about to make a move.
From a P position you can only go in N positions, and from N position there's always a move that is a P position.

*This modality is solved, for any number of rows, for any value of k => the game is already decided depending on the initial state (if both players play perfectly).*

### Nim "misère" variation
The optimal strategy for this variation is to apply the same rules as the normal game *unless*:
We can make a move that goes into a state composed of rows with only one element remained, and the amount of this rows is odd.
For example [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 1] ...

*Let's call these previous positions `final stages`, then we can say:*
*This variation is not solved for k<inf, or better for k<(2\*#rows)-1, since it is possible that we cannot force a final stage, and moreover a final stage will be imposed by the opponent, so the result of the game depends on k*

### Results
For both variation results are shown with graphs, in the `/results` folders

### EA strategy
The EA strategy is applied on the "noraml" Nim game.
The indiviudal is composed of `2*N_STRATEGIES`, with the first half representing the tendencies to play each strategy, and the second half represents the 'learning rates' of each strategy (sigmas), that plays a role in the mutation performed with a gaussian. Crossovers are also taken into consideration.
The fitness is calculated as the percentages of wins the individual gets by playing against all the strategies, alternating starting player, in an even way.
With some parameters tuning (population size, tournament_size, mutation probability, #offsprings) we get the following result:

*The fittest individual will be the one that tends to play the most of the moves with the rule of sum=0, thus confirming the previous considerations.*