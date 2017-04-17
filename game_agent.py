"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import logging
import datetime
import math
from random import choice

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    free_spaces = game.height * game.width - game.move_count

    def heuristic_zero():
        return own_moves - opp_moves

    def heuristic_one():
        return own_moves - 2 * opp_moves

    def heuristic_two():
        return float(((own_moves + 1)) / ((opp_moves + 1)))

    def heuristic_three():
        return float((2 * (own_moves + 1)) / ((opp_moves + 1)))

    def heuristic_four():
        return float((own_moves - (2*opp_moves))/free_spaces)

    def heuristic_five():
        """ Most Moves in Quadrant """
        quadrant = quadrant_with_most_open_space(game)

        corners = [(game.width - 1, 0), (0, 0), (0, game.height - 1), (game.width - 1, game.height - 1)]

        pos = game.get_player_location(player)

        sq_dist_from_corner = (corners[quadrant[0]][0] - game.get_player_location(player)[0]) ** 2 + \
            (corners[quadrant[0]][1] - game.get_player_location(player)[1]) ** 2

        return sq_dist_from_corner

    def heuristic_six():
        """ Longest Path Heuristic """
        # do the following on a copy because we don't want to change
        # the current game state
        own_longest_path = get_longest_path_length(game.copy(), player)
        opp_longest_path = get_longest_path_length(game.copy(), game.get_opponent(player))
        path_difference = own_longest_path - opp_longest_path
        #logging.debug('path_difference: ' + str(path_difference))
        return path_difference

    # if free_spaces > 17:
    #     return heuristic_three()
    # else:
    #     return heuristic_six()

    # if free_spaces > 30:
    #     return heuristic_zero()
    # elif free_spaces > 17:
    #     return heuristic_five()
    # else:
    #     return heuristic_six()
    return heuristic_three()


def quadrant_with_most_open_space(game):

    equator = int(game.width / 2)
    prime_meridian = int(game.height / 2)

    quadrant_1 = list()
    quadrant_2 = list()
    quadrant_3 = list()
    quadrant_4 = list()

    for space in game.get_blank_spaces():
        if space[0] < equator and space[1] > prime_meridian:
            quadrant_1.append(space)
        elif space[0] < equator and space[1] < prime_meridian:
            quadrant_2.append(space)
        elif space[0] > equator and space[1] < prime_meridian:
            quadrant_3.append(space)
        elif space[0] > equator and space[1] > prime_meridian:
            quadrant_4.append(space)

    quadrants = [quadrant_1, quadrant_2, quadrant_3, quadrant_4]

    most_moves_quadrant = max(enumerate(quadrants), key=lambda tup: len(tup[1]))

    return most_moves_quadrant

def get_longest_path_length(game, player):
    opponent = game.get_opponent(player)
    game._active_player = player
    game._inactive_player = opponent
    longest = 0

    for move in game.get_legal_moves():
        path_length = get_longest_path_length(game.forecast_move(move), player) + 1
        if path_length > longest:
            longest = path_length

        if longest > 20:
            logging.warning('Depth 20 reached on longest path')
            return longest
    return longest


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=14., montecarlo_max_moves=100,
                 montecarlo_C=1.4):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.montecarlo_board_states = []
        self.montecarlo_plays ={}
        self.montecarlo_wins = {}
        self.montecarlo_max_moves = montecarlo_max_moves
        self.iterative_depth = 1
        self.montecarlo_C = montecarlo_C

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!
        logging.debug('Active Player: ' + str(game.active_player))
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # local variable for optimizations sake
        legal_moves = game.get_legal_moves()

        # check for no legal moves
        if not legal_moves:
            return (-1, -1)

        if len(legal_moves) == 1:
            return legal_moves[0]

        # initializations
        move = legal_moves[0]
        logging.debug('moves: ' + str(legal_moves))
        logging.debug('initial move: ' + str(move))

        # check if a move has not been made
        # TODO: finish implementing this logic



        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == "minimax":
                if self.iterative:
                    iterative_depth = 1
                    while True:
                        #logging.debug('Iterative Depth: ' + str(iterative_depth))
                        _, move = self.minimax(game, iterative_depth)
                        iterative_depth += 1
                else:
                    _, move = self.minimax(game, self.search_depth)

            elif self.method == "alphabeta":
                if self.iterative:
                    self.iterative_depth = 1
                    while True:
                        #logging.debug('iterative_depth: ' + str(iterative_depth))
                        _, move = self.alphabeta(game, self.iterative_depth)
                        #logging.debug('move_choice: ' + str(move))

                        self.iterative_depth += 1
                else:
                    _, move = self.alphabeta(game, self.search_depth)
            elif self.method == "montecarlo":
                self.iterative_depth = 0
                # self.timeout = 15
                player = game.active_player

                games = 0
                begin = datetime.datetime.utcnow()
                while self.time_left() >= self.TIMER_THRESHOLD + 18:
                    self.run_monte_carlo_simulation(game)
                    games += 1

                moves_states = [(m, game.forecast_move(m).hash()) for m in legal_moves]

                # Display the number of call of `run_simulation` and the time elapsed
                logging.debug('Games: ' + str(games) + ' Time Elapsed: ' + str(datetime.datetime.utcnow() - begin))

                # Pick the move with highest percentage of wins
                # m, S = moves_states[0]
                # if self.montecarlo_wins.get((player, S), 0):
                #     print('match_found')

                percent_wins, move = max(
                    (self.montecarlo_wins.get((player, S), 0) /
                     self.montecarlo_plays.get((player, S), 1),
                      m)
                    for m, S in moves_states
                )

                # Display the stats for each possible play
                for x in sorted(
                        ((100 * self.montecarlo_wins.get((player, S), 0) /
                          self.montecarlo_plays.get((player, S), 1),
                          self.montecarlo_wins.get((player, S), 0),
                          self.montecarlo_plays.get((player, S), 0), m)
                            for m, S in moves_states),
                    reverse=True
                ):
                    logging.debug("{3}: {0:.2f}% ({1} / {2})".format(*x))

                logging.debug("Maximum depth searched: " + str(self.iterative_depth))


        except Timeout:
            # Handle any actions required at timeout, if necessary
            logging.debug("Ran out of time")
            if self.iterative:
                logging.debug("Depth: " + str(self.iterative_depth))
            logging.debug("move: " + str(move))
            return move

        # Return the best move from the last completed search iteration
        logging.debug("move: " + str(move))
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        # logging.debug("Time Threshold: " + str(self.TIMER_THRESHOLD))
        # logging.debug("Time Left: " + str(self.time_left()))
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        best_move = (int, int)

        moves = game.get_legal_moves()

        if len(moves) == 0 or depth == 0:
            # logging.debug('score: ' + str(self.score(game,self)))
            return self.score(game, self), game.get_player_location(self)

        # max level - equivalent to max-value in the pseudocode
        if maximizing_player:

            v = float("-inf")

            for move in moves:
                #logging.debug("maximizing_player")
                #logging.debug(move)
                next_v = self.minimax(game.forecast_move(move), depth - 1, False)[0]
                if next_v > v:
                    v = next_v
                    best_move = move
                    #logging.debug('best_move: ' + str(best_move))
                #logging.debug(v)
            return v, best_move

        # min level - equivalent to min-value in the pseudocode
        else:

            v = float("inf")   # stand in for +inf for the scale of our isolation game

            for move in moves:
                # logging.debug("minimizing_player")
                next_v = self.minimax(game.forecast_move(move), depth - 1, True)[0]
                if next_v < v:
                    v = next_v
                    best_move = move
            return v, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #logging.debug('depth: ' + str(depth))
        #logging.debug("Time Left: " + str(self.time_left())
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        #moves = game.get_legal_moves()

        # if len(moves) == 0 or depth == 0: #(self.iterative and depth == 0):
        #     #logging.debug('score: ' + str(self.score(game,self)))
        #     return self.score(game, self), game.get_player_location(self)

        if not game.get_legal_moves():
            #logging.warning('AlphaBeta: No Legal Moves')
            return game.utility(self), (-1, -1)
        if depth == 0:
            return self.score(game, self), (-1, -1)

        else:
            best_move = game.get_legal_moves()[0]
            # max level - equivalent to max-value in the pseudocode
            if maximizing_player:

                v = float("-inf")

                for move in game.get_legal_moves():
                    #logging.debug("maximizing_player")
                    #logging.debug('move: ' + str(move))
                    #logging.debug('depth: ' + str(depth))
                    next_v = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, False)[0]

                    if next_v > v:
                        v = next_v
                        best_move = move
                    #logging.debug("v: " + str(v))
                    if v >= beta:
                        return v, best_move
                    alpha = max(alpha, v)
                    #logging.debug("alpha: " + str(alpha))
                return v, best_move

            # min level - equivalent to min-value in the pseudocode
            else:

                v = float("inf")   # stand in for +inf for the scale of our isolation game

                for move in game.get_legal_moves():
                    #logging.debug("minimizing_player")
                    #logging.debug('move: ' + str(move))
                    #logging.debug('depth: ' + str(depth))
                    next_v = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, True)[0]

                    if next_v < v:
                        v = next_v
                        best_move = move
                    #logging.debug("v: " + str(v))
                    if v <= alpha:
                        return v, best_move
                    beta = min(beta, v)
                    #logging.debug("beta: " + str(alpha))
                return v, best_move

    # def montecarlo_choose_move(self, move_states):
    #     legal_moves[randint(0, len(legal_moves) - 1)]
    #
    #     return move, state

    def run_monte_carlo_simulation(self, game):
        # some optimization with local variable lookups instead of attribute lookups
        plays, wins = self.montecarlo_plays, self.montecarlo_wins

        visited_states = set()
        game_states = [game]
        state = game_states[-1]
        #state = game
        player = game.active_player

        expand = True
        for t in range(1, self.montecarlo_max_moves + 1):
            legal_moves = state.get_legal_moves()
            moves_states = [(m, state.forecast_move(m).hash(), state.forecast_move(m)) for m in legal_moves]
            #moves_states_hashed = [(m, state.forecast_move(m).hash()) for m in legal_moves]

            if all(plays.get((player, Sh)) for m, Sh, S in moves_states):
                # if we have stats on all of the legal moves here, use them
                log_total = math.log(
                    sum(plays[(player, Sh)] for m, Sh, S in moves_states))

                value, move, _, state = max(
                    ((wins[(player, Sh)] / plays[(player, Sh)]) +
                     self.montecarlo_C * math.sqrt(log_total / plays[(player, Sh)]), m, Sh, S)
                    for m, Sh, S in moves_states
                )
            else:
                # otherwise make an arbitrary decision
                move, _, state = choice(moves_states)
                # logging.debug('Montecarlo Else')
                # best_val = float("-inf")
                # for m, Sh, S in moves_states:
                #     val = self.score(S, player)
                #     if val > best_val:
                #         best_val = val
                #         move = m
                #         state = S
                # value, move, _, state = max(
                #     (self.score(S, player), m, Sh, S) for m, Sh, S in moves_states
                # )


            # move = choice(game.get_legal_moves())
            # state = game.forecast_move(move)
            game_states.append(state)

            # `player` here and below refers to the player
            # who moved into that particular state
            if expand and (player, state.hash()) not in plays:
                expand = False

                plays[(player, state.hash())] = 0
                wins[(player, state.hash())] = 0
                if t > self.iterative_depth:
                    self.iterative_depth = t

            visited_states.add((player, state.hash()))

            # check to see if the player that made the move just won
            winner = None
            if state.is_winner(player):
                winner = player
            elif state.is_loser(player):
                winner = game.get_opponent(player)

            # set the player to the player who will be making the next move
            player = state.active_player

            # if there is a winner this turn, no need to simulate longer
            if winner:
                break

        # collect the statistics for the simulation
        for player, state in visited_states:
            if (player, state) not in plays:
                continue
            plays[(player, state)] += 1
            if player != winner:
                wins[(player, state)] += 1

        #self.montecarlo_plays = plays
        #self.montecarlo_wins = wins
