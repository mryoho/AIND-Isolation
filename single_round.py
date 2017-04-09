import random
import warnings
import logging

from collections import namedtuple

from isolation import Board
from sample_players import RandomPlayer
from sample_players import null_score
from sample_players import open_move_score
from sample_players import improved_score
from game_agent import CustomPlayer
from game_agent import custom_score

NUM_MATCHES = 5  # number of matches against each opponent
TIME_LIMIT = 150  # number of milliseconds before timeout

Agent = namedtuple("Agent", ["player", "name"])

def play_game(player1, player2):
    num_timeouts = {player1: 0, player2: 0}
    num_invalid_moves = {player1: 0, player2: 0}

    game = Board(player1, player2)

    # initialize game with a random move and response
    for _ in range(2):
        move = random.choice(game.get_legal_moves())
        game.apply_move(move)
    # apply specific opening moves to test longest path
    # game.apply_move((3, 3))
    # game.apply_move((0, 1))

    winner, history, termination = game.play(time_limit=TIME_LIMIT)

    if player1 == winner:
        print('Winner: Player 1')
    elif player2 == winner:
        print('Winner: Player 2')
    print('Termination: ' + termination)
    print('Move History: ' + str(history))


def main():
    CUSTOM_ARGS = {"method": 'montecarlo', 'iterative': False}

    student = Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS), "ID_Improved")
    improved = Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS), "Student")

    play_game(student.player, improved.player)

if __name__ == "__main__":
    Log = logging.getLogger()
    Log.setLevel(logging.DEBUG)
    main()
