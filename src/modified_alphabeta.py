from typing import Optional
import random
from schnapsen.game import Bot, PlayerPerspective, Move, GamePhase
from schnapsen.bots import AlphaBetaBot

class ModifiedAlphaBetaBot(Bot):
    """
    A bot that plays random moves in the first phase of the game and follows the AlphaBetaBot strategy in the second phase.
    """

    def __init__(self, rand: random.Random, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.rand = rand
        self.alphabeta_bot = AlphaBetaBot(name = name)

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        phase = perspective.get_phase()
        
        if phase == GamePhase.ONE:
            valid_moves = perspective.valid_moves()
            return self.rand.choice(valid_moves)
        
        elif phase == GamePhase.TWO:
            return self.alphabeta_bot.get_move(perspective, leader_move)
        
        else:
            valid_moves = perspective.valid_moves()
            return self.rand.choice(valid_moves)