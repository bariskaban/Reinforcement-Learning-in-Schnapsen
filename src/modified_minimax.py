from typing import Optional
import random
from schnapsen.game import Bot, PlayerPerspective, Move, GamePhase
from schnapsen.bots import MiniMaxBot

class ModifiedMiniMaxBot(Bot):
    """
    A bot that plays random moves in the first phase of the game and follows minimax strategy in the second phase of the game. 
    """
    def __init__(self, rand: random.Random, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.rand = rand
        self.minimax_bot = MiniMaxBot(name = name)

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        phase = perspective.get_phase()

        if phase == GamePhase.ONE:
            valid_moves = perspective.valid_moves()
            return self.rand.choice(valid_moves)
        
        elif phase == GamePhase.TWO:
            return self.minimax_bot.get_move(perspective, leader_move)
        
        else: 
            valid_moves = perspective.valid_moves()
            return self.rand.choice(valid_moves)
        
    