from schnapsen.game import PlayerPerspective, Hand, SchnapsenDeckGenerator
from schnapsen.deck import Card, Suit, Rank

from schnapsen.game import Score, GamePhase
from typing import List, Optional


class MockPlayerPerspective(PlayerPerspective):
    def __init__(self):
        deck_generator = SchnapsenDeckGenerator()
        all_cards = list(deck_generator.get_initial_deck())
 
        self._hand = Hand(cards=[all_cards[0], all_cards[1]])  
        self._trump_card = all_cards[2]  
        self._won_cards = Hand(cards=[all_cards[3]])
        self._opponent_won_cards = Hand(cards=[all_cards[4]])
        self._known_opponent_cards = Hand(cards=[all_cards[5]])

        self._my_score = Score(direct_points=0, pending_points=0)
        self._opponent_score = Score(direct_points=0, pending_points=0)
        self._trump_suit = Suit.HEARTS
        self._phase = GamePhase.ONE
        self._talon_size = 10
        self._is_leader = True

    def am_i_leader(self) -> bool:
        return self._is_leader

    def get_hand(self) -> Hand:
        return self._hand

    def get_my_score(self) -> Score:
        return self._my_score

    def get_opponent_score(self) -> Score:
        return self._opponent_score

    def get_opponent_won_cards(self) -> Hand:
        return self._opponent_won_cards

    def get_won_cards(self) -> Hand:
        return self._won_cards

    def get_opponent_hand_in_phase_two(self) -> Optional[Hand]:
        return None

    def valid_moves(self) -> List[Card]:
        return self._hand.cards

    def get_trump_suit(self) -> Suit:
        return self._trump_suit

    def get_phase(self) -> GamePhase:
        return self._phase

    def get_talon_size(self) -> int:
        return self._talon_size

    def get_trump_card(self) -> Optional[Card]:
        return self._trump_card

    def get_known_cards_of_opponent_hand(self) -> Hand:
        return self._known_opponent_cards



