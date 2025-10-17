import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from schnapsen.game import Bot, PlayerPerspective, Move, GamePhase, SchnapsenDeckGenerator, RegularMove, Marriage, TrumpExchange
from schnapsen.deck import Rank, Suit, OrderedCardCollection, Card
from typing import List, Optional, Tuple


def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> List[int]:
        """
        Translating the suit of a card into a one-hot vector encoding of size 4.
        """
        if card_suit == Suit.HEARTS:
            return [0, 0, 0, 1]
        elif card_suit == Suit.CLUBS:
            return [0, 0, 1, 0]
        elif card_suit == Suit.SPADES:
            return [0, 1, 0, 0]
        elif card_suit == Suit.DIAMONDS:
            return [1, 0, 0, 0]
        else:
            raise ValueError("Unknown card suit!")

    
def generate_custom_deck():
    """
    Generate the complete 52-card Schnapsen deck.
    """
    custom_cards = [
        Card.JACK_HEARTS, Card.QUEEN_HEARTS, Card.KING_HEARTS, Card.TEN_HEARTS, Card.ACE_HEARTS,
        Card.NINE_HEARTS, Card.EIGHT_HEARTS, Card.SEVEN_HEARTS, Card.SIX_HEARTS, Card.FIVE_HEARTS,
        Card.FOUR_HEARTS, Card.THREE_HEARTS, Card.TWO_HEARTS,

        Card.JACK_CLUBS, Card.QUEEN_CLUBS, Card.KING_CLUBS, Card.TEN_CLUBS, Card.ACE_CLUBS,
        Card.NINE_CLUBS, Card.EIGHT_CLUBS, Card.SEVEN_CLUBS, Card.SIX_CLUBS, Card.FIVE_CLUBS,
        Card.FOUR_CLUBS, Card.THREE_CLUBS, Card.TWO_CLUBS,

        Card.JACK_SPADES, Card.QUEEN_SPADES, Card.KING_SPADES, Card.TEN_SPADES, Card.ACE_SPADES,
        Card.NINE_SPADES, Card.EIGHT_SPADES, Card.SEVEN_SPADES, Card.SIX_SPADES, Card.FIVE_SPADES,
        Card.FOUR_SPADES, Card.THREE_SPADES, Card.TWO_SPADES,

        Card.JACK_DIAMONDS, Card.QUEEN_DIAMONDS, Card.KING_DIAMONDS, Card.TEN_DIAMONDS, Card.ACE_DIAMONDS,
        Card.NINE_DIAMONDS, Card.EIGHT_DIAMONDS, Card.SEVEN_DIAMONDS, Card.SIX_DIAMONDS, Card.FIVE_DIAMONDS,
        Card.FOUR_DIAMONDS, Card.THREE_DIAMONDS, Card.TWO_DIAMONDS,
    ]
    assert len(custom_cards) == 52, f"Expected 52 cards, got {len(custom_cards)}"
    return OrderedCardCollection(cards=custom_cards)




class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class LearningMLBot(Bot):
    def __init__(self, input_size: int, hidden_size: int, action_size: int, epsilon: float = 0.1, lr: float = 0.001, gamma: float = 0.95, buffer_size: int = 10000, batch_size: int = 16, patience: int = 5):
        super().__init__("LearningMLBot")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_to_index = self.create_move_to_index_mapping()
        input_size = 382
        self.q_network = QNetwork(input_size, hidden_size, len(self.move_to_index)).to(self.device)
        self.target_network = QNetwork(input_size, hidden_size, len(self.move_to_index)).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995  
        self.min_epsilon = 0.1
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_size = len(self.move_to_index)
        self.epsilon = 0.5
        self.epsilon_decay = 0.9999

        self.patience = patience
        self.best_loss = float('inf')
        self.stagnation_counter = 0
        self.early_stop = False

    def create_move_to_index_mapping(self) -> dict:
        """
        Creates a mapping from moves to unique indices.
        """
        move_to_index = {}
        idx = 0

        for card in generate_custom_deck()._cards:
            move_to_index[RegularMove(card)] = idx
            idx += 1

        for suit in [Suit.HEARTS, Suit.CLUBS, Suit.SPADES, Suit.DIAMONDS]:
            queen_card = getattr(Card, f"QUEEN_{suit.name}")
            king_card = getattr(Card, f"KING_{suit.name}")
            move_to_index[Marriage(queen_card=queen_card, king_card=king_card)] = idx
            idx += 1
        
        for jack_card in [Card.JACK_HEARTS, Card.JACK_CLUBS, Card.JACK_SPADES, Card.JACK_DIAMONDS]:
            move_to_index[TrumpExchange(jack=jack_card)] = idx
            idx += 1
        
        return move_to_index
    
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        state = self.encode_state(perspective, leader_move)
        valid_moves = perspective.valid_moves()
    
        if not valid_moves:
            raise ValueError("No valid moves available for this state.")
        
        valid_move_indices = []
        valid_moves_mapped = []

        for move in valid_moves:
            mapped_move = RegularMove(move) if isinstance(move, Card) else move
            if mapped_move in self.move_to_index:
                valid_move_indices.append(self.move_to_index[mapped_move])
                valid_moves_mapped.append(mapped_move)
            else:
                print(f"[WARNING] Move not found in move_to_index mapping: {mapped_move}") 
        if not valid_move_indices:
            return random.choice(valid_moves)
        
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze(0)
                valid_q_values = q_values[valid_move_indices]
                
                if len(valid_q_values) == 0:
                    print("[ERROR] No valid Q-values for the current state. Defaulting to random choice.")
                    return random.choice(valid_moves)

                best_move_idx = valid_q_values.argmax().item()
                chosen_move = valid_moves_mapped[best_move_idx]
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            return chosen_move


    def encode_state(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> List[float]:
        """
        Encode the game state as a feature vector.
        """
        state_feature_list: List[float] = []

      
        player_score = perspective.get_my_score()
        opponent_score = perspective.get_opponent_score()
        state_feature_list += [player_score.direct_points, player_score.pending_points]
        state_feature_list += [opponent_score.direct_points, opponent_score.pending_points]

        
        trump_suit = perspective.get_trump_suit()
        trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)
        state_feature_list += trump_suit_one_hot

        
        game_phase_encoded = [1, 0] if perspective.get_phase() == GamePhase.TWO else [0, 1]
        state_feature_list += game_phase_encoded

        
        talon_size = perspective.get_talon_size()
        state_feature_list += [talon_size]

        
        i_am_leader = [1, 0] if perspective.am_i_leader() else [0, 1]
        state_feature_list += i_am_leader

        
        hand_cards = perspective.get_hand().cards
        trump_card = perspective.get_trump_card()
        won_cards = perspective.get_won_cards().get_cards()
        opponent_won_cards = perspective.get_opponent_won_cards().get_cards()
        opponent_known_cards = perspective.get_known_cards_of_opponent_hand().get_cards()

       
        all_cards = generate_custom_deck()._cards
        hand_card_ranks = [card.rank.value for card in hand_cards]
        max_hand_size = 5
        while len(hand_card_ranks) < max_hand_size:
            hand_card_ranks.append(0)  
        hand_card_ranks += [0] * (5 - len(hand_card_ranks))
        state_feature_list += hand_card_ranks
        cards_played = [1 if card in won_cards or card in opponent_won_cards else 0 for card in all_cards]
        state_feature_list += cards_played
        
        deck_knowledge = []
        for card in all_cards:
            if card in perspective.get_hand().cards:
                deck_knowledge += [1, 0, 0, 0, 0, 0]
            elif card in perspective.get_won_cards().get_cards():
                deck_knowledge += [0, 1, 0, 0, 0, 0]
            elif card in perspective.get_known_cards_of_opponent_hand().get_cards():
                deck_knowledge += [0, 0, 1, 0, 0, 0]
            elif card in perspective.get_opponent_won_cards().get_cards():
                deck_knowledge += [0, 0, 0, 1, 0, 0]
            elif card == perspective.get_trump_card():
                deck_knowledge += [0, 0, 0, 0, 1, 0]
            else:  
                deck_knowledge += [0, 0, 0, 0, 0, 1]
        

        state_feature_list += deck_knowledge

        state_feature_list = [float(feature) / 100 if isinstance(feature, int) else feature for feature in state_feature_list]
        
        expected_feature_size = len(deck_knowledge) + len(hand_card_ranks) + len(cards_played) + 13

        
        assert len(state_feature_list) == expected_feature_size, (f"Expected {expected_feature_size} features, got {len(state_feature_list)}, ", f"Deck knowledge feature count: {len(deck_knowledge)}, Total state feature length: {len(state_feature_list)}")

        return state_feature_list
    
    def calculate_reward(self, perspective: PlayerPerspective, move: Move, outcome: str, leader_move: Optional[Move]) -> float:
        """
        Calculate the reward for the current move.
        :param perspective: The bot's current game perspective.
        :param move: The move being evaluated.
        :param outcome: The result of the move ("win", "loss", or "draw").
        :param leader_move: The leader's move in the current trick, if any.
        :return: A numeric reward value.
        """
        reward = 0.0
         
        if outcome == "win":
            reward += 10.0  
        elif outcome == "loss":
            reward -= 10.0  
        elif outcome == "draw":
            reward += 0.0  

        my_score = perspective.get_my_score().direct_points + perspective.get_my_score().pending_points
        opponent_score = (perspective.get_opponent_score().direct_points + perspective.get_opponent_score().pending_points)
        reward += (my_score - opponent_score) * 0.1  # Reward relative score advantage

       
        if isinstance(move, RegularMove) and move.card == perspective.get_trump_card():
            reward -= 1.0
       
        if isinstance(move, Marriage):
            reward += 2.0

        if perspective.get_phase() == GamePhase.TWO:
            reward += 1.5 
        
        if leader_move and perspective.am_i_leader():
            reward += 1.0  
        
        return reward

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store gameplay experience in the replay buffer.
        :param state: Encoded game state before the move.
        :param action: Action taken in the encoded state.
        :param reward: Reward received for the move.
        :param next_state: Encoded game state after the move.
        :param done: Whether the game has ended.
        """
        outcome = "win" if done and reward > 0 else "loss" if done and reward < 0 else "draw"
        calculated_reward = self.calculate_reward(state, action, outcome, None)
        self.replay_buffer.append((state, action, calculated_reward, next_state, done))
        print(f"[DEBUG] Experience added to buffer. Current size: {len(self.replay_buffer)}")

    def train(self, game_index: int):
        """
        Train the Q-network using experiences sampled from the replay buffer.
        """
        print("[DEBUG] Entered train method.")
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        
        
        batch, indices = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32, requires_grad=True).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        predictions = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        td_errors = (targets - predictions).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        loss = nn.MSELoss()(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       
        if game_index % 1000 == 0:  
            print(f"[DEBUG] Loss after {game_index} games: {loss.item()}")
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.patience:
                    print("[INFO] Early stopping triggered.")
                    self.early_stop = True
                    return

        if len(self.replay_buffer) % 500 == 0:
            self.update_target_network()

    def reset_early_stopping(self):
        """
        Reset early stopping parameters to restart training.
        """
        self.best_loss = float('inf')
        self.stagnation_counter = 0
        self.early_stop = False

    def update_target_network(self):
        """
        Updates the target network to match the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reset_memory(self):
        self.replay_buffer.clear()



class ReplayBuffer:
    """
    A simple Replay Buffer to store gameplay experiences.
    """
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """
        Initialize the buffer.
        :param capacity: Maximum number of experiences to store.
        :param alpha: Degree of prioritization (0 = uniform sampling, 1 = full prioritization).
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  
        self.alpha = alpha


    def add(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool, td_error: float = 1.0) -> None:
        """
        Add an experience to the buffer with an associated TD error.
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(td_error + 1e-5)

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences from the buffer.
        :param batch_size: Number of experiences to sample.
        :return: A list of sampled experiences.
        """
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = random.choices(range(len(self.buffer)), probabilities, k=batch_size)
        sampled_experiences = [self.buffer[i] for i in indices]
        return sampled_experiences, indices 

    def updated_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """
        Update priorities for sampled experiences.
        """
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) + 1e-5

    def size(self) -> int:
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

    

