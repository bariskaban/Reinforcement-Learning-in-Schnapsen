import random
from typing import List, Optional
from schnapsen.game import SchnapsenGamePlayEngine, RegularMove, Marriage, TrumpExchange, LeaderPerspective, PlayerPerspective, Score, Hand, GamePhase, SchnapsenDeckGenerator, Bot
from schnapsen.deck import Card, Suit, Rank, OrderedCardCollection
from schnapsen.alternative_engines.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine
from schnapsen.bots import RandBot, BullyBot
from schnapsen.bots.rdeep import RdeepBot
from schnapsen.bots.ml_bot import MLPlayingBot
from modified_alphabeta import ModifiedAlphaBetaBot
from modified_minimax import ModifiedMiniMaxBot
from learning_mlbot import LearningMLBot, generate_custom_deck
from mock_player_perspective import MockPlayerPerspective



seed = 4765
rng = random.Random(seed)
regular_schnapsen = SchnapsenGamePlayEngine()
twenty_four_schnapsen = TwentyFourSchnapsenGamePlayEngine()


randbot = RandBot(rand = rng)
randbot.name = "Randbot"
bullybot = BullyBot(rand = rng)
bullybot.name = "Bullybot"
modified_alphabetabot = ModifiedAlphaBetaBot(rand = rng)
modified_alphabetabot.name = "ModifiedAlphabetabot" 
modified_minimaxbot = ModifiedMiniMaxBot(rand = rng)
modified_minimaxbot.name = "Minimaxbot"
rdeepbot = RdeepBot(num_samples= 10, depth= 5, rand = rng)
rdeepbot.name = "Rdeep"



def simulating_player_perspective():
    return MockPlayerPerspective()

temp_bot = LearningMLBot(input_size=1, hidden_size=128, action_size=5)  
input_size = len(temp_bot.encode_state(simulating_player_perspective(), None))  
del temp_bot

hidden_size = 128
action_size = len(generate_custom_deck()._cards)
learning_ml_bot = LearningMLBot(input_size, hidden_size, action_size, epsilon = 0.5, lr = 0.0005, gamma = 0.95, buffer_size = 100000, batch_size = 8)


def regular_game_simulation(allybot, opponentbot, num_games: int, game_engine: SchnapsenGamePlayEngine) -> float:
    """
    Simulate games between two bots and return the win rate of the ally bot.
    """
    allybot_wins: int = 0


    for _ in range(num_games):
        winner, _, _ = game_engine.play_game(allybot, opponentbot, rng)
        if winner == allybot:
            allybot_wins += 1
    
    winning_rate = (allybot_wins / num_games)
    return winning_rate

def training(bot: LearningMLBot, opponent: Bot, num_games: int, game_engine: SchnapsenGamePlayEngine) -> None:
    """
    Train the bot by playing games against the specified opponent.
    """
    
    for game_index in range(num_games):
        if bot.early_stop:  
            print(f"[INFO] Early stopping triggered at game {game_index}. Stopping training.")
            break
        winner, _, _ = game_engine.play_game(bot, opponent, rng)


        if game_index % 100 == 0:
            bot.update_target_network()
        
        
        if len(bot.replay_buffer) >= bot.batch_size and game_index % 100 == 0:
            print(f"[DEBUG] Calling train at game {game_index}. Buffer size: {len(bot.replay_buffer)}")
            bot.train(game_index)


        if game_index % 1000 == 0:
            win_rate = regular_game_simulation(bot, opponent, 100, regular_schnapsen)
            #print(f"[DEBUG] Win rate after {game_index} games: {win_rate * 100:.2f}%")


    
    print(f"[INFO] Training completed against {opponent.name}.")

def reset_learning_ml_bot(bot: LearningMLBot) -> None:
    """
    Resets the replay buffer of the LearningMLBot.
    """
    bot.replay_buffer.clear()
    print("[INFO] LearningMLBot's memory has been reset.")

if __name__ == "__main__":
    num_games: int = 10000
    training_games: int = 3000
    bots_to_test = [randbot, bullybot, modified_alphabetabot, modified_minimaxbot, rdeepbot]
    testiiing = [randbot]

    for test_bot in bots_to_test:
        reset_learning_ml_bot(learning_ml_bot)
        
        print(f"Evaluating LearningMLBot against {test_bot.name} before training...")
        pre_train_win_rate = regular_game_simulation(learning_ml_bot, test_bot, num_games, regular_schnapsen)
        print(f"Win rate before training against {test_bot.name} in regular schnapsen: {pre_train_win_rate * 100:.2f}%")

        print(f"Training LearningMLBot against {test_bot.name}...")
        training(learning_ml_bot, test_bot, training_games, regular_schnapsen)

        print(f"Evaluating LearningMLBot against {test_bot.name} after training...")
        post_train_win_rate = regular_game_simulation(learning_ml_bot, test_bot, num_games, regular_schnapsen)
        print(f"Win rate after training against {test_bot.name} in regular schnapsen: {post_train_win_rate * 100:.2f}%")
    
    for test_bot in bots_to_test:
        reset_learning_ml_bot(learning_ml_bot)
        
        print(f"Evaluating LearningMLBot against {test_bot.name} before training...")
        pre_train_win_rate = regular_game_simulation(learning_ml_bot, test_bot, num_games, twenty_four_schnapsen)
        print(f"Win rate before training against {test_bot.name} in twenty four schnapsen: {pre_train_win_rate * 100:.2f}%")

        print(f"Training LearningMLBot against {test_bot.name}...")
        training(learning_ml_bot, test_bot, training_games, twenty_four_schnapsen)

        print(f"Evaluating LearningMLBot against {test_bot.name} after training...")
        post_train_win_rate = regular_game_simulation(learning_ml_bot, test_bot, num_games, twenty_four_schnapsen)
        print(f"Win rate after training against {test_bot.name} in twenty four schnapsen: {post_train_win_rate * 100:.2f}%")
    

    
    
