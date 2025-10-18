# Reinforcement Learning in Schnapsen  
*Vrije Universiteit Amsterdam – Project Intelligent Systems 2024-2025*  

## Overview  
This project implements a **reinforcement learning agent** to play the two-player card game **Schnapsen**, expanding the standard 20-card version to a **24-card environment** to test scalability and learning adaptability. The agent uses **Q-learning with a neural network (Q-network)** to estimate state–action values, enabling autonomous decision-making in dynamic game conditions.  

Experiments were conducted to compare the trained model’s performance against several rule-based and search-based bots (RandBot, BullyBot, AlphaBetaBot, MinimaxBot, and RdeepBot).  
Statistical analyses using *t-tests* and *Wilcoxon signed-rank tests* confirmed that the model maintains stable learning performance across increasing complexity.  

**Keywords:** Reinforcement Learning, Q-Learning, Neural Networks, Python, Game AI  

---

## Features  
- Reinforcement learning bot with Q-value approximation using neural networks  
- Configurable training environment and opponent selection  
- CLI and GUI interfaces for interactive gameplay and testing  
- Evaluation using paired t-test and Wilcoxon signed-rank test  
- Modular design for easy implementation of additional bots  

---

## Installation  

### Prerequisites  
- Python **≥ 3.10** (tested on Python 3.13)  
- Conda or venv recommended for environment management  

## Project Structure
Reinforcement-Learning-in-Schnapsen/
│
├── schnapsen/                     # Core game and logic
│   ├── pycache/
│   ├── .github/
│   ├── build/
│   ├── executables/               # CLI and server scripts
│   ├── src/
│   │   └── schnapsen/
│   │       ├── pycache/
│   │       ├── alternative_engines/
│   │       ├── bots/              # Bot implementations (RL, rule-based, hybrid)
│   │       ├── deck.py            # Card and deck logic
│   │       ├── game.py            # Game environment and rules
│   │       ├── py.typed
│   │       ├── learning_mlbot.py  # Reinforcement learning agent
│   │       ├── mock_player_perspective.py
│   │       ├── modified_alphabeta_bot.py
│   │       ├── modified_minimax_bot.py
│   │       ├── statistical_test.py
│   │       └── main.py
│
├── tests/                         # Unit tests
│   ├── bots/
│   ├── test_deck.py
│   ├── test_game.py
│   ├── test_repr.py
│   └── test_schnapsen_integration.py
│
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.cfg
└── setup.py

---

## Research Context 
Developed as part of Project Intelligent Systems at Vrije Universiteit Amsterdam, this work explores reinforcement learning scalability in games with partial and full information.
The agent demonstrates consistent adaptability when faced with increased state and action spaces, highlighting the robustness of Q-learning in uncertain environments.

---

## License - This project is distributed for educational and research purposes.

Author: Barış Kaban / Cem Saygıvar
Artificial Intelligence BSc, Vrije Universiteit Amsterdam

---
## Environment Setup
```bash
# Create and activate environment
conda create --name schnapsen_rl python=3.10
conda activate schnapsen_rl

# Clone and install
git clone https://github.com/bariskaban/Reinforcement-Learning-in-Schnapsen.git
cd Reinforcement-Learning-in-Schnapsen
pip install -e .

# Run tests
pip install -e ".[test]"
pytest ./tests

# Command-Line Interface (CLI)
python executables/cli.py random-game

# Graphical User Interface (GUI)
python executables/server.py

# Example: training RL bot vs RandBot
python train_rl_bot.py --opponent randbot --games 10000 --train 3000
