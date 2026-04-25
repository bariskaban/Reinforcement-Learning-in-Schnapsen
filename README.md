# Reinforcement Learning in Schnapsen

A neural network-based Q-learning agent for the strategic card game Schnapsen, featuring scalability experiments with extended deck configurations and comprehensive bot comparisons.

> A full research report is included in this repository: [`project_report.pdf`](./project_report.pdf)

---

## Authors

**Barış Kaban** | **Cem Saygıvar**  
*Artificial Intelligence BSc*  
*Vrije Universiteit Amsterdam*

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Approach](#technical-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Research Findings](#research-findings)
- [Bot Opponents](#bot-opponents)
- [Testing & Evaluation](#testing--evaluation)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

This project implements a **reinforcement learning agent** capable of playing Schnapsen, a classic two-player trick-taking card game. The system extends the standard 20-card version to a **24-card environment** to investigate learning scalability and adaptability in increasingly complex state spaces.

### What is Schnapsen?

Schnapsen is a strategic Austrian card game requiring players to:
- Win tricks to accumulate points
- Make strategic decisions about card play and trump declarations
- Manage partial information about opponent hands
- Balance aggressive and defensive tactics

### Research Motivation

This project explores whether Q-learning with neural network approximation can:
- Scale effectively to larger state and action spaces
- Maintain performance against diverse opponent strategies
- Learn robust policies in environments with partial observability
- Generalize from training opponents to novel adversaries

---

## Key Features

### Reinforcement Learning

- **Q-Learning Algorithm**: Neural network-based Q-value approximation
- **Adaptive Learning**: Self-improving through iterative gameplay
- **Scalable Architecture**: Successfully handles 20-card and 24-card deck configurations
- **Experience Replay**: Efficient learning from historical game states

### Opponent Diversity

- **RandBot**: Random action selection baseline
- **BullyBot**: Aggressive rule-based strategy
- **MinimaxBot**: Perfect information minimax search
- **AlphaBetaBot**: Optimized alpha-beta pruning
- **RdeepBot**: Advanced hybrid bot

### Evaluation & Analysis

- **Statistical Validation**: Paired t-tests and Wilcoxon signed-rank tests
- **Performance Metrics**: Win rate, average score, decision quality
- **Cross-Opponent Testing**: Generalization assessment across bot types
- **Complexity Analysis**: Scalability experiments with extended decks

### User Interfaces

- **Command-Line Interface (CLI)**: Headless bot tournaments and training
- **Graphical User Interface (GUI)**: Interactive web-based gameplay
- **Configurable Environments**: Adjustable opponents, game parameters, and training settings

---

## Technical Approach

### Q-Learning with Neural Networks

The agent estimates Q-values Q(s, a) using a feedforward neural network:

```
State → Feature Extraction → Neural Network → Q-Values → Action Selection
```

**Training Process:**
1. Observe current game state
2. Select action using ε-greedy policy
3. Execute action and observe reward
4. Update Q-network using TD-error
5. Store experience in replay buffer
6. Periodically sample and train on batches

### State Representation

The state space includes:
- Current hand composition
- Visible cards (tricks won, trump card)
- Game score and phase
- Legal action mask
- Opponent modeling features

### Reward Structure

- **Winning the game**: +100
- **Losing the game**: -100
- **Intermediate rewards**: Based on points gained per trick
- **Penalty for illegal moves**: -10

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher (tested on Python 3.13)
- **Environment Manager**: Conda or venv recommended

### Step 1: Create Virtual Environment

```bash
# Using Conda
conda create --name schnapsen_rl python=3.10
conda activate schnapsen_rl

# Using venv (alternative)
python -m venv schnapsen_rl
source schnapsen_rl/bin/activate  # On Windows: schnapsen_rl\Scripts\activate
```

### Step 2: Clone Repository

```bash
git clone https://github.com/<your-username>/Reinforcement-Learning-in-Schnapsen.git
cd Reinforcement-Learning-in-Schnapsen
```

### Step 3: Install Package

```bash
# Standard installation
pip install -e .

# With testing dependencies
pip install -e ".[test]"
```

### Step 4: Verify Installation

```bash
# Run test suite
pytest ./tests

# Quick functionality check
python executables/cli.py random-game
```

---

## Usage

### Command-Line Interface (CLI)

#### Run Bot vs. Bot Matches

```bash
# Random bot vs. random bot
python executables/cli.py random-game

# RL bot vs. specific opponent
python executables/cli.py --player1 rlbot --player2 alphabetabot --games 100

# Tournament mode (all bots)
python executables/cli.py tournament --rounds 50
```

#### Training the RL Agent

```bash
# Train against RandBot for 10,000 games
python train_rl_bot.py --opponent randbot --games 10000 --train 3000

# Train with custom parameters
python train_rl_bot.py \
    --opponent bullybot \
    --games 20000 \
    --train 5000 \
    --learning-rate 0.001 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.995
```

#### Evaluation and Statistics

```bash
# Evaluate trained model against multiple opponents
python executables/evaluate.py \
    --model models/rl_bot_trained.pth \
    --opponents randbot bullybot alphabetabot \
    --games 500

# Run statistical significance tests
python src/schnapsen/statistical_test.py \
    --model1 models/rl_bot_v1.pth \
    --model2 models/rl_bot_v2.pth \
    --games 1000
```

### Graphical User Interface (GUI)

```bash
# Start web server
python executables/server.py

# Open browser to http://localhost:5000
# Play interactively or watch bot matches
```

**GUI Features:**
- Interactive card selection
- Real-time game state visualization
- Bot vs. bot spectating mode
- Game history and replay

---

## Project Structure

```
Reinforcement-Learning-in-Schnapsen/
│
├── executables/                    # Entry points
│   ├── cli.py                      # Command-line interface
│   ├── server.py                   # Web server for GUI
│   └── train_rl_bot.py             # Training script
│
├── src/
│   └── schnapsen/
│       ├── bots/                   # Bot implementations
│       │   ├── rl_bot.py           # Q-learning agent
│       │   ├── rand_bot.py         # Random baseline
│       │   ├── bully_bot.py        # Aggressive rule-based
│       │   ├── alphabeta_bot.py    # Alpha-beta search
│       │   ├── minimax_bot.py      # Minimax search
│       │   └── rdeep_bot.py        # Hybrid advanced bot
│       │
│       ├── alternative_engines/    # Game variants (24-card)
│       ├── deck.py                 # Card and deck logic
│       ├── game.py                 # Core game engine
│       ├── learning_mlbot.py       # RL training utilities
│       ├── modified_alphabeta_bot.py
│       ├── modified_minimax_bot.py
│       ├── statistical_test.py     # Evaluation metrics
│       └── main.py                 # Core game loop
│
├── tests/                          # Unit and integration tests
│   ├── bots/
│   ├── test_deck.py
│   ├── test_game.py
│   ├── test_repr.py
│   └── test_schnapsen_integration.py
│
├── Executing_instructions.txt
├── LICENSE
├── README.md
├── project_report.pdf
├── pyproject.toml
├── setup.cfg
└── setup.py
```

---

## Research Findings

### Performance Summary

The Q-learning agent achieved competitive performance against diverse opponents:

| Opponent | Win Rate (20-card) | Win Rate (24-card) | Statistical Significance |
|----------|-------------------:|-------------------:|------------------------:|
| RandBot  | 78.3% | 76.1% | p < 0.01 |
| BullyBot | 64.2% | 61.8% | p < 0.05 |
| MinimaxBot | 52.7% | 51.3% | p > 0.05 (ns) |
| AlphaBetaBot | 48.9% | 47.2% | p > 0.05 (ns) |
| RdeepBot | 42.1% | 40.5% | p > 0.05 (ns) |

### Key Insights

**Scalability**
- The agent maintains stable learning performance when transitioning from 20-card to 24-card decks
- Statistical tests (t-test, Wilcoxon) confirm no significant performance degradation
- Q-network architecture generalizes effectively to larger state spaces

**Learning Dynamics**
- Fastest learning against predictable opponents (RandBot, BullyBot)
- Slower but consistent improvement against search-based opponents
- Evidence of strategic adaptation beyond rule-based heuristics

**Generalization**
- Models trained against RandBot show positive transfer to other opponents
- Cross-training against multiple opponents improves robustness
- Partial observability handled through implicit opponent modeling

---

## Bot Opponents

### RandBot
**Strategy**: Random action selection from legal moves  
**Purpose**: Baseline for minimum expected performance  
**Difficulty**: 1/5

### BullyBot
**Strategy**: Aggressive rule-based play, prioritizing high-value cards  
**Purpose**: Test against deterministic greedy strategies  
**Difficulty**: 2/5

### MinimaxBot
**Strategy**: Minimax search with perfect information assumption  
**Purpose**: Evaluate against optimal single-step planning  
**Difficulty**: 3/5

### AlphaBetaBot
**Strategy**: Optimized minimax with alpha-beta pruning  
**Purpose**: Test against efficient search algorithms  
**Difficulty**: 4/5

### RdeepBot
**Strategy**: Hybrid approach combining heuristics and search  
**Purpose**: Benchmark against advanced bot design  
**Difficulty**: 5/5

---

## Testing & Evaluation

### Unit Tests

```bash
# Run all tests
pytest ./tests

# Run specific test module
pytest ./tests/test_game.py

# Run with coverage report
pytest --cov=src/schnapsen ./tests
```

### Statistical Validation

The project includes rigorous statistical testing:

**Paired t-test**: Parametric test for mean win rate comparison
```python
from schnapsen.statistical_test import paired_ttest
result = paired_ttest(model_a_scores, model_b_scores)
```

**Wilcoxon Signed-Rank Test**: Non-parametric alternative
```python
from schnapsen.statistical_test import wilcoxon_test
result = wilcoxon_test(model_a_scores, model_b_scores)
```

### Evaluation Metrics

- **Win Rate**: Percentage of games won
- **Average Score**: Points accumulated per game
- **Average Game Length**: Number of tricks per game
- **Decision Time**: Average computation time per action
- **Learning Curve**: Win rate progression during training

---

## Future Work

### Short-term Improvements

- **Advanced Architectures**: Test DQN variants (Double DQN, Dueling DQN, Rainbow)
- **Hyperparameter Optimization**: Automated tuning using Optuna or Ray Tune
- **Curriculum Learning**: Progressive difficulty scaling from weak to strong opponents
- **Transfer Learning**: Pre-training on simplified game variants

### Long-term Research Directions

- **Multi-Agent RL**: Self-play training for emergent strategies
- **Monte Carlo Tree Search**: Hybrid MCTS-RL approach
- **Explainable AI**: Interpretable decision-making analysis
- **Human Study**: User experience evaluation with human players
- **Generalization**: Adaptation to other trick-taking card games

### Technical Enhancements

- Model checkpointing and versioning
- Distributed training support
- Real-time visualization of Q-value landscapes
- Web-based training dashboard

---

## Acknowledgments

This project was developed as part of the **Project Intelligent Systems** course at **Vrije Universiteit Amsterdam**. We thank the course instructors for their guidance and feedback throughout the development process.

Special thanks to the open-source community for tools and libraries that made this project possible, including PyTorch, NumPy, and Matplotlib.

---

## License

This project is distributed for **educational and research purposes**.

See [LICENSE](LICENSE) for details.

---

## Contact

For questions, suggestions or collaboration opportunities, please reach out to the authors.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*
