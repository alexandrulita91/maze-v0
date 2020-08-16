# Maze game
Maze game is a video game genre description first used by journalists during the 1980s to describe any game in which the entire playing field is a maze.

## Requirements
- [Python 3.6 or 3.7](https://www.python.org/downloads/release/python-360/)
- [Pipenv](https://pypi.org/project/pipenv/)

## How to install the packages
You can install the required Python packages using the following command:
- `pipenv sync`

## Q-learning
Q-learning is an off policy reinforcement learning algorithm that seeks to find the best action to take given the current state. It's considered off-policy because the q-learning function learns from actions that are outside the current policy, like taking random actions, and therefore a policy isn't needed

## How to train the agent
You can train the agent using the following command:
- `pipenv run python q_maze_game.py`

## Improvement ideas
- improve the code quality
- remove unnecessary comments
