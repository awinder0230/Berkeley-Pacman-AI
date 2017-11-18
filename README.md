# Berkeley Pacman AI Project

Class project of National Taiwan University's "Introduction to Artificial Intelligence and Machine Learning" course, EE 4061, 2017 Full. This project is originally the class material of UC Berkeley's course, CS 188.

## 1. Search in Pacman

In this project, Pacman agent aims to find paths through the maze, as well as collecting food along the way. The followings are the algorithms used in this project:

- Depth First Search
- Breadth First Search
- Uniform Cost Search
- A* Search

Also, this project includes problem formulation and heuristic design. 

## 2. Multi-Agent Pacman

In this project, Pacman tries to eat all the food without being attacked by ghosts. The core algorithms of this project involve the following techniques from adversarial searching problem:

- Minimax Search
- Alpha-Beta Pruning
- Expectimax Search
- Evaluation Function Design

## 3. Tracking

Different from problem 1 and 2, Pacman is now able to hunt for ghosts in this project. However, Pacman could only track ghosts by their banging and clanging, rather than knowing the ghosts' exact positions. 

We may solve this problem by constructing a Bayesian Model, and calculate the probability distribution of ghosts' positions by the following methods:

- Exact Inference
- Approximate Inference: Particle Filtering
- Joint Particle Filtering (for multi-ghosts cases)
  

