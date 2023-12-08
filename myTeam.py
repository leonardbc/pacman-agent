# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint

"""
LEO:

My strategy is based on 5 main points:

1. Offensive agent has to run away from enemy when he sees it (except when the poser capsule is active, in such case it ignores it). 
2. Offensive agent has to avoid being trapped in some key positions.
3. Offensive agent has to come back for each 4 dots eaten so we avoid losing points (even when the enemy's power capsule is active, after experimentation I found this approach more effective).
4. When the offensive agent comes back, if it sees an enemy in our zone, he should attack it, before returning to the offensive zone (except if our power capsule is active).

5. Defensive agent should run away from enemy pacman if the power capsule is eaten in our zone (for the duration of the power capsule).
6. When not seeing enemies, defensive agent should rotate around specific positions in the map.
"""


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """
    offensive_capsule_moves = 40 # Use to count duration of their capsule
    defensive_capsule_moves = 40 # Use to count duration of our capsule

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # Checking if a capsule was activated
        offensive_capsule = len(super().get_capsules(game_state))
        defensive_capsule = len(super().get_capsules_you_are_defending(game_state))

        # Capsules time control
        if not offensive_capsule and self.offensive_capsule_moves > 0:
            self.offensive_capsule_moves -= 1

        if not defensive_capsule and self.defensive_capsule_moves > 0:
            self.defensive_capsule_moves -= 1

        values = [self.evaluate(game_state, a) for a in actions]
        best_actions = [a for a, v in sorted(zip(actions, values), key=lambda x: x[1], reverse=True)]
        if best_actions[0] == 'Stop':
            return best_actions[1]
        else:
            return best_actions[0]

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    go_back = [16, 12, 8, 4, 2] # When the pacman eats these amounts of dots it should go back, except for when the power capsule is active.
    turn = 0 # This variable is used to control the go_back list
    restart_positions = [(15,4),(15,7),(15, 12)] # Positions where the offence agent will return in base

    # Avoiding getting stuck variables
    death_positions = [(21,4),(23,12), (28,9)] # These are positions the pacman can get trapped
    avoiding_position = (0,0) # Current avoinding position
    avoid_moves = 0 # amount of avoid moves the pacman will make once he gets to any of the death_positions

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor) # checks if there is food in the next successor

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes distance to defenders we can see and runs away from them
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        offensive_capsule = len(super().get_capsules(game_state))
        deffensive_capsule = len(super().get_capsules_you_are_defending(game_state))
        if len(defenders) > 0:
            if not offensive_capsule and self.offensive_capsule_moves > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
                features['defender_distance'] = -min(dists)
            else:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
                features['defender_distance'] = min(dists)

        # Avoid position
        if self.avoid_moves > 0:
            dists = self.get_maze_distance(my_pos, self.avoiding_position)
            features['avoid_position'] = dists
            self.avoid_moves -= 1

        # Offense should help defense when close to base
        my_pos = game_state.get_agent_position(self.index)
        dist = self.get_maze_distance((1,7), my_pos)
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if dist < 40 and len(invaders) > 0:
            # Except if our capsule is active
            if not deffensive_capsule and self.defensive_capsule_moves > 0:
                features['help_defence'] = 0
            else:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['help_defence'] = min(dists)
        else: 
            features['help_defence'] = 0

        # Compute distance to the nearest food
        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'defender_distance': 750, 'avoid_position': 1000, 'help_defence': -5}
    
    def return_to_base(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        #dist = self.get_maze_distance(self.start, my_pos)
        if my_pos in self.restart_positions: # At this distance the pacman has crossed to the base
            self.turn += 1
            return super().choose_action(game_state) # Normal offense behaviour
        else: # Else go back to base (specific positions)
            actions = game_state.get_legal_actions(self.index)
            best_dist = float("inf")
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist1 = self.get_maze_distance(self.restart_positions[0], pos2)
                dist2 = self.get_maze_distance(self.restart_positions[1], pos2)
                dist3 = self.get_maze_distance(self.restart_positions[2], pos2)
                dist = min(dist1,dist2,dist3)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
    
    def choose_action(self, game_state): # overwrites the main choose_action
        food = len(self.get_food(game_state).as_list()) # List of food
        my_pos = game_state.get_agent_position(self.index)

        # Avoiding positions
        if my_pos in self.death_positions:
            self.avoid_moves = 3
            self.avoiding_position = my_pos

        # Controlling amount of food and go back mechanism
        if food < self.go_back[self.turn]: # This can happen when the pacman eats a capsule
            while food < self.go_back[self.turn]:
                self.turn +=1
            
        # Go back
        if food == self.go_back[self.turn]:
            return self.return_to_base(game_state)

        # Continue with the original choose_action method
        return super().choose_action(game_state)

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    rotation = [(5,5), (6,13), (12,7)] # The defence agent will be rotating between these positions if it doesn't see a pacman
    next = 0

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        capsules = len(super().get_capsules_you_are_defending(game_state))

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Controls position rotation
        if game_state.get_agent_position(self.index) == self.rotation[self.next]:
            if self.next == 2:
                self.next = 0
            else:
                self.next += 1

        # Defense agent attacks invaders (except defense capsule is active), if it can't see invaders it rotates
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            if not capsules and self.defensive_capsule_moves > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_or_rotation'] = -min(dists)
            else:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_or_rotation'] = min(dists)
        else: 
            dists = self.get_maze_distance(my_pos, self.rotation[self.next])
            features['invader_or_rotation'] = dists
            
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_or_rotation': -10, 'stop': -100, 'reverse': -2}
