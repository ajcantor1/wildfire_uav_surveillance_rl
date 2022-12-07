from collections import namedtuple
Transition = namedtuple('Transition',('belief_map', 'state_vector', 'action', 'next_belief_map', 'next_state_vector', 'reward'))