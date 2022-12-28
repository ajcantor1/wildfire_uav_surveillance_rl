import gym


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for action_space in agents_action_space:
            assert isinstance(action_space, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):

        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]