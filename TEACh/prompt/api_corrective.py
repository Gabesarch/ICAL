class AgentCorrective:
    '''
    This class represents agent corrective actions that can be taken to fix a subgoal error
    Example usage:
    agent = AgentCorrective()
    agent.move_back()
    '''

    def move_back(self):
        """Step backwards away from the object

        Useful when the object is too close for the agent to interact with it
        """
        pass

    def move_closer(self):
        """Step forward to towards the object to get closer to it

        Useful when the object is too far for the agent to interact with it
        """
        pass

    def move_alternate_viewpoint(self):
        """Move to an alternate viewpoint to look at the object

        Useful when the object is occluded or an interaction is failing due to collision or occlusion.
        """
        pass

