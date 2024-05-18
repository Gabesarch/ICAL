class AgentCorrective:
    '''
    This class represents agent corrective actions that can be taken to fix a subgoal error
    Example usage:
    agent = AgentCorrective()
    agent.move_back()
    '''
    def __init__(self, agent):
        self.agent = agent

    def move_back(self):
        """Step backwards away from the object

        Useful when the object is too close for the agent to interact with it
        """
        self.agent.navigation.step_back(vis=self.agent.vis, text=f"Stepping back", object_tracker=self.agent.object_tracker)

    def move_closer(self):
        """Step forward to towards the object to get closer to it

        Useful when the object is too far for the agent to interact with it
        """
        if self.agent.obj_center_camX0 is not None:
            map_pos = self.agent.navigation.get_map_pos_from_aithor_pos(self.agent.obj_center_camX0)
            ind_i, ind_j  = self.agent.navigation.get_interaction_reachable_map_pos(map_pos, location_quandrant='third')
            self.agent.navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.agent.dist_thresh)
            self.agent.navigation.navigate_to_point_goal(vis=self.agent.vis, text=f"Navigate closer", object_tracker=self.agent.object_tracker, max_fail=5, add_obs=True)
            self.agent.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.agent.dist_thresh)
            self.agent.navigation.orient_camera_to_point(self.agent.obj_center_camX0, vis=self.agent.vis, text=f"Orient to object", object_tracker=self.agent.object_tracker) 
        else:
            self.agent.navigation.take_action("MoveAhead")

    def move_alternate_viewpoint(self):
        """Move to an alternate viewpoint to look at the object

        Useful when the object is occluded or an interaction is failing due to collision or occlusion.
        """
        if self.agent.obj_center_camX0 is not None:
            map_pos = self.agent.navigation.get_map_pos_from_aithor_pos(self.agent.obj_center_camX0)
            ind_i, ind_j  = self.agent.navigation.get_interaction_reachable_map_pos(map_pos, location_quandrant='third')
            self.agent.navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.agent.dist_thresh)
            self.agent.navigation.navigate_to_point_goal(vis=self.agent.vis, text=f"Navigate to alternate", object_tracker=self.agent.object_tracker, max_fail=5, add_obs=True)
            self.agent.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.agent.dist_thresh)
            self.agent.navigation.orient_camera_to_point(self.agent.obj_center_camX0, vis=self.agent.vis, text=f"Orient to object", object_tracker=self.agent.object_tracker)
        else:
            self.agent.navigation.take_action("RotateRight")
            self.agent.navigation.take_action("MoveAhead")
            self.agent.navigation.take_action("RotateLeft")

