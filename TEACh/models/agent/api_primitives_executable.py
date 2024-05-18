from arguments import args
import ipdb
st = ipdb.set_trace
import copy
from deepdiff import DeepDiff

class InteractionObject:
    """
    Represents an expression that uniquely identifies an object in the house.

    Attributes:
        object_class (str): Object category (e.g., "Mug", "Apple").
        landmark (str): Optional. Landmark object category in relation to the interaction object (e.g., "apple is on the countertop").
        object_instance (dict): Optional. Represents the object instance from the current scene state. 
    """

    def __init__(
        self, 
        agent, 
        object_class: str, landmark: str = None, 
        object_instance: str = None, 
        parent_object: str = None,
        grounding_phrase: str = None, 
        ):
        """
        Initialize InteractionObject.

        Args:

        agent (class SubGoalController): main agent controller class
        object_class (str): Category of the interaction object.
        landmark (str, optional): Landmark object category.
        object_instance (dict, optional): Specific instance of the interaction object in the current scene state. 
                                         If not provided, the agent will search for the object based on the 
                                         provided object_class and landmark inputs.
        parent_object (str, optional): Parent object that this object instance originated from. For example, if a tomato is sliced, new TomatoSliced instances are created that originated from the tomato instance.
        grounding_phrase (dict, optional): grounding phrase that references the object to help find the object if no object_instance is given.
        """
        self.agent = agent
        if object_class in self.agent.name_to_mapped_name.keys():
            object_class = self.agent.name_to_mapped_name[object_class]
        self.object_class = object_class
        self.parent_object = parent_object
        self.landmark = landmark
        self.object_instance = object_instance
        self.check_success_change_state = args.check_success_change_state
        # self.emptied = False
        # self.attribute_dict = None
        if self.object_instance is None:
            self.get_object_data(object_class, grounding_phrase, parent_object)
        else:
            try:
                self.object_id = int(object_instance.split('_')[-1])
                object_class_ = object_instance.split('_')[0]
            except:
                self.object_id = None
                object_class_ = None
            if (self.object_id not in self.agent.object_tracker.objects_track_dict.keys() 
                    or self.agent.object_tracker.objects_track_dict[self.object_id]["label"]!=object_class_):
                self.get_object_data(object_class, grounding_phrase, parent_object)
        if self.object_id is not None:
            self.object_class = self.agent.object_tracker.objects_track_dict[self.object_id]["label"]
            self.object_instance = f'{self.object_class}_{self.object_id}'
            # self.attribute_dict = self.agent.object_tracker.objects_track_dict[self.object_id]
        self.attributes_check = {
            "label",
            "holding",
            "sliced",
            "dirty",
            "cooked",
            "filled",
            "fillLiquid",
            "toggled",
            "open",
            }

        # This is now handled in place() 
        # if self.object_class in self.agent.EMPTYABLE and self.object_id is not None:
        #     # Pre-condition check to empty object if it is full
        #     supporting_objects = self.get_attribute("supporting")
        #     if supporting_objects is not None and len(supporting_objects)>0:
        #         self.empty()

    def get_object_data(self, object_class, grounding_phrase, parent_object):
        # search for object
        _, _, obj_ID = self.agent.get_object_data(object_class, grounding_phrase, parent_object)
        if obj_ID is None:
            self.object_id = None
        else:
            self.object_id = int(obj_ID)
            if object_class in self.agent.clean_classes and self.check_attribute("dirty", True):
                self.agent.clean(self.object_id, object_class)
                self.change_state("dirty", False)
        self.agent.state_before_execution = copy.deepcopy(self.agent.object_tracker.objects_track_dict)
        self.agent.metadata_before_execution = copy.deepcopy(self.agent.controller.last_event.metadata)

    def change_state(self, attribute: str, value) -> None:
        """
        Update and log the state of a specified attribute in self.object_instance for the interaction object.

        This function modifies the recorded state of the object_instance by setting the provided 
        attribute to the given value. It is used primarily for internal tracking of the 
        object's state. No external actions or reactions are triggered by this change.

        Args:
        attribute (str): Attribute to be changed.
        value : New value for the attribute.

        Note:
        Ensure attribute exists and value is valid before using.
        """
        if self.object_id is None:
            return
        if args.online_skill_learning and attribute not in ["holding", "supported_by", "supporting", "toggled", "open", "sliced"]:
            if self.check_attribute(attribute, value):
                if attribute in ["filled", "fillLiquid"]:
                    self.agent.err_message = f"Tried to change the state of object {self.object_instance} for attribute {attribute} to value {value}, but object attribute {attribute} already takes on this value for this object. If the object is already filled with liquid it should be poured out in the sink using the pour() primitive."
                    self.agent.help_message = ''
                else:
                    self.agent.err_message = f"Tried to change the state of object {self.object_instance} for attribute {attribute} to value {value}, but object attribute {attribute} already takes on this value for this object. The code should be altered to make sure steps are consistent with the state of the environment at the time of execution."
                    self.agent.help_message = ''
                print(self.agent.err_message)
                raise CustomError(self.agent.err_message)
        print(f"Changing attribute {attribute} to value {value} for object {self.object_class}, id={self.object_id}")
        # self.attribute_dict[attribute] = value
        if self.check_success_change_state and not self.agent.step_success:
            return
        # if attribute=="cooked":
        #     st()
        self.agent.object_tracker.objects_track_dict[self.object_id][attribute] = value
        if args.use_gt_attributes and args.online_skill_learning:
            self.agent.track_dict_before = copy.deepcopy(self.agent.object_tracker.objects_track_dict)

    def check_attribute(self, attribute: str, value) -> bool:
        """
        Checks if the InteractionObject's object_instance has an attribute with a specified value.

        Args:
        attribute (str): Attribute to check.
        value: Value to compare against the object attribute.

        Returns:
        bool: True if attribute exists and matches the provided value, otherwise False.
        """
        if self.object_id is None:
            return False
        check_dict = self.agent.object_tracker.objects_track_dict[self.object_id] #self.attribute_dict
        if attribute not in check_dict.keys():
            if args.online_skill_learning:
                self.agent.err_message = f"Attribute {attribute} is not a valid attribute. The following are valid attributes: {list(self.attributes_check)}. Remove this attribute check from the code to avoid this error."
                self.agent.help_message = ''
                print(self.agent.err_message)
                raise CustomError(self.agent.err_message)
            return True
        return True if check_dict[attribute]==value else False

    def get_attribute(self, attribute: str):
        """
        Get the desired attribute of InteractionObject.

        Args:
        attribute (str): Attribute to check.

        Returns:
        attribute if it exists, else None
        """
        if self.object_id is None:
            return None
        check_dict = self.agent.object_tracker.objects_track_dict[self.object_id]
        if attribute in check_dict.keys():
            return check_dict[attribute]
        else:
            return None

    def pickup(self) -> None:
        """
        Assumes object is in view and picks it up.
        
        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.check_attribute("holding", True):
            return True
        if self.object_class not in self.agent.PICKUPABLE_OBJECTS:
            self.agent.step_success = False
            return False
        if self.agent.object_tracker.get_ID_of_holding() is not None:
            # pre-condition check: put down the object in hand before picking up a new one
            # if args.online_skill_learning and args.error_on_action_fail:
            #     self.agent.err_message = f"The agent tried to pick up the {self.object_class} when the agent was already holding {self.agent.object_tracker.get_label_of_holding()}. The code should be altered to only pick up one object at a time. Or use the put_down() function to put down the object in hand before picking up a new one."
            #     self.agent.help_message = ''
            #     print(self.agent.err_message)
            #     raise CustomError(self.agent.err_message)
            self.put_down()
            self.go_to()
        # if self.get_attribute("supported_by") is not None:
        #     class_supported = self.get_attribute("supported_by").split('_')[0]
        #     if class_supported in self.agent.OPENABLE_OBJECTS:
        #         self.get_attribute("supported_by").split('_')[0]
        if not self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: make sure looking at object
            self.go_to()
        return self.execute_manipulation("Pickup")

    def place(self, landmark_name) -> None:
        """
        Places the interaction object on the given landmark.

        Args:
        landmark_name (InteractionObject): Landmark to place the object on.

        Note:
        Assumes robot has an object and the landmark is in view.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None or landmark_name.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.PICKUPABLE_OBJECTS:
            self.agent.step_success = False
            return False
        if landmark_name.object_class not in self.agent.RECEPTACLE_OBJECTS:
            self.agent.step_success = False
            return False
        if self.agent.object_tracker.get_ID_of_holding() is None or self.agent.object_tracker.get_ID_of_holding()!=self.object_id:
            # pre-condition check: pick up before placing
            # if args.online_skill_learning and args.error_on_action_fail:
            #     self.agent.err_message = f"The agent tried to place the {self.object_class} on {landmark_name} without picking up the {self.object_class} first. The code should be altered to pick up the {self.object_class} first before trying to place it."
            #     self.agent.help_message = ''
            #     print(self.agent.err_message)
            #     raise CustomError(self.agent.err_message)
            self.pickup()
        if landmark_name.check_attribute("emptied", False) and landmark_name.object_class in self.agent.EMPTYABLE and landmark_name.object_id is not None:
            landmark_name.empty()
            landmark_name.change_state("emptied", True)
        if not self.agent.navigate_obj_info["obj_ID"]==landmark_name.object_id:
            # pre-condition: make sure looking at object
            landmark_name.go_to()
        if landmark_name.object_class in self.agent.OPENABLE_OBJECTS:
            if landmark_name.check_attribute("open", False):
                landmark_name.open()

        success_place = self.execute_manipulation("Place", landmark_name)
        if success_place:
            # if 
            #     obj_metas = {}
            #     for obj in self.agenwt.controller.last_event.metadata['objects']:
            #         if obj['']
            #         obj_metas[obj['objectId']] = 
            # if 
            self.change_state("supported_by", [landmark_name.object_instance])
            # self.change_state("supported_by", [landmark_name.object_class])

        return success_place

    def slice(self) -> None:
        """
        Slices the object into pieces, assuming the agent is holding a knife and is at the object's location.
        
        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.SLICEABLE:
            self.agent.step_success = False
            return False
        if self.check_attribute("sliced", True):
            # pre-condition check: already sliced?
            return True
        if not self.agent.object_tracker.get_label_of_holding() in ["Knife", "ButterKnife"]:
            # pre-condition: make sure holding knife
            self.agent.pickup_category("Knife")
        if not self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: make sure looking at object
            self.go_to()
        return self.execute_manipulation("Slice")

    def toggle_on(self) -> None:
        """
        Toggles on the interaction object.

        Note:
        Assumes object is off and agent is at its location.
        Objects like lamps, stoves, microwaves can be toggled on.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.TOGGLEABLE:
            self.agent.step_success = False
            return False
        if self.check_attribute("toggled", True):
            # pre-condition check: already toggled on?
            return True
        if not self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: make sure looking at object
            self.go_to()
        if self.object_class in self.agent.OPENABLE_OBJECTS:
            if self.check_attribute("open", True):
                self.close()
        return self.execute_manipulation("ToggleOn")

    def toggle_off(self) -> None:
        """
        Toggles off the interaction object.

        Note:
        Assumes object is on and agent is at its location.
        Objects like lamps, stoves, microwaves can be toggled off.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.TOGGLEABLE:
            self.agent.step_success = False
            return False
        if self.check_attribute("toggled", False):
            # pre-condition check, already toggled off?
            return True
        if not self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: make sure looking at object
            self.go_to()
        return self.execute_manipulation("ToggleOff")

    def go_to(self) -> None:
        """Navigates to the object.
        
        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            # # pre-condition: object could not be found
            # if args.online_skill_learning and args.error_on_action_fail:
            #     self.agent.err_message = f"Even after searching, object {self.object_class} could not be found in the current environment. A different object category other than {self.object_class} should be used in InteractionObject."
            #     self.agent.help_message = 'None'
            #     print(self.agent.err_message)
            #     # raise CustomError(self.agent.err_message)
            # if args.error_on_action_fail:
            #     self.agent.err_message = f"Even after searching, object {self.object_class} could not be found in the current environment. A different object category other than {self.object_class} should be used in InteractionObject."
            #     self.agent.help_message = ''
            #     print(self.agent.err_message)
            #     raise CustomError(self.agent.err_message)
            return False
        if self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: already just navigated there?
            return True
        success = self.agent.navigate(obj_ID=self.object_id)
        return success

    def open(self) -> None:
        """
        Opens the interaction object.

        Note:
        Assumes object is closed and agent is at its location.
        Objects like fridges, cabinets, drawers can be opened.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.OPENABLE_OBJECTS:
            self.agent.step_success = False
            return False
        if self.check_attribute("open", True):
            # pre-condition check: already open?
            return True
        if not self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: make sure looking at object
            self.go_to()
        if self.check_attribute("toggled", True):
            self.toggle_off()
        return self.execute_manipulation("Open")

    def close(self) -> None:
        """
        Closes the interaction object.

        Note:
        Assumes object is open and agent is at its location.
        Objects like fridges, cabinets, drawers can be closed.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.OPENABLE_OBJECTS:
            self.agent.step_success = False
            return False
        if self.check_attribute("open", False):
            # pre-condition check: already closed?
            return True
        if not self.agent.navigate_obj_info["obj_ID"]==self.object_id:
            # pre-condition: make sure looking at object
            self.go_to()
        return self.execute_manipulation("Close")

    def put_down(self) -> None:
        """
        Puts down the object in hand on a nearby receptacle.

        Note:
        Assumes object is in hand and agent wants to free its hand.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.PICKUPABLE_OBJECTS:
            self.agent.step_success = False
            return False
        id_holding = self.agent.object_tracker.get_ID_of_holding()
        if id_holding is None:
            return True
        success = self.agent.put_down()
        if success:
            self.agent.object_tracker.objects_track_dict[id_holding]["holding"] = False
            if args.use_gt_attributes and args.online_skill_learning:
                self.agent.track_dict_before = copy.deepcopy(self.agent.object_tracker.objects_track_dict)
        return success

    def pour(self, landmark_name) -> None:
        """
        Pours the contents of the interaction object into the specified landmark.

        Args:
        landmark_name (InteractionObject): Landmark to pour contents into.

        Note:
        Assumes object is picked up, filled with liquid.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.object_class not in self.agent.FILLABLE_CLASSES:
            self.agent.step_success = False
            return False
        if landmark_name.object_class not in (self.agent.FILLABLE_CLASSES + ["SinkBasin", "BathtubBasin"]):
            self.agent.step_success = False
            return False
        if self.check_attribute("filled", False):
            # pre-condition check: object must be filled to pour
            # return True
            if args.online_skill_learning and args.error_on_action_fail:
                self.agent.err_message = f"Object must be filled with liquid in order to pour it. Depending on the task, the object should either be checked for if it is filled before trying to pour it or the object should be filled up with a liquid before attempting to pour it."
                self.agent.help_message = 'None'
                print(self.agent.err_message)
                raise CustomError(self.agent.err_message)
        if self.check_attribute("holding", False):
            # pre-condition: make sure object is in hand
            self.go_to()
            self.pickup()
        if not self.agent.navigate_obj_info["obj_ID"]==landmark_name.object_id:
            # pre-condition: make sure looking at landmark object
            landmark_name.go_to()
        return self.execute_manipulation("Pour", landmark_name)

    def empty(self):
        """Empty the object of any other objects on/in it to clear it out. 

        Useful when the object is too full to place an object inside it.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        success = self.agent.empty(obj_ID=self.object_id)
        if args.use_gt_attributes and args.online_skill_learning:
            self.agent.track_dict_before = copy.deepcopy(self.agent.object_tracker.objects_track_dict)
        return success

    def clean(self):
        """wash the interaction object to clean it in the sink.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            return False

    # def cook(self):
    #     """Empty the object of any other objects on/in it to clear it out. 

    #     Useful when the object is too full to place an object inside it.

    #     Returns:
    #     bool: True if action successful, otherwise False.
    #     """
    #     if self.object_id is None:
    #         return False
    #     return self.agent.empty(obj_ID=self.object_id)

    def pickup_and_place(self, landmark_name) -> None:
        """
        Picks up the interaction object and places it on the specified landmark.

        Args:
        landmark_name (InteractionObject): Landmark to place the object on.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            self.agent.step_success = False
            return False
        if self.check_attribute("holding", False):
            self.go_to()
            success = self.pickup()
            if success:
                self.change_state("holding", True)
        landmark_name.go_to()
        success = self.place(landmark_name)
        if success:
            self.change_state("holding", False)
        return success

    def check_state_matching(self):
        pass

    def execute_manipulation(self, subgoal_name, landmark_name=None, check_success=True) -> None:
        """
        Executes manipulation action subgoal_name

        Args:
        subgoal_name (string): action name to execute.
        landmark_name (InteractionObject) (optional): Landmark to place or pour the object on.

        Returns:
        bool: True if action successful, otherwise False.
        """
        if self.object_id is None:
            return False

        if args.error_on_action_fail and self.check_attribute("sliced", True) and self.object_class in self.agent.SLICEABLE:
            self.agent.err_message = f'Error when performing {subgoal_name} {self.object_class}: The object {self.object_instance} has been sliced and is not usable. If an object is sliced, this will create individual slices of the object (e.g., whole {self.object_class} -> many slices of {self.object_class}Sliced). A new InteractionObject with parent_object argument set to the whole object instance should be instantiate to interact with a single slice of the sliced object. For example: {self.object_class.lower()}_slice = InteractionObject("{self.object_class}Sliced", object_instance = None, parent_object = "{self.object_instance}") # Initialize new sliced object from sliced parent'
            raise CustomError(self.agent.err_message+self.agent.help_message.replace('None', ''))

        object_class = landmark_name.object_class if landmark_name is not None else self.object_class
        
        # # attribute change check
        # if args.use_gt_attributes and args.online_skill_learning and not args.simulate_actions:
        #     self.agent.object_tracker.update_attributes_from_metadata()
        #     self.track_dict_after = copy.deepcopy(self.agent.object_tracker.objects_track_dict)
        #     difference = DeepDiff(self.agent.track_dict_before, self.track_dict_after)
        #     if len(difference)>0:
        #         if 'values_changed' in difference.keys():
        #             text = f'Moved onto code execution of {subgoal_name} {object_class}, but the following object state mismatch was detected from the previous execution of code lines:\n'
        #             count = 0
        #             for key_change in difference['values_changed']:
        #                 key_object = int(key_change.split("root[")[-1].split("]")[0])
        #                 object_label = self.agent.object_tracker.objects_track_dict[key_object]["label"]
        #                 key_changed = key_change.split("[")[-1].split("]")[0].replace("'", "")
        #                 if key_changed not in self.attributes_check:
        #                     continue
        #                 if key_changed in ["holding", "supported_by", "supporting", "toggled", "open"]:
        #                     continue
        #                 count += 1
        #                 if key_changed=="dirty" and difference["values_changed"][key_change]["new_value"]==True:
        #                     text_to_add = f'{count}. Object {object_label}_{key_object} was changed to attribute "dirty" being False, but the object is still "dirty" = True in actuality. This means the object was not cleaned properly before the object state was changed in the code. The object needs to be placed in the sink basin, and the faucet toggled on to clean the object, and only then should the change_state("dirty", False) be called to change the state of the object. Please verify that the code follows these steps to clean the object.'
        #                 else:
        #                     text_to_add = f'{count}. Object {object_label}_{key_object} changed attribute state in the environment for "{key_changed}" from {difference["values_changed"][key_change]["old_value"]} to {difference["values_changed"][key_change]["new_value"]} before execution of {subgoal_name} {object_class}, but this change was not carried out by calling change_state() in the function code to properly keep track of the object state. You should change the code to include change_state("{key_changed}", {difference["values_changed"][key_change]["new_value"]}) for object "{object_label}_{key_object}" before executing {subgoal_name} {object_class} to ensure the state tracker is consistent with the actual object states.\n'
        #                     if object_label in ["Potato", "PotatoSliced"] and key_changed in ["cooked"] and difference["values_changed"][key_change]["new_value"]==True:
        #                         text_to_add += f'The {object_label} was likely cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on, so this is the reason that the object was cooked. The change_state("cooked", True) should be added after 1) putting the {object_label} in a pot or pan (if its not already in a pot or pan), and 2) putting the pot or pan on a stove burner (if its not already on the stove burner), and 3) the stove burner is turned on (if its not on already).\n'
        #                     elif object_label in ["Potato", "PotatoSliced"] and key_changed in ["cooked"] and difference["values_changed"][key_change]["new_value"]==False:
        #                         # text_to_add = f"The {object_label} was likely changed to be cooked pre-maturely. {object_label}s are cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on. The change_state() should be ONLY be added after 1) putting the {object_label} in a pot or pan, and 2) putting the pot or pan on a stove burner, and 3) the stove burner is turned on. Only after these three conditions are met should the change_state be called.\n"
        #                         text_to_add = f'The {object_label} was likely changed to be cooked pre-maturely. {object_label}s are cooked when put on a boiling pot or hot pan that is on a stove burner that is turned on. The change_state() should ONLY be added after 1) putting the {object_label} in a pot or pan (if its not already in a pot or pan), and 2) putting the pot or pan on a stove burner (if its not already on the stove burner), and 3) the stove burner is turned on (if its not on already).\n'
        #                 text += text_to_add
        #             if count>0:
        #                 self.agent.err_message = text
        #                 self.agent.help_message = ''
        #                 raise CustomError(text)
        
        retry_image = True if (subgoal_name in self.agent.RETRY_ACTIONS_IMAGE and object_class in self.agent.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.agent.RETRY_ACTIONS_LOCATION and object_class in self.agent.RETRY_DICT_LOCATION[subgoal_name]) else False
        success, error = self.agent.execute_action(
            subgoal_name, 
            object_class, 
            object_done_holding=False, 
            retry_image=retry_image, 
            retry_location=retry_location
            )   
        # if args.online_skill_learning and check_success:
        #     if not success:
        #         # plt.figure()
        #         # plt.imshow(self.agent.controller.last_event.frame)
        #         # plt.savefig('output/test.png')
        #         self.format_feedback(self.agent.err_message+self.agent.help_message.replace('None', ''), subgoal_name, object_class)
        #         raise CustomError(self.agent.err_message+self.agent.help_message.replace('None', ''))
        if args.error_on_action_fail and check_success:
            if not success:
                self.format_feedback(self.agent.err_message+self.agent.help_message.replace('None', ''), subgoal_name, object_class)
                self.agent.err_message = f'Error when performing {subgoal_name} {object_class}: {self.agent.err_message}'
                raise CustomError(self.agent.err_message+self.agent.help_message.replace('None', ''))
        if args.use_gt_attributes:
            self.agent.track_dict_before = copy.deepcopy(self.agent.object_tracker.objects_track_dict)
        return success

    def format_feedback(self, feedback, subgoal_name, object_class):
        msg = self.agent.err_message
        if 'This target object is NOT a receptacle!' in feedback and subgoal_name in ["Place"]:
            if object_class not in self.agent.RECEPTACLE_OBJECTS:
                self.agent.err_message = f'The target object category is not a valid receptacle to place the object on. The following are valid receptacles: {self.agent.RECEPTACLE_OBJECTS}.'
                self.agent.help_message = ''
            else:
                if object_class in ["Toaster"] and subgoal_name in ["Place"]:
                    self.agent.err_message = f"The Toaster is full right now. Only one slice of bread can be toasted at a time. Remove any bread slices currently in the toaster (using the pickup() followed by put_down() functions with the bread slices in the toaster currently) before trying to place an additional bread slice in the toaster."
                    self.agent.help_message = ''
                else:
                    self.agent.err_message = f'The receptacle {object_class} may be too filled to place the target on it. The receptacle should be checked for if it is filled, and if it filled, emptied using the empty() primitive.'
                    self.agent.help_message = ''
        # if "CanPickup to be" in msg:
        #     return 'Object "%s" can\'t be picked up.' % msg.split()[0].split("|")[0]
        # # Example: "Object ID appears to be invalid." # noqa: E800
        elif ("Object ID" in msg and "invalid" in msg) or "Could not retrieve object" in msg:
            self.agent.err_message = "No error. Nothing to change in the code."
            self.agent.help_message = ''
        # Example "Can't place an object if Agent isn't holding anything # noqa: E800
        # if "if Agent isn't holding" in msg:
        #     return "Must be holding an object first."
        # # Example: "Slice: ObjectInteraction only supported for held object Knife" # noqa: E800
        # if "Slice: ObjectInteraction" in msg:
        #     return "Must be holding a knife."
        # # Example: "object is not toggleable" # noqa: E800
        elif "Object is already toggled on." in msg:
            self.agent.err_message = f'The {object_class} is already toggled on. The object should first be checked for if it is already has attribute value True for "toggled" before trying to toggle it on.'
            self.agent.help_message = ''
        elif "Object is already toggled off." in msg:
            self.agent.err_message = f'The {object_class} is already toggled off. The object should first be checked for if it is already has attribute value False for "toggled" before trying to toggle it off.'
            self.agent.help_message = ''
        elif "only supported for held object Knife" in msg:
            self.agent.err_message = f"Action {subgoal_name} is only support when holding a knife. The agent must pick up a knife first before trying to perform {subgoal_name}."
            self.agent.help_message = ''
        elif "is full right now" in msg:
            if object_class in ["Toaster"] and subgoal_name in ["Place"]:
                self.agent.err_message = f"The Toaster is full right now. Only one slice of bread can be toasted at a time. Remove any bread slices currently in the toaster (using the pickup() followed by put_down() functions) before trying to place an additional bread slice in the toaster."
                self.agent.help_message = ''
            else:
                self.agent.err_message = f"Object {object_class} is full right now. Empty the {object_class} before interacting with it."
                self.agent.help_message = ''
        elif "not toggleable" in msg:
            self.agent.err_message = f"Object {object_class} cannot be turned on or off."
            self.agent.help_message = ''
        # Example: "can't toggle object off if it's already off!" # noqa: E800
        elif "toggle object off if" in msg:
            self.agent.err_message = f"Object {object_class} is already turned off."
            self.agent.help_message = ''
        # Example: "can't toggle object on if it's already on!" # noqa: E800
        elif "toggle object on if" in msg:
            self.agent.err_message = f"Object {object_class} is already turned on."
            self.agent.help_message = ''
        # Example: "CounterTop|-00.08|+01.15|00.00 is not an Openable object" # noqa: E800
        elif "is not an Openable object" in msg:
            self.agent.err_message = f'Object {object_class} can\'t be opened.'
            self.agent.help_message = ''
        # Example: "CounterTop_d7cc8dfe Does not have the CanBeSliced property!" # noqa: E800
        elif "Does not have the CanBeSliced" in msg:
            self.agent.err_message = f"Object {object_class} cannot be sliced."
            self.agent.help_message = ''
        # Example: "Object failed to open/close successfully." # noqa: E800
        elif "failed to open/close" in msg:
            self.agent.err_message = "No error. Nothing to change in the code."
            self.agent.help_message = ''
        # Example: "StandardIslandHeight is blocking Agent 0 from moving 0" # noqa: E800
        elif "is blocking" in msg:
            self.agent.err_message = "No error. Nothing to change in the code."
            self.agent.help_message = ''
        # Example: "a held item: Book_3d15d052 with something if agent rotates Right 90 degrees" # noqa: E800
        elif "a held item" in msg and "if agent rotates" in msg:
            self.agent.err_message = "No error. Nothing to change in the code."
            self.agent.help_message = ''
        # Example: "No valid positions to place object found" # noqa: E800
        elif "No valid positions to place" in msg:
            if object_class in ["Toaster"] and subgoal_name in ["Place"]:
                self.agent.err_message = f"The Toaster is full right now. Only one slice of bread can be toasted at a time. Remove any bread slices currently in the toaster (using the pickup() followed by put_down() functions with the bread slices in the toaster currently) before trying to place an additional bread slice in the toaster."
                self.agent.help_message = ''
            else:
                self.agent.err_message = f'The receptacle {object_class} may be too filled to place the target on it. The receptacle should be checked for if it is filled, and if it filled, emptied using the empty() primitive.'
                self.agent.help_message = ''
        # Example: "This target object is NOT a receptacle!" # noqa: E800
        elif "NOT a receptacle" in msg:
            if object_class not in self.agent.RECEPTACLE_OBJECTS:
                self.agent.err_message = f'The target object category is not a valid receptacle to place the object on. The following are valid receptacles: {self.agent.RECEPTACLE_OBJECTS}.'
                self.agent.help_message = ''
            else:
                if object_class in ["Toaster"] and subgoal_name in ["Place"]:
                    self.agent.err_message = f"The Toaster is full right now. Only one slice of bread can be toasted at a time. Remove any bread slices currently in the toaster (using the pickup() followed by put_down() functions with the bread slices in the toaster currently) before trying to place an additional bread slice in the toaster."
                    self.agent.help_message = ''
                else:
                    self.agent.err_message = f'The receptacle {object_class} may be too filled to place the target on it. The receptacle should be checked for if it is filled, and if it filled, emptied using the empty() primitive.'
                    self.agent.help_message = ''
        # Example: "Target must be OFF to open!" # noqa: E800
        # if "OFF to open!" in msg:
        #     return "Object must be turned off before it can be opened."
        # Example: "cracked_egg_5(Clone) is not a valid Object Type to be placed in StoveBurner_58b674c4" # noqa: E800
        # if "not a valid Object Type to be placed" in msg:
        #     return "Held object cannot be placed there."
        # Example: "No target found" # noqa: E800
        elif "No target found" in msg:
            # if args.error_on_action_fail:
            #     # if object_class in self.SLICEABLE:
            #     #     centroids_target, labels = self.agent.object_tracker.get_centroids_and_labels(return_ids=False, object_cat=object_class, return_sliced=True)
            #     #     if len()
            #     self.agent.err_message = f"Could not find target object {object_class}. Try using a different object as input."
            #     self.agent.help_message = ''
            # else:
            self.agent.err_message = "No error. Nothing to change in the code."
            self.agent.help_message = ''
        # Example: "Knife|-01.70|+01.71|+04.01 is not interactable and (perhaps it is occluded by something)." # noqa: E800
        elif "it is occluded by something" in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Another object is blocking the agent from performing {object_class} {subgoal_name}. Move to a different viewpoint or move back."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''
        # "Could not find a target object at the specified location" # noqa: E800
        elif "Could not find a target object" in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Could not find target object {object_class}. Try using a different object as input."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''
        # "another object's collision is blocking held object from being placed" # noqa: E800
        elif "another object's collision is blocking" in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Another object is blocking the agent from performing {object_class} {subgoal_name}. Move to a different viewpoint or move back."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''
        # "CounterTop|+00.69|+00.95|-02.48 is too far away to be interacted with" # noqa: E800
        elif "too far away to" in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Too far to perform {object_class} {subgoal_name}. Move closer."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''
        # "Your partner is too far away for a handoff." # noqa: E800
        elif "too far away for" in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Too far to perform {object_class} {subgoal_name}. Move closer."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''
        elif 'Object is too far from agent' in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Too far to perform {object_class} {subgoal_name}. Move closer."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''
        # "Place: ObjectInteraction only supported when holding an object" # noqa: E800
        # if "only supported when holding" in msg:
        #     return "You are not holding an object."
        # "Picking up object would cause it to collide and clip into something!" # noqa: E800
        elif "would cause it to collide and" in msg:
            if args.error_on_action_fail:
                self.agent.err_message = f"Another object is blocking the agent from performing {object_class} {subgoal_name}. Move to a different viewpoint or move back."
                self.agent.help_message = ''
            else:
                self.agent.err_message = "No error. Nothing to change in the code."
                self.agent.help_message = ''

class CustomError(Exception):
    pass