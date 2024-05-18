class InteractionObject:
    """
    Represents a unique interactable object in the house.
    """

    def __init__(self, object_class: str, object_instance: str = None, parent_object: str = None, grounding_phrase: str = None):
        """
        Initialize InteractionObject.

        Args:
        object_class (str): Category of the interaction object.
        object_instance (dict, optional): Specific instance of the interaction object in the current scene state. 
                                         If not provided, the agent will search for the object based on the 
                                         provided object_class and landmark inputs.
        parent_object (str, optional): Parent object that this object instance originated from. For example, if a tomato is sliced, new TomatoSliced instances are created that originated from the tomato instance.
        grounding_phrase (dict, optional): grounding phrase that references the object to help find the object if no object_instance is given.
        """
        self.object_class = object_class
        self.object_instance = object_instance
        self.parent_object = parent_object
        self.grounding_phrase = grounding_phrase

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
        pass

    def check_attribute(self, attribute: str, value) -> bool:
        """
        Checks if the InteractionObject's object_instance has an attribute with a specified value.

        Args:
        attribute (str): Attribute to check.
        value: Value to compare against the object attribute.

        Returns:
        bool: True if attribute exists and matches the provided value, otherwise False.
        """
        pass

    def pickup(self) -> None:
        """Assumes object is in view and picks it up.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

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
        pass

    def slice(self) -> None:
        """Slices the object into pieces, assuming the agent is holding a knife and is at the object's location.
        
        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

    def toggle_on(self) -> None:
        """
        Toggles on the interaction object.

        Note:
        Assumes object is off and agent is at its location.
        Objects like lamps, stoves, microwaves can be toggled on.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

    def toggle_off(self) -> None:
        """
        Toggles off the interaction object.

        Note:
        Assumes object is on and agent is at its location.
        Objects like lamps, stoves, microwaves can be toggled off.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

    def go_to(self) -> None:
        """Navigates to the object."""
        pass

    def open(self) -> None:
        """
        Opens the interaction object.

        Note:
        Assumes object is closed and agent is at its location.
        Objects like fridges, cabinets, drawers can be opened.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

    def close(self) -> None:
        """
        Closes the interaction object.

        Note:
        Assumes object is open and agent is at its location.
        Objects like fridges, cabinets, drawers can be closed.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

    def put_down(self) -> None:
        """
        Puts down the object in hand on a nearby receptacle.

        Note:
        Assumes object is in hand and agent wants to free its hand.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

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
        pass

    def pickup_and_place(self, landmark_name) -> None:
        """
        Picks up the interaction object and places it on the specified landmark.

        Args:
        landmark_name (InteractionObject): Landmark to place the object on.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass

    def empty(self):
        """Empty the object of any other objects on/in it to clear it out. 

        Useful when the object is too full to place an object inside it.

        Returns:
        bool: True if action successful, otherwise False.
        """
        pass
