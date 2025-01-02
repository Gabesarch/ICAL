import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Dict

from browser_env.actions import action2str
from browser_env import Action, Trajectory
from agent import PromptAgent

def handle_human_in_the_loop(
    agent: PromptAgent,
    trajectory: Trajectory,
    intent: str,
    images: List[Image.Image],
    meta_data: Dict[str, List[str]],
    config_file: str,
    action: Action,
    args: argparse.Namespace,
    count: int,
    human_feedback_list: List[Union[None, List[Tuple[int, str, str, Action, str, Action]]]],
) -> Tuple[Action, int]:
    """
    Handles human-in-the-loop interaction by collecting user feedback and
    updating actions accordingly.

    Args:
        agent (PromptAgent): The agent instance.
        trajectory (Trajectory): The agent's current trajectory.
        intent (str): The current intent/task being executed.
        images (List[Image.Image]): Input images for the task.
        meta_data (Dict[str, List[str]]): Metadata, such as action history.
        config_file (str): The configuration file being used.
        action (Action): The initial action taken by the agent.
        args (argparse.Namespace): Runtime arguments.
        count (int): Current step count in the trajectory.
        human_feedback_list (List[Union[None, List[Tuple]]]): A list to store feedback data.

    Returns:
        Tuple[Action, int]: Updated action and the incremented human feedback count.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    human_satisfaction = False
    num_human_feedbacks = 0

    # Display the image to the user for feedback
    image = np.float32(trajectory[-1]["observation"]["image"][:, :, :3])
    image = image.astype(np.uint8)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(f"Step {count}: Agent's Current Action: {action2str(action, args.action_set_tag)}")
    plt.show()

    while not human_satisfaction:
        # Ask the user for feedback
        print("Agent ACTION: ", action2str(action, args.action_set_tag))
        print("Was this action suboptimal? (yes/no)")
        human_reaction = input("Your feedback: ").strip().lower()

        if human_reaction == "no":
            # User is satisfied with the current action
            print("Thank you! Proceeding with the current action.")
            human_satisfaction = True
        else:
            # User provides feedback
            print("Please describe the issue with the action and suggest an alternative.")
            human_feedback = input("Your feedback: ").strip()

            # Update the agent with the feedback
            new_action = agent.next_action_humanFeedback(
                trajectory=trajectory,
                intent=intent,
                images=images,
                meta_data=meta_data,
                humanFeedback=human_feedback,
                prev_action=action,
            )

            # Log feedback details
            feedback_tuple = (count, intent, trajectory[-1]["observation"]["text"], action, human_feedback, new_action)
            if human_feedback_list[-1] is None:
                human_feedback_list[-1] = [feedback_tuple]
            else:
                human_feedback_list[-1].append(feedback_tuple)

            print("Updated Agent ACTION: ", action2str(new_action, args.action_set_tag))
            print("Was this new action acceptable? (yes/no)")
            human_reaction = input("Your feedback: ").strip().lower()

            if human_reaction == "no":
                print("Understood. Refining the action further.")
                action = new_action
                num_human_feedbacks += 1
            else:
                print("Thank you! Proceeding with the updated action.")
                human_satisfaction = True
                action = new_action

    return action, num_human_feedbacks