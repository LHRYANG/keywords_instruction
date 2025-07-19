# prompt_builder.py

def build_motion_prompt(object_name: str, motion_text: str, attr_text: str = "", max_attributes: int = 1) -> str:
    use_attribute = max_attributes > 0
    attribute_block = f"""- You may add **up to {max_attributes} attribute{'s' if max_attributes > 1 else ''}** (e.g., color, size, age, role, emotion) to describe the object, making the sentence more vivid.
- Attributes should be realistic and relevant to the object.

Attribute keywords for inspiration, you can choose to use or not:
{attr_text}

""" if use_attribute else ""

    prompt = f"""You are given the name of an object (typically a human or animal), and your task is to generate a list of **natural and plausible phrase pairs**, where each pair describes the **same object performing two different actions** (motions){', possibly with some additional attributes' if use_attribute else ''}.

- Each phrase should be descriptive, suitable for use in visual editing prompts or image descriptions, such as:
    ("The fluffy dog is barking loudly.", "The fluffy dog is sniffing curiously.") if attributes are used,
    or ("Here is a girl that is laughing.", "Here is a girl that is crying loudly.") otherwise.
- The object in each phrase should remain the same — and the motion (action) should change.
- You may add **adverbs** to modify the motion (e.g., "running quickly", "barking loudly") to make it more expressive and natural.
- Use **diverse and meaningful motion changes**, as long as they are visually plausible for the object.

{attribute_block}Motion candidates for inspiration:
{motion_text}

Output only the result as a Python list of string pairs, in the format:
[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]

Now, generate 3 pairs for the object: {object_name}
Output:\n
"""
    return prompt



def build_add_remove_prompt(object_name: str, reference_text: str, max_attributes=1) -> str:
    if max_attributes == 0:
        return (
            f"Given an object, please generate up to 5 natural keyword phrases that describe the object.\n\n"
            f"Each phrase should avoid using extra attributes like color, size, or state — focus only on natural descriptors of the object.\n\n"
            f"Return the result as a list and only return the list. For example:\n"
            f"Given 'umbrella', Output:\n"
            f"{{\"umbrella\": [\"black umbrella\", \"an umbrella with flowers\", \"small umbrella\"]}}\n\n"
            f"Please output the JSON results for the object: {object_name}, Output:\n"
        )
    else:
        return (
            f"Given an object, please generate up to 5 slightly more complex but still natural keyword phrases.\n\n"
            f"Each phrase should include **{max_attributes}** additional attribute{'s' if max_attributes > 1 else ''}, "
            f"such as color, material, state, size, position, or style.\n\n"
            f"You may refer to the following attribute keywords:\n{reference_text}\n\n"
            f"Return the result as a list and only return the list. For example:\n"
            f"Given 'umbrella', Output:\n"
            f"{{\"umbrella\": [\"black umbrella\", \"an umbrella with flowers\", \"small umbrella\"]}}\n\n"
            f"Please output the JSON results for the object: {object_name}, Output:\n"
        )


def build_color_change_prompt(object_name: str, reference_text: str, reference_color: str, max_attributes=1) -> str:
    if max_attributes == 0:
        return (
            f"You are given an object name. Your task is to generate up to 5 phrase pairs that describe the same object "
            f"with only the **color attribute changed**.\n\n"
            f"- Each pair must refer to the exact same object, but use a different color adjective.\n"
            f"- The phrases should be natural and descriptive, such as:\n"
            f"  ('a red umbrella', 'a blue umbrella'), ('a green umbrella with flowers', 'a white umbrella with flowers')\n"
            f"- Use everyday, human-like language that might appear in image captions or edit instructions.\n"
            f"- Only vary the color in each pair — do not change any other attributes like material, size, or shape.\n"
            f"- Use the following colors as reference:\n{reference_color}\n\n"
            f"Output only the result as a Python list of string pairs, in the format:\n"
            f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
            f"Now, generate the color-change phrase pairs for the object: {object_name}\nOutput:\n"
        )
    else:
        return (
            f"You are given an object name. Your task is to generate up to 5 phrase pairs that describe the same object "
            f"with only the **color attribute changed**.\n\n"
            f"- Each pair must refer to the exact same object, but use a different color adjective.\n"
            f"- The phrases should be natural and descriptive, such as:\n"
            f"  ('a red umbrella', 'a blue umbrella'), ('a green umbrella with flowers', 'a white umbrella with flowers')\n"
            f"- Use everyday, human-like language that might appear in image captions or edit instructions.\n"
            f"- Only vary the color in each pair — do not change any other attributes like material, size, or shape.\n"
            f"- Use the following colors as reference:\n{reference_color}\n\n"
            f"You can also add {max_attributes} other attributes, such as material, state, size, position, or style. But the difference should only be in the color attribute. Here are some reference values for other attributes:\n{reference_text}\n\n"
            f"Output only the result as a Python list of string pairs, in the format:\n"
            f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
            f"Now, generate the color-change phrase pairs for the object: {object_name}\nOutput:\n"
        )
        