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


def build_modify_change_prompt(object_name: str, reference_text: str, reference_modify: str, max_attributes=1) -> str:
    prompt = (
        f"You are given the name of an object and a set of attributes that can describe or modify it.\n\n"
        f"Your task is to generate **up to 5 natural phrase pairs**, where each pair refers to the **same object** "
        f"but differs by **modifying exactly one attribute** from a specified attribute category.\n\n"
        f"Requirements:\n"
        f"- Each pair should describe the same object but with a different value for **one attribute**.\n"
        f"- All pairs must modify attributes from the **same attribute category**.\n"
        f"- Use natural and concise expressions that are plausible and visually distinct.\n"
        f"- Do **not** introduce new objects, actions, or attribute categories beyond the target.\n"
        f"- Modify **only one attribute** in each pair.\n"
        f"Here is an example:\n"
        f"  ('a small umbrella', 'a long umbrella'), ('a round coin', 'a rectangular coin')\n\n"
    )

    if max_attributes > 0:
        prompt += (
            f"- You may optionally **add up to {max_attributes} supporting attributes** "
            f"to make the descriptions more vivid, as long as they do not change across the two phrases in a pair.\n"
            f"- Here are some inspirations for additional attributes:\n{reference_text}\n"
        )
    else:
        prompt += "- Do **not** add any extra attributes beyond the target attribute to be modified.\n"

    prompt += (
        f"\n📌 The allowed attribute category for modification is:\n{reference_modify}\n\n"
        f"Please return only a Python list of tuples. Each tuple should contain two phrases, formatted as:\n"
        f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
        f"Now generate the phrase pairs for the object: **{object_name}**\n"
        f"Output:\n"
    )
    return prompt




def build_transform_global_prompt(object_name: str, reference_text: str, reference_scene: str, max_attributes=1) -> str:

    if max_attributes == 0:
        return (
            f"You are given an object name. Your task is to generate up to 5 phrase pairs that describe the same object "
            f"with only the **scene attribute changed**.\n\n"
            f"- Each pair must refer to the exact same object, but in different scenes.\n"
            f"- The phrases should be natural and descriptive, such as:\n"
            f"  ('a red umbrella in rainy day', 'a red umbrella in sunny day')\n"
            f"- Use everyday, human-like language that might appear in image captions or edit instructions.\n"
            f"- Only vary the scene in each pair — do not change any other attributes like material, size, or shape.\n"
            f"- You may reference the following scenes:\n{reference_scene}\n\n"
            f"Output only the result as a Python list of string pairs, in the format:\n"
            f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
            f"Now, generate the scene-change phrase pairs for the object: {object_name}\nOutput:\n"
        )
    else:
        return (
            f"You are given an object name. Your task is to generate up to 5 phrase pairs that describe the same object "
            f"with only the **scene attribute changed**.\n\n"
            f"- Each pair must refer to the exact same object, but in different scenes.\n"
            f"- The phrases should be natural and descriptive, such as:\n"
            f"  ('a red umbrella in rainy day', 'a red umbrella in sunny day')\n"
            f"- Use everyday, human-like language that might appear in image captions or edit instructions.\n"
            f"- Only vary the scene in each pair — do not change any other attributes like material, size, or shape.\n"
            f"- You may reference the following scenes:\n{reference_scene}\n\n"
            f"You can also add {max_attributes} other attributes, such as material, state, size, position, or style to describe the object or the scene. But the difference should only be in the scene attribute. Here are some reference values for other attributes:\n{reference_text}\n\n"
            f"Output only the result as a Python list of string pairs, in the format:\n"
            f"[('description 1-1', 'description 1-2'), ('description 2-1', 'description 2-2'), ...]\n\n"
            f"Now, generate the scene-change phrase pairs for the object: {object_name}\nOutput:\n"
        )



def build_replace_prompt(object_name: str, reference_text: str, max_attributes=1) -> str:

    if max_attributes==0:
        return f"""
You are given an object name and asked to generate a list of **natural and plausible phrase pairs**, where each pair describes two objects that can reasonably **replace each other** in a scene.

- Each phrase should be short and descriptive, suitable for use in visual editing prompts or image descriptions.
- Use **common or creative substitutions**, as long as they are visually plausible in some context. For example:
(a apple, a watermelon)
- Use **common or creative substitutions**, as long as they are visually plausible in some context.

Return only a list of 5 phrase pairs in this format:
[
("phrase1-1", "phrase1-2"),
("phrase2-1", "phrase2-2"),
...
]

Please output the results for the object: {object_name}
Output:\n 
"""
    else:
        return f"""
You are given an object name and asked to generate a list of **natural and plausible phrase pairs**, where each pair describes two objects that can reasonably **replace each other** in a scene.

- Each phrase should be short and descriptive, suitable for use in visual editing prompts or image descriptions.
- Replacements can vary in **type, category, function, or style** — they should **not** have the same identity.
- Use **common or creative substitutions**, as long as they are visually plausible in some context. For example:
(a red apple, a green watermelon)
- Each phrase may contain **up to {max_attributes} attribute{'s' if max_attributes > 1 else ''}**, such as color, material, shape, size, or function.

Reference attribute keywords for inspiration:
{reference_text}

Return only a list of 5 phrase pairs in this format:
[
("phrase1-1", "phrase1-2"),
("phrase2-1", "phrase2-2"),
...
]

Please output the results for the object: {object_name}
Output:\n 
"""
