#!/usr/bin/env python3

import numpy as np
import re
import textwrap
from cellects.utils.utilitarian import vectorized_len
from cellects.gui.ui_strings import *

def wrap_tip(text, max_length=100):
    # Step 1: Insert newline before '-'
    modif_text = re.sub("-", "\n-", text)

    # Step 2: Insert newline after hyphen followed by whitespace
    modif_text = re.sub("NB:", "\nNB:", modif_text)

    # Step 3: Fill the modified text into lines of max_length, respecting newlines from above
    split_text = np.array(modif_text.split("\n"))
    wrapped_text = ""
    for paragraph in split_text:
        wrapped_text += textwrap.fill(paragraph, width=max_length) + r"\n"
    wrapped_text = re.sub(r"\\n", "\n", wrapped_text)
    return wrapped_text

def process_tips_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pattern = re.compile(r'# START_TIP.*?# END_TIP', re.DOTALL)
    matches = pattern.findall(content)
    for match in matches: # match=matches[1]
        tip_start, _content = match.split('=', 1)
        _, _content, _ = _content.split('"""', 2)
        # first_splitting = _content.split('\n    ')
        first_splitting = _content.split('\n')
        final_split = []
        for f_split in first_splitting:
            final_split += f_split.split('\\n')
        final_split = np.array(final_split)
        content_len = vectorized_len(final_split)
        final_split = final_split[content_len != 0]
        final_split = ' '.join(final_split)
        processed_tip = wrap_tip(final_split)
        tip_end = '\n# END_TIP'

        # Replace the original content with wrapped tip

        new_match = f'{tip_start}= \\\nf"""{processed_tip}"""{tip_end}'
        content = content.replace(match, new_match)

    with open(file_path, 'w') as file:
        file.write(content)

def wrap_md_tip(text):
    final_note = ""
    if "NB:" in text:
        main, note = text.split("NB:")
        split_text = np.array(note.split("\n-"))
        wrapped_text = split_text[0]
        if len(split_text) > 1:
            for paragraph in split_text[1:]:
                wrapped_text += "\n\t -" + re.sub("\n", "", paragraph)# + "\n"
        final_note = "!!! note\n" + wrapped_text
    else:
        main = text
    split_text = np.array(main.split("-"))
    if len(split_text) > 1:
        wrapped_main = split_text[0] + "\n-" + '-'.join(split_text[1:])
    else:
        wrapped_main = split_text[0]
    return wrapped_main + final_note

def update_markdown(file_path, dynamic_content):
    # Generate dynamic content
    for key, value in dynamic_content.items():
        dynamic_lines = []
        # Read the Markdown file as a list of lines
        with open(file_path, "r") as f:
            lines = f.readlines()

        start_marker = f"<!-- START_{key} -->"
        end_marker = f"<!-- END_{key} -->"

        # Find indices of the marker block
        try:
            start_idx = lines.index(start_marker + "\n")
            end_idx = lines.index(end_marker + "\n")
        except ValueError:
            raise RuntimeError(f"Markers not found in Markdown file: {start_marker} and {end_marker}")
        modif_text = wrap_md_tip(value["tips"])
        dynamic_lines.append(f"## {value["label"]}:\n{modif_text}\n")
        # Replace the marker block
        lines[start_idx + 1:end_idx] = dynamic_lines

        # Write back to the file
        with open(file_path, "w") as f:
            f.writelines(lines)

if __name__ == "__main__":
    process_tips_in_file("../src/cellects/gui/UI_strings.py")
    update_markdown("first-analysis/data-localisation.md", FW)
    update_markdown("first-analysis/image-analysis.md", IAW)
    update_markdown("first-analysis/video-tracking.md", VAW)
    update_markdown("advanced/multiple-folders.md", MF)
    update_markdown("advanced/outputs.md", RO)
    update_markdown("advanced/parameters.md", AP)