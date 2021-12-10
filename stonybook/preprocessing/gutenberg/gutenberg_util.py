import os

from stonybook.preprocessing.gutenberg.gutenberg_constants_custom import TEXT_END_MARKERS
from stonybook.preprocessing.gutenberg.gutenberg_constants_custom import TEXT_START_MARKERS
from stonybook.preprocessing.gutenberg.gutenberg_constants_custom import LEGALESE_END_MARKERS
from stonybook.preprocessing.gutenberg.gutenberg_constants_custom import LEGALESE_START_MARKERS
from stonybook.preprocessing.gutenberg.gutenberg_constants_custom import FULL_LINE_START_MARKERS

def get_strip_locations(text):
    """Remove lines that are part of the Project Gutenberg header or footer.
    Sourced from Gutenberg python package
    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp
    Args:
        text (unicode): The body of the text to clean up.
    Returns:
        unicode: The text with any non-text content removed.
    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    reset = True
    footer_found = False
    ignore_section = False
    
    start_line = None
    end_line = None

    for line_number, line in enumerate(lines):
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS) or any(line.strip() == l for l in FULL_LINE_START_MARKERS) or "gutenberg.net" in line.lower() or "www.archive.org" in line.lower() or (line.strip().startswith('Florida') and line_number > 0 and lines[line_number - 1].strip().lower().endswith('university of')):
                if (not start_line) or (start_line and line_number - start_line <= 100):
                    reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                start_line = None
                continue

        if i >= len(lines) - 600:#100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                end_line = line_number
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            if start_line is None:
                start_line = line_number
            out.append(line.rstrip(sep))
            i += 1
    
    if start_line is None:
        start_line = 0
    if end_line is None:
        end_line = len(lines)
    # If next line is not empty, move to next line
    entered = False
    while start_line < len(lines) - 1 and len(lines[start_line + 1].strip()) != 0:
        entered = True
        start_line += 1
    if entered:
        start_line += 1
    # Move start line to first non-empty line
    while start_line < len(lines) and len(lines[start_line].strip()) == 0:
        start_line += 1
    # Move end line to the line after the last non-empty line
    while end_line - 1 >= 0 and len(lines[end_line - 1].strip()) == 0:
        end_line -= 1
    
    return start_line, end_line