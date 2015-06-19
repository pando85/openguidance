

def phrase_to_lines(phrase, length=80):
    """splits a string along whitespace and distributes the parts into
    lines of the given length.

    each paragraph is followed by a blank line, replacing all blank
    lines separating the paragraphs in the phrase; if paragraphs get
    squashed in your multiline strings, try inserting explicit newlines.

    """

    import re
    parag_ptn = r'''(?x)      # verbose mode
    (?:                       # non-capturing group:
        [ \t\v\f\r]*          #    any non-breaking space
        \n                    #    linebreak
    ){2,}                     # at least two of these
    '''

    paragraphs = re.split(parag_ptn, phrase)
    lines = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        words = paragraph.split()
        line = ''
        for word in words:
            if len(line) + len(word) > length:
                lines.append(line.rstrip())
                line = ''
            line += word + ' '
        lines += [line.rstrip(), '']
    return lines
