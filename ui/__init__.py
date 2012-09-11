def get_choice( choice_to_explanation ):
    """
    choice_to_explanation: a dictionary mapping possible user responses
    to strings describing what that response will
    cause the script to do
    """
    d = choice_to_explanation

    for key in d:
        print '\t'+key + ': '+d[key]
    prompt = '/'.join(d.keys())+'? '

    first = True
    choice = ''
    while first or choice not in d.keys():
        if not first:
            print 'unrecognized choice'
        first = False
        choice = raw_input(prompt)
    return choice
