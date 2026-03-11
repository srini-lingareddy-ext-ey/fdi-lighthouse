def parse_snake_case(s: str) -> str:
    """
    Convert a snake_case string to Title Case with spaces.

    Parameters
    ----------
    s : str
        A string in snake_case format (words separated by underscores).

    Returns
    -------
    str
        A string in Title Case format with spaces replacing underscores.

    Examples
    --------
    >>> parse_snake_case('hello_world')
    'Hello World'
    >>> parse_snake_case('my_variable_name')
    'My Variable Name'
    """
    return s.replace('_', ' ').title()
