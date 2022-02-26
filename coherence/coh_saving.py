"""Class and function for saving objects.

CLASSES
-------
SaveObject
-   A class for inheriting specified attributes from another class object.

METHODS
-------
confirm_overwrite
-   Asks the user to confirm whether a pre-existing file should be overwitten.

check_before_overwrite
-   Checks whether a file exists at a specified filepath.
"""




from copy import deepcopy
from os.path import exists
from typing import Any




class SaveObject():
    """A class for inheriting specified attributes from another object.

    PARAMETERS
    ----------
    obj : Any
    -   The object to inherit attributes from.

    attr_to_save : list[str]
    -   The names of the attributes to extract from the object.
    """

    def __init__(self,
        obj: Any,
        attr_to_save: list[str]
        ) -> None:

        for attr_name in attr_to_save:
            setattr(self, attr_name, deepcopy(getattr(obj, attr_name)))



def confirm_overwrite(fpath: str) -> bool:
    """Asks the user to confirm whether a pre-existing file should be
    overwitten.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath where the object will be saved.

    RETURNS
    -------
    write : bool
    -   Whether or not the pre-existing file should be overwritten or not based
        on the user's response.
    """

    write = False
    valid_response = False
    while valid_response is False:
        response = input(
            f"The file '{fpath}' already exists.\nDo you want to "
            "overwrite it? y/n: "
        )
        if response not in ['y', 'n']:
            print(
                "The only accepted responses are 'y' and 'n'. "
                "Please input your response again."
            )
            break
        if response == 'n':
            print(
                "You have requested that the pre-existing file not "
                "be overwritten. The new file has not been saved."
            )
            valid_response = True
        if response == 'y':
            write = True
            valid_response = True

    return write



def check_before_overwrite(
    fpath: str,
    ask_before_overwrite: bool = True
    ) -> bool:
    """Checks whether a file exists at a specified filepath.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath where the object will be saved.

    ask_before_overwrite : bool; default True
    -   If True, the user is asked to confirm whether or not to overwrite a
        pre-existing file if one exists. If False, the user is not asked to
        confirm this and it is overwritten automatically.

    RETURNS
    -------
    bool : str
    -   Whether or not the object should be saved to the filepath.
    """

    if exists(fpath):
        if ask_before_overwrite:
            write = confirm_overwrite(fpath)
        else:
            write = True
    else:
        write = True

    return write
