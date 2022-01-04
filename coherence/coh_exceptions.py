from abc import ABC



class CustomError(ABC):
    """Abstract class for defining custom exceptions/errors. Implemented in the
    subclasses.

    PARAMETERS
    ----------
    N/A

    METHODS
    -------
    N/A
    """
    pass



class ProcessingOrderError(CustomError):
    """Class for raising exceptions/errors associated with performing processing
    steps in an incorrect order. Subclass of the abstract class CustomError.

    PARAMETERS
    ----------
    N/A

    METHODS
    -------
    N/A
    """
    pass



class MissingAttributeError(CustomError):
    """Class for raising exceptions/errors associated with attributes of an
    object that have not been instantiated. Subclass of the abstract class
    CustomError.

    PARAMETERS
    ----------
    N/A

    METHODS
    -------
    N/A
    """
    pass



class EntryLengthError(CustomError):
    """Class for raising exceptions/error associated with entries within a list
    having a nonidentical length. Subclass of the abstract class CustomError.

    PARAMETERS
    ----------
    N/A

    METHODS
    -------
    N/A
    """