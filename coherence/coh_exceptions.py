from abc import ABC



class CustomError(ABC):
    """An abstract class for defining custom exceptions/errors.

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
    steps in an incorrect order.

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
    object that have not been instantiated.

    PARAMETERS
    ----------
    N/A

    METHODS
    -------
    N/A
    """
    pass


