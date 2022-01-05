



class ProcessingOrderError(Exception):
    """Class for raising exceptions/errors associated with performing processing
    steps in an incorrect order.
    """
    pass



class MissingAttributeError(Exception):
    """Class for raising exceptions/errors associated with attributes of an
    object that have not been instantiated.
    """
    pass



class EntryLengthError(Exception):
    """Class for raising exceptions/error associated with entries within a list
    having a nonidentical length.
    """



class ChannelTypeMismatch(Exception):
    """Class for raising exceptions/errors associated with trying to handle
    channels of different types.
    """