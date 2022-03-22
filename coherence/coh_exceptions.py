"""Custom exception classes.

CLASSES
-------
ProcessingOrderError : Exception
-   Class for raising exceptions/errors associated with performing processing
    steps in an incorrect order.

UnavailableProcessingError : Exception
-   Class for raising exceptions/error associated with trying to perform
    processing steps that cannot be done.

MissingAttributeError : Exception
-   Class for raising exceptions/errors associated with attributes of an
    object that have not been instantiated.

EntryLengthError : Exception
-   Class for raising exceptions/errors associated with entries within a list
    having a nonidentical length.

DuplicateEntryError : Exception
-   Class for raising exceptions/errors associated with duplicate entries
    occuring within an object.

MissingEntryError : Exception
-   Class for raising exceptions/errors associated with missing entries between
    objects.

ChannelOrderError : Exception
-   Class for raising exceptions/errors associated with lists of channel names
    being in different orders.

ChannelTypeError : Exception
-   Class for raising exceptions/errors associated with trying to handle
    channels of different types.

InputTypeError : Exception
-   Class for raising exceptions/errors associated with input objects being of
    the wrong type.

MissingFileExtensionError : Exception
-   Class for raising exceptions/errors associated with a filename string
    missing a filetype extension.
"""


class ProcessingOrderError(Exception):
    """Class for raising exceptions/errors associated with performing processing
    steps in an incorrect order.
    """


class UnavailableProcessingError(Exception):
    """Class for raising exceptions/error associated with trying to perform
    processing steps that cannot be done.
    """


class MissingAttributeError(Exception):
    """Class for raising exceptions/errors associated with attributes of an
    object that have not been instantiated.
    """


class EntryLengthError(Exception):
    """Class for raising exceptions/error associated with entries within a list
    having a nonidentical length.
    """


class DuplicateEntryError(Exception):
    """Class for raising exceptions/errors associated with duplicate entries
    occuring within an object.
    """


class MissingEntryError(Exception):
    """Class for raising exceptions/errors associated with missing entries between
    objects.
    """


class ChannelOrderError(Exception):
    """Class for raising exceptions/errors associated with lists of channel
    names being in different orders.
    """


class ChannelTypeError(Exception):
    """Class for raising exceptions/errors associated with trying to handle
    channels of different types.
    """


class InputTypeError(Exception):
    """Class for raising exceptions/errors associated with input objects being
    of the wrong type.
    """


class MissingFileExtensionError(Exception):
    """Class for raising exceptions/errors associated with a filename string
    missing a filetype extension."""
