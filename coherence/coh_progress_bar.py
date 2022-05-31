"""Class for creating a tkinter-based progress bar in an external window."""

import tkinter
from tkinter import ttk


class ProgressBar:
    """Class for generating a tkinter-based progress bar shown in an external
    window.

    PARAMETERS
    ----------
    n_steps : int
    -   The total number of steps in the process whose progress is being
        tracked.

    title : str
    -   The name of the process whose progress is being tracked.

    handle_n_exceeded : str; default "warning"
    -   How to act in the case that the total number of steps is exceeded by the
        current step number. If "warning", a warning is raised. If "error", an
        error is raised.

    METHODS
    -------
    update_progress
    -   Increments the current step number in the total number of steps by one.

    close
    -   Closes the progress bar window.
    """

    def __init__(
        self, n_steps: int, title: str, handle_n_exceeded: str = "warning"
    ) -> None:

        self.n_steps = n_steps
        self.title = title
        self.handle_n_exceeded = handle_n_exceeded

        self._step_n = 0
        self.step_increment = 100 / n_steps
        self.window = None
        self.percent_label = None
        self.bar = None

        self._sort_inputs()
        self._create_bar()

    def _sort_inputs(self) -> None:
        """Sorts inputs to the object.

        RAISES
        ------
        NotImplementedError
        -   Raised if the requested method for 'handle_n_exceeded' is not
            supported.
        """

        supported_handles = ["warning", "error"]
        if self.handle_n_exceeded not in supported_handles:
            raise NotImplementedError(
                "Error: The method for hanlding instances of the total number "
                f"of steps being exceeded '{self.handle_n_exceeded}' is not "
                f"supported. Supported inputs are {supported_handles}."
            )

    def _create_bar(self) -> None:
        """Creates the tkinter root object and progress bar window with the
        requested title."""

        self.root = tkinter.Tk()
        self.root.wm_attributes("-topmost", True)

        self.bar = ttk.Progressbar(self.root, length=250)
        self.bar.pack(padx=10, pady=10)
        self.percent_label = tkinter.StringVar()
        self.percent_label.set(self.progress)
        ttk.Label(self.root, textvariable=self.percent_label).pack()

        self.root.update()

    @property
    def progress(self) -> str:
        """Getter for returning the percentage completion of the progress bar as
        a formatted string with the title.

        RETURNS
        -------
        str
        -   The percentage completion of the progress bar, formatted into a
            string with the structure: title; percentage.
        """

        return f"{self.title}\n{str(int(self.bar['value']))}% complete"

    @property
    def step_n(self) -> int:
        """Getter for returning the number of steps completed in the process
        being followed."""

        return self._step_n

    @step_n.setter
    def step_n(self, value) -> None:
        """Setter for the number of steps completed in the process being
        followed.

        RAISES
        ------
        ValueError
        -   Raised if the current number of steps is greater than the total
            number of steps specified when the object was created, and if
            'handle_n_exceeded' was set to "error".
        """

        if value >= self.n_steps:
            if self.handle_n_exceeded == "warning":
                print(
                    "Warning: The maximum number of steps in the progress bar "
                    "has been exceeded.\n"
                )
            else:
                raise ValueError(
                    "Error: The maximum number of steps in the progress bar "
                    "has been exceeded."
                )

        self._step_n = value

    def update_progress(self) -> None:
        """Increments the step number of the process by one and updates the
        progress bar value and label appropriately."""

        self.step_n += 1
        self.bar["value"] += self.step_increment
        self.percent_label.set(self.progress)
        self.root.update()

    def close(self) -> None:
        """Destroys the tkinter root object the progress bar is linked to."""

        self.root.destroy()
