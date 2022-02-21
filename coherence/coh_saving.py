


class SaveObject():

    def __init__(self,
        obj: Any,
        attr_to_save: list[str]
        ) -> None:

        for attr_name in attr_to_save:
            setattr(self, attr_name, getattr(self, attr_name))
