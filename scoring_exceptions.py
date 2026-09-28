class ZeroDivisionError(Exception):
    def __init__(self, message="Processing Failed: Invalid image dimensions. Image area cannot be zero."):
        super().__init__(message)

class NegativeValueError(Exception):
    def __init__(self, message="Processing Failed: Corrupted metric data. Area calculations cannot be negative."):
        super().__init__(message)

class InvalidRibCountError(Exception):
    def __init__(self, message="Processing Failed: Corrupted detection data. Rib count cannot be negative."):
        super().__init__(message)

class MissingDataError(Exception):
    def __init__(self, message="Processing Failed: Incomplete segmentation data. Ensure all anatomy masks are processed."):
        super().__init__(message)