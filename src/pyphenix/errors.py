"""Public warning and exception types for pyphenix."""


class FFCCoverageWarning(UserWarning):
    """Emitted when FFC profiles cover only some of the requested channels.

    Raised by :meth:`OperaPhenixReader.apply_ffc` and
    :meth:`OperaPhenixReader.ffc_correction_images` when at least one requested
    channel is missing a real correction profile (either absent from
    ``ffc_profiles`` entirely or present with ``has_correction() == False``).
    The reader is silent only when ``ffc_profiles`` is empty (no FFC XML
    detected at all) or when every requested channel has a real profile.
    """
