"""Public warning and exception types for pyphenix."""


class UnsupportedHarmonyVersionError(ValueError):
    """Raised when a Harmony XML file carries an unrecognised XML namespace.

    Harmony declares its schema version as the default XML namespace on the
    root element (see ``pyphenix._reader.HARMONY_NAMESPACES``). Raised by
    :class:`OperaPhenixReader` when an index XML is un-namespaced or written
    for a Harmony version pyphenix cannot parse, rather than failing later
    with an opaque ``AttributeError`` from a lookup that matched nothing.
    """


class FFCCoverageWarning(UserWarning):
    """Emitted when FFC profiles cover only some of the requested channels.

    Raised by :meth:`OperaPhenixReader.apply_ffc` and
    :meth:`OperaPhenixReader.ffc_correction_images` when at least one requested
    channel is missing a real correction profile (either absent from
    ``ffc_profiles`` entirely or present with ``has_correction() == False``).
    The reader is silent only when ``ffc_profiles`` is empty (no FFC XML
    detected at all) or when every requested channel has a real profile.
    """
