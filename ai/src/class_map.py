TARGET_CLASSES = ["hardhat", "no_hardhat", "vest", "no_vest"]

# Mapping from raw dataset class names to target classes
RAW_TO_TARGET = {
    "Hardhat": "hardhat",
    "NO-Hardhat": "no_hardhat",
    "Safety Vest": "vest",
    "NO-Safety Vest": "no_vest",
}

def normalize_class(raw_name: str) -> str:
    """Convert a raw class name to the canonical target vocabulary.

    Returns the target class name if it exists in ``TARGET_CLASSES``;
    otherwise returns ``None``.
    """
    return RAW_TO_TARGET.get(raw_name)
