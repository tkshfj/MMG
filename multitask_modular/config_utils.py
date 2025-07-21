# config_utils.py
def load_and_validate_config(wandb_config):
    """
    Loads and validates experiment config. Sets defaults for missing optional fields.
    Args:
        wandb_config: wandb.config object or a dict.
    Returns:
        config (dict): Validated configuration dictionary.
    Raises:
        ValueError: If required keys are missing.
    """
    # Accept both wandb.config and dict
    config = dict(wandb_config)

    # Defaults for optional fields
    optional_defaults = {
        "optimizer": "Adam",
        "input_shape": (256, 256),
        "base_filters": 32,
        "dropout": 0.2,
        "base_learning_rate": 2e-4,
        "l2_reg": 1e-4,
        "split": (0.7, 0.15, 0.15),
        "task": "multitask",
    }
    for k, v in optional_defaults.items():
        config.setdefault(k, v)

    # Fallback for required file path (warn if missing)
    required = ["batch_size", "epochs", "metadata_csv"]
    missing = [k for k in required if k not in config]
    if "metadata_csv" in missing:
        # Provide a safe fallback but alert the user
        default_csv = "../data/processed/cbis_ddsm_metadata_full.csv"
        print(f"WARNING: 'metadata_csv' not found in config. Using default: {default_csv}")
        config["metadata_csv"] = default_csv
        missing.remove("metadata_csv")
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Type conversions for tuples (from str, e.g., in sweep configs)
    def to_tuple(val, typ=float):
        if isinstance(val, str):
            val = val.strip("()[] ").replace(" ", "")
            return tuple(typ(x) for x in val.split(",") if x)
        if isinstance(val, (list, tuple)):
            return tuple(typ(x) for x in val)
        return val

    config["input_shape"] = to_tuple(config["input_shape"], int)
    config["split"] = to_tuple(config["split"], float)

    # Standardize key types
    if isinstance(config["batch_size"], str):
        config["batch_size"] = int(config["batch_size"])
    if isinstance(config["epochs"], str):
        config["epochs"] = int(config["epochs"])

    return config


def print_config(config):
    print("Experiment Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")


# def load_and_validate_config(wandb_config):
#     """
#     Loads and validates experiment config. Sets defaults for missing optional fields.
#     Args: wandb_config: wandb.config object or a dict.
#     Returns: config (dict): Validated configuration dictionary.
#     Raises: ValueError: If required keys are missing.
#     """
#     # Accept both wandb.config and dict
#     config = dict(wandb_config)

#     # Required keys
#     required = ["batch_size", "epochs", "metadata_csv"]
#     missing = [k for k in required if k not in config]
#     if missing:
#         raise ValueError(f"Missing required config keys: {missing}")

#     # Set defaults for optional fields (use standard names)
#     config.setdefault("optimizer", "Adam")
#     config.setdefault("input_shape", (256, 256))
#     config.setdefault("base_filters", 32)
#     config.setdefault("dropout", 0.2)
#     config.setdefault("base_learning_rate", 2e-4)
#     config.setdefault("l2_reg", 1e-4)
#     config.setdefault("split", (0.7, 0.15, 0.15))
#     config.setdefault("task", "multitask")

#     # Type conversions if needed
#     if isinstance(config["input_shape"], str):
#         config["input_shape"] = tuple(map(int, config["input_shape"].strip("() ").replace(" ", "").split(",")))
#     if isinstance(config["split"], str):
#         config["split"] = tuple(map(float, config["split"].strip("() ").replace(" ", "").split(",")))

#     return config


# def print_config(config):
#     print("Experiment Config:")
#     for k, v in config.items():
#         print(f"  {k}: {v}")


# def load_and_validate_config(wandb_config):
#     """
#     Loads and validates experiment config. Sets defaults for missing optional fields.
#     Args: wandb_config: wandb.config object or a dict.
#     Returns: config (dict): Validated configuration dictionary.
#     Raises: ValueError: If required keys are missing.
#     """
#     # Accept both wandb.config and dict
#     config = dict(wandb_config)
#     # Required keys
#     required = ["batch_size", "epochs", "metadata_csv"]
#     missing = [k for k in required if k not in config]
#     if missing:
#         raise ValueError(f"Missing required config keys: {missing}")
#     # Set defaults for optional fields
#     config.setdefault("batch_size", 16)
#     config.setdefault("learning_rate", 2e-4)
#     config.setdefault("weight_decay", 1e-4)
#     config.setdefault("epochs", 40)
#     config.setdefault("optimizer", "Adam")
#     config.setdefault("input_shape", (256, 256))
#     config.setdefault("base_filters", 32)
#     config.setdefault("dropout", 0.2)
#     config.setdefault("base_learning_rate", 2e-4)
#     config.setdefault("l2_reg", 1e-4)
#     config.setdefault("split", (0.7, 0.15, 0.15))
#     config.setdefault("task", "multitask")

#     # Type conversions if needed
#     if isinstance(config["input_shape"], str):
#         # Convert string "(256, 256)" to tuple
#         config["input_shape"] = tuple(map(int, config["input_shape"].strip("() ").split(",")))
#     if isinstance(config["split"], str):
#         config["split"] = tuple(map(float, config["split"].strip("() ").split(",")))

#     # Could add more sophisticated type checking or conversions here

#     return config


# # Optional: function to print config summary
# def print_config(config):
#     print("Experiment Config:")
#     for k, v in config.items():
#         print(f"  {k}: {v}")
