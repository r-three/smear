def string_to_float(string, default=-1.):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def string_to_int(string, default=-1):
    """Converts string to int, using default when conversion not possible."""
    try:
        return int(string)
    except ValueError:
        return default

def string_processor(string, default=-1):
    string = string.strip().lower()
    if string in ["yes", "entailment", "equivalent", "acceptable", "positive", "good", "happy", "duplicates", "true", "correct", "always", "guaranteed"]:
        return 1
    elif string in ["no", "not entailment", "not equivalent", "unacceptable", "negative", "bad", "sad", "not duplicates", "false", "incorrect", "never", "impossible"]:
        return 0
    elif string in ["neither", "inconclusive", "sometimes", "maybe", "possible"]:
        return 2
    else:
        return default

def get_post_processor(task):
    """Returns post processor required to apply on the predictions/targets
    before computing metrics for each task."""
    if task in ['sst2', 'mnli', "qnli", "rte", "qqp", "cola", "mrpc"]:
        return string_processor
    if task == "stsb":
        return string_to_float
    else:
        return None
