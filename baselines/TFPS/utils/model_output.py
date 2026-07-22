def unpack_model_output(model_output):
    """Normalize TFPS model outputs to cluster affinities plus forecast."""
    if isinstance(model_output, tuple):
        if len(model_output) != 3:
            raise ValueError(
                "TFPS tuple output must contain "
                "(time_affinity, frequency_affinity, forecast)"
            )
        return model_output
    return None, None, model_output
