def noRejectionBasedOnPrior(prior):
    return False


def noRejectionBasedOnSimulation(sim):
    return False


def rejectLowPrior(prior):
    if any(x < 0.15 for x in prior):
        return True
