from numpy import random

def generate_inter_event_data(rates: [int], N: int, length):
    """
    Generating univariate-inter-event times from various Exponential Distributions parametized by rates.  Sequences
    of inter-event times are varying in length.  Output is a List of List of Numpy Arrays.

    rates: Array of int
    length: How many time-steps to generate interevent times for
    N: Number of samples per Exponential Distribution

    """
    tasks = []
    for r in rates:
        task = []
        for n in range(N):
            size = random.randint(low=length-5, high=length+5, size=1)[0]
            task.append(random.exponential(r, size))
        tasks.append(task)
    return tasks

