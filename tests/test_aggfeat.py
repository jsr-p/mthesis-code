import random

import numpy as np
import polars as pl

from dstnx.features import agg_neighbors


def gen_data():
    # Initialize empty lists for count, average, and observations
    counts = []
    averages = []
    observations = []
    idc = []

    num_observations = 10  # Number of observations to generate

    for _ in range(num_observations):
        # Step 1: Create a list of numbers with a given average
        num_count = random.randint(5, 20)  # Random number of values in the list
        num_list = [random.randint(1, 100) for _ in range(num_count)]  # Random numbers
        avg = sum(num_list) / num_count  # Calculate average
        idx = random.randint(0, 3)  # Random index
        # Step 3: Save the count, average, and observations
        counts.append(num_count)
        averages.append(avg)
        observations.append(num_list)
        idc.append(idx)

    data = {
        "id": idc,
        "arblos_avg": averages,
        "arblos_count": counts,
    }
    return data, observations, idc


def test_aggfeat():
    data, observations, idc = gen_data()
    df = pl.DataFrame(data)
    gp = df.groupby("id").agg([agg_neighbors.avg_combined("arblos")])

    for idx in np.unique(idc):
        observed = [obs for obs, i in zip(observations, idc) if i == idx]
        observed_counts = [len(obs) for obs in observed]
        observed = sum([sum(obs) for obs in observed]) / sum(observed_counts)
        assert observed == gp.filter(pl.col("id") == idx)["arblos_avg"].item()


if __name__ == "__main__":
    test_aggfeat()
