import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from copy import deepcopy



def gen_bootstrap_data(data, n=None):
    if n is None:
        n = len(data)
    indices = np.random.choice(np.arange(len(data)), size=n, replace=True)
    
    # should work with both numpy arrays and pandas dataframes
    if isinstance(data, np.ndarray):
        return data[indices]
    else:
        return data.iloc[indices]

def gen_subset_data(data, n=None):
    if n is None:
        raise ValueError("n must be specified for subset data generation")
    indices = np.random.choice(np.arange(len(data)), size=n, replace=False)
    
    # should work with both numpy arrays and pandas dataframes
    if isinstance(data, np.ndarray):
        return data[indices]
    else:
        return data.iloc[indices]

def std(data):
    return np.std(data, ddof=1)

def bootstrap_estimation(org_data, experiment, sampling_method=gen_bootstrap_data, func=std, n=None, n_experiments=1000):
    # use mp to parallelize the process
    pool = mp.Pool(mp.cpu_count())

    results = []
    for _ in range(n_experiments):
        data = sampling_method(org_data, n)
        results.append(pool.apply_async(experiment, args=(data,)))

    pool.close()

    results = [res.get() for res in results]
    raw_results = deepcopy(results)

    # should work with both numpy arrays and pandas dataframes
    if isinstance(results[0], np.ndarray):
        results = np.array(results)
        return func(results), raw_results, results
        # return np.std(results, ddof=1) # ddof=1 for sample standard deviation so that it is unbiased
    else:
        result_df = results[0].copy()
        # measurement df holds the lists of values instead of the processed values
        measurement_df = pd.DataFrame(index=result_df.index, columns=result_df.columns, dtype=object)
        
        # get array for each row, column pair in the dataframe
        for row in result_df.index:
            for col in result_df.columns:
                res = np.array([res.loc[row, col] for res in results])

                # count number of nan values, remove them, but print a warning
                n_nan = np.sum(np.isnan(res))
                if n_nan > 0:
                    print(f"[WARNING]: {n_nan} of {len(res)} values in the results are NAN for {row}, {col}. Removing them.")
                    res = res[~np.isnan(res)]

                # result_df.loc[row, col] = np.std(res, ddof=1)
                result_df.loc[row, col] = func(res)
                measurement_df.loc[row, col] = res
        # return result_df, raw_results
        return result_df, raw_results, measurement_df
