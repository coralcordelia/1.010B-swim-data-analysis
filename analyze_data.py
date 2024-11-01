import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from functools import reduce

data = []
with open('swim_data_strict.csv','r') as raw_data:
   all_data = list(csv.reader(raw_data))
   names = all_data[0]
   data = np.array(all_data[1:])
       
data_cols = {
    names[index]: data[:, index] for index in range(len(names))
}

for name, col in data_cols.items():
    if name in names[8:]:
        data_cols[name] = np.array(col, float)
    elif name != 'State':
        data_cols[name] = np.array(col, int)

column_catagorizations = {
    'Times': names[8:],
    'Integer Qualities': names[0:2] + names[3:8],
    'Catagoricals': names[2:4] + names[6:8]
}



print(data_cols['Split-1'][:100])

def find_stat_single_var(col, stat_name):
    if stat_name == 'Mean':
        return sum(value for value in col) / len(col)
    if stat_name == 'Minimum':
        return min(col)
    if stat_name == 'Sample Variance':
        mean = find_stat_single_var(col, 'Mean')
        return sum((value - mean) ** 2 for value in col) / (len(col) - 1)
    elif stat_name == 'Sample Standard Deviation':
        return find_stat_single_var(col, 'Sample Variance') ** 0.5

print(f"{'Mean':20} {find_stat_single_var(data_cols['Time'], 'Mean'):10.2f}")  
print(f"{'Sample Variance':20} {find_stat_single_var(data_cols['Time'], 'Sample Variance'):10.2f}")    
print(f"{'Standard Deviation':20} {find_stat_single_var(data_cols['Time'], 'Sample Standard Deviation'):10.2f}")    

def cond_var(col1, col2):
    mean_col2 = find_stat_single_var(col2, 'Mean')
    filtered_col1 = [col1[i] for i in range(len(col2)) if col2[i] < mean_col2]
    return find_stat_single_var(filtered_col1, 'Sample Variance')

def covar(col1, col2):
    mean_col1 = find_stat_single_var(col1, 'Mean')
    mean_col2 = find_stat_single_var(col2, 'Mean')
    return sum((col1[i] - mean_col1) * (col2[i] - mean_col2) for i in range(len(col1))) / (len(col1) - 1)

def reg_coef(col1, col2):
    return (covar(col1, col2)) / (find_stat_single_var(col1, 'Sample Standard Deviation') * find_stat_single_var(col2, 'Sample Standard Deviation'))

def slope(col1, col2):
    return reg_coef(col1, col2) * find_stat_single_var(col1, 'Sample Standard Deviation') / find_stat_single_var(col2, 'Sample Standard Deviation')

def T_val(col1, col2):
    r = reg_coef(col1, col2)
    return r * ((len(col1) - 2) ** 0.5) / ((1 - r ** 2) ** 0.5)

def plot(col1, col2, name1, name2, ax):
    ax.scatter(col1, col2)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_title(name2 + ' vs. ' + name1)

def print_information_variable (name):
    print(f"Variable: {name:10} Cond. S. Var.: {cond_var(data_cols['Time'], data_cols[name]):8.2f} CSSD: {cond_var(data_cols['Time'], data_cols[name])**0.5:6.2f}")
    print(f"Mean: {find_stat_single_var(data_cols[name], 'Mean'):10.2f} Variance: {find_stat_single_var(data_cols[name], 'Sample Variance'):10.2f} Standard Deviation: {find_stat_single_var(data_cols[name], 'Sample Standard Deviation'):10.2f}")
    print(f"Regression Coef: {reg_coef(data_cols['Time'], data_cols[name]):4.3f} T:{T_val(data_cols['Time'], data_cols[name]):5.2f} Slope {slope(data_cols['Time'], data_cols[name]):6.3f}")

for ind, name in enumerate(column_catagorizations['Times']):
    print_information_variable(name)

for name in column_catagorizations['Integer Qualities']:    
    print_information_variable(name)


print(find_stat_single_var(data_cols['Time'], 'Minimum'))
fig, ax = plt.subplots()

plot(data_cols['Seed'], data_cols['Time'], 'Seed', 'Time', ax)


def partition_set(partitions1, partitions2, var_name1, var_name2):
    column1 = data_cols[var_name1]
    column2 = data_cols[var_name2]
    buckets1 = list(zip(partitions1, partitions1[1:]))
    buckets2 = list(zip(partitions2, partitions2[1:]))
    output = np.zeros([len(buckets1), len(buckets2)])
    for i, bucket1 in enumerate(buckets1):
        for j, bucket2 in enumerate(buckets2):
            output[i, j] = len(reduce(np.intersect1d, (np.where(column1 >= bucket1[0]), np.where(column1 < bucket1[1]), np.where(column2 >= bucket2[0]), np.where(column2 < bucket2[1]))))
    return output



def chi_squared(arr):
    total = np.sum(arr)
    freq1 = np.sum(arr, 0)
    freq2 = np.sum(arr, 1)
    expected = np.zeros_like(arr)
    it = np.nditer(arr, flags=['multi_index'])
    for x in it:
        i, j = it.multi_index
        expected[i, j] = freq2[i] * freq1[j] / total
    values = (expected - arr)
    values *= values
    values /= expected
    return np.sum(values), arr.shape

chi_squared_value, dims = chi_squared(partition_set([0, 320, 340, 360, 380, 400, 420, 440, 460, 480, 510, 540, 570, 600, 660, 720, 780, 900, 1080, 3600], 
                    [2009, 2011, 2013, 2015], 'Time', 'Year'))
n = (dims[0] - 1) * (dims[1] - 1)

print(f"Chi-squared = {chi_squared_value:6.2f}, n = {n}")



plt.show()
