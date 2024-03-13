#!/usr/bin/env python
# coding: utf-8

# # Practical Implementation of Multithreading and Multiprocessing

# In[20]:


# Import reqired libraries
import pandas as pd
import numpy as np
import time
import threading
import multiprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# In[36]:


# Read the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\Siddheshwar Pandhare\Desktop\data_3.csv")
df.shape


# In[37]:


# Count the number of unique products
num_unique_products = df['Product'].nunique()

# Count the number of unique locations
num_unique_locations = df['Location'].nunique()

print("Number of unique products:", num_unique_products)
print("Number of unique locations:", num_unique_locations)


# ## Multithreading

# In[38]:



# Define parameters
num_threads = 3  # Change this number as needed

# Function to calculate correlation matrix for each location
def calculate_correlation(location, pivot_df):
    correlation_matrix = pivot_df[pivot_df['Location'] == location].drop(['Location', 'Date'], axis=1).corr()
    # Calculate total price ratio for each location
    total_price_ratio = pivot_df[pivot_df['Location'] == location].drop(['Location', 'Date'], axis=1).sum(axis=1)
    return correlation_matrix, total_price_ratio

# Pivot the table to have products as columns
pivot_df = df.pivot_table(index=['Location', 'Date'], columns='Product', values='Price Ratio').reset_index()

# Start time measurement for direct correlation calculation
start_time_direct = time.time()

# Calculate correlation matrix for each location directly
for location in pivot_df['Location'].unique():
    correlation_matrix, total_price_ratio = calculate_correlation(location, pivot_df)
    # Print or store the results as needed
#     print(f"\nCorrelation matrix for {location}:")
#     print(correlation_matrix)
#     print(f"Total price ratio for {location}:")
#     print(total_price_ratio)


# End time measurement for direct correlation calculation
end_time_direct = time.time()

# Print the time taken for direct correlation calculation
elapsed_time_direct = end_time_direct - start_time_direct
print(f"\nTime taken for direct correlation calculation: {elapsed_time_direct:.6f} seconds")

# Start time measurement for threaded correlation calculation
start_time_threaded = time.time()

# Calculate correlation matrix for each location using threading
threads = []
for location in pivot_df['Location'].unique():
    if len(threads) < num_threads:
        thread = threading.Thread(target=calculate_correlation, args=(location, pivot_df))
        threads.append(thread)
        thread.start()
    else:
        threads[0].join()
        threads.pop(0)
        thread = threading.Thread(target=calculate_correlation, args=(location, pivot_df))
        threads.append(thread)
        thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# End time measurement for threaded correlation calculation
end_time_threaded = time.time()

# Print the time taken for threaded correlation calculation
elapsed_time_threaded = end_time_threaded - start_time_threaded
print(f"Time taken for threaded correlation calculation: {elapsed_time_threaded:.6f} seconds")


# ## Multiprocessing and Multithreading

# In[39]:



num_threads = 4  # Change this number as needed
num_processes = 4  # Change this number as needed

# Function to calculate correlation matrix and sum of price ratios by month for each product
def calculate_correlation(location, pivot_df):
    # Calculate correlation matrix
    correlation_matrix = pivot_df[pivot_df['Location'] == location].drop(['Location', 'Date'], axis=1).corr()
    
    # Calculate total price ratio for each location
    total_price_ratio = pivot_df[pivot_df['Location'] == location].drop(['Location', 'Date'], axis=1).sum(axis=1)
    
    # Calculate square of total price ratio
    total_price_ratio_square = total_price_ratio ** 1000
    
    return correlation_matrix, total_price_ratio, total_price_ratio_square


# Pivot the table to have products as columns
pivot_df = df.pivot_table(index=['Location', 'Date'], columns='Product', values='Price Ratio').reset_index()

# Start time measurement for direct correlation calculation
start_time_direct = time.time()

# Calculate correlation matrix for each location directly
for location in pivot_df['Location'].unique():
    calculate_correlation(location, pivot_df)

# End time measurement for direct correlation calculation
end_time_direct = time.time()

# Print the time taken for direct correlation calculation
elapsed_time_direct = end_time_direct - start_time_direct
print(f"\nTime taken for direct correlation calculation: {elapsed_time_direct:.6f} seconds")



# Start time measurement for threaded correlation calculation
start_time_threaded = time.time()

# Calculate correlation matrix for each location using threading
def threaded_correlation():
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(lambda loc: calculate_correlation(loc, pivot_df), pivot_df['Location'].unique())

# Call threaded_correlation function
threaded_correlation()

# End time measurement for threaded correlation calculation
end_time_threaded = time.time()

# Print the time taken for threaded correlation calculation
elapsed_time_threaded = end_time_threaded - start_time_threaded
print(f"Time taken for threaded correlation calculation: {elapsed_time_threaded:.6f} seconds")

# Start time measurement for multiprocessing correlation calculation
start_time_multiprocessing = time.time()

# Calculate correlation matrix for each location using multiprocessing
def multiprocessing_correlation():
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(lambda loc: calculate_correlation(loc, pivot_df), pivot_df['Location'].unique())

# Call multiprocessing_correlation function
multiprocessing_correlation()

# End time measurement for multiprocessing correlation calculation
end_time_multiprocessing = time.time()

# Print the time taken for multiprocessing correlation calculation
elapsed_time_multiprocessing = end_time_multiprocessing - start_time_multiprocessing
print(f"Time taken for multiprocessing correlation calculation: {elapsed_time_multiprocessing:.6f} seconds")


# ## Multiprocessing VS Multithreading

# In[40]:



# Function to calculate correlation matrix and sum of price ratios by month for each product
def calculate_correlation(location, pivot_df):    
    # Calculate correlation matrix
    correlation_matrix = pivot_df[pivot_df['Location'] == location].drop(['Location', 'Date'], axis=1).corr()
    
    # Calculate total price ratio for each location
    total_price_ratio = pivot_df[pivot_df['Location'] == location].drop(['Location', 'Date'], axis=1).sum(axis=1)
    
    # Calculate square of total price ratio
    total_price_ratio_square = total_price_ratio ** 1000
    
    return correlation_matrix, total_price_ratio, total_price_ratio_square

# Function to perform correlation calculation using threading
def threaded_correlation(num_threads):
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(lambda loc: calculate_correlation(loc, pivot_df), pivot_df['Location'].unique())
    end_time = time.time()
    return end_time - start_time

# Function to perform correlation calculation using multiprocessing
def multiprocessing_correlation(num_processes):
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(lambda loc: calculate_correlation(loc, pivot_df), pivot_df['Location'].unique())
    end_time = time.time()
    return end_time - start_time

# Generate sample data
# Code to generate sample data goes here...

# Define parameters
num_threads_list = list(range(1, 8))  # Number of threads from 1 to 7
num_processes_list = list(range(1, 8))  # Number of processes from 1 to 7
thread_timings = []
process_timings = []

# Pivot the table to have products as columns
pivot_df = df.pivot_table(index=['Location', 'Date'], columns='Product', values='Price Ratio').reset_index()

# Calculate timings for different number of threads
for num_threads in num_threads_list:
    thread_timings.append(threaded_correlation(num_threads))

# Calculate timings for different number of processes
for num_processes in num_processes_list:
    process_timings.append(multiprocessing_correlation(num_processes))

# Plotting the timings
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(num_threads_list))
plt.bar(index, thread_timings, bar_width, label='Multithreading')
plt.bar(index + bar_width, process_timings, bar_width, label='Multiprocessing')
plt.xlabel('Number of Threads/Processes')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Threading vs Multiprocessing')
plt.xticks(index + bar_width / 2, num_threads_list)
plt.legend()
plt.show()


# In[ ]:




