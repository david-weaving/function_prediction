import numpy as np


def rearrange_arrays(array1, array2):
    # Step 1: Combine and sort arrays
    combined = sorted(zip(array1, array2), key=lambda x: x[0])
    
    # Unzip sorted pairs back into separate arrays
    sorted_array1, sorted_array2 = zip(*combined)
    
    # Convert to lists for easier manipulation
    sorted_array1 = list(sorted_array1)
    sorted_array2 = list(sorted_array2)
    
    # Step 2: Extract highest, lowest, and third highest values
    highest = sorted_array1[-1]
    lowest = sorted_array1[0]
    third_highest = sorted_array1[-3]
    
    # Step 3: Remove highest, lowest, and third highest from sorted lists
    sorted_array1.remove(highest)
    sorted_array1.remove(lowest)
    sorted_array1.remove(third_highest)
    
    index_highest = sorted_array2.pop(-1)
    index_lowest = sorted_array2.pop(0)
    index_third_highest = sorted_array2.pop(-2)  # third highest is now the second from the end after popping highest
    
    # Step 4: Rearrange arrays
    new_array1 = [highest, lowest, third_highest] + sorted_array1
    new_array2 = [index_highest, index_lowest, index_third_highest] + sorted_array2
    
    return new_array1, new_array2




# standard deviation
# Given points
x = np.array([3,18,2,25,35,41])
y = np.array([1,5,8,15,25,40])


# Calculate the mean point
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the distance of each point from the mean point
distances = np.sqrt((x - mean_x)**2 + (y - mean_y)**2)

# Calculate the mean distance
mean_distance = np.mean(distances)

# Calculate the standard deviation of the distances
std_dev_distance = np.std(distances)

# Tolerance factor (number of standard deviations considered acceptable)
tolerance_factor = 1.1 # For example, 1.5 times the standard deviation

# Determine the range within the specified tolerance factor
lower_bound = mean_distance - tolerance_factor * std_dev_distance
upper_bound = mean_distance + tolerance_factor * std_dev_distance

# Check for outliers
outliers = [(x[i], y[i]) for i in range(len(distances)) if distances[i] < lower_bound or distances[i] > upper_bound]

print("Mean Point:", (mean_x, mean_y))
print("Mean Distance:", mean_distance)
print("Standard Deviation of Distances:", std_dev_distance)
print(f"Range within {tolerance_factor} standard deviations:", (lower_bound, upper_bound))
print("Outliers:", outliers)

