import random

# Generate a list containing 1 random number
random_number = random.randint(0, 50)

my_arr = []

x = 0

while x < 6:  # Loop 6 times to generate 6 numbers
    my_arr.append(random_number + x)
    x += 1

# Calculate the next value
next_value = my_arr[5] + 1

# Print and save to text file
with open('C:/Users/Administrator/func pred/function_prediction/model training sets/data.txt', 'a') as file:
    file.write(f"{my_arr}\n")
    file.write(f"{next_value}\n\n")

print("Sequential: ", my_arr)
print("The next value: ", next_value)

# Calculate squared array
squared_arr = [x ** 2 for x in my_arr]

# Calculate the next value for squared array
next_value_squared = next_value ** 2

# Print and save to text file
with open('C:/Users/Administrator/func pred/function_prediction/model training sets/data.txt', 'a') as file:
    file.write(f"{squared_arr}\n")
    file.write(f"{next_value_squared}\n\n")

print("Quad: ", squared_arr)
print("The next value: ", next_value_squared)

# Generate two random numbers for alternating array
rand1 = random.randint(0, 500)
rand2 = random.randint(0, 500)

# Create the alternating array
final = [rand1, rand2, rand1, rand2, rand1, rand2]

# Print and save to text file
with open('C:/Users/Administrator/func pred/function_prediction/model training sets/data.txt', 'a') as file:
    file.write(f"{final}\n")
    file.write(f"{rand1}\n\n")

print("Alternating: ", final)
print("The next value: ", rand1)
