#task 1
def greet_user(name):
    print(f"hello, {name}! Welcome aboard.")
def add_numbers(x, y):
    addition = x + y
    print(f"The sum of {x} and {y} is {addition}")

greet_user("Shah")
add_numbers(5, 6)

#task 2

def describe_pet(pet_name, animal_type = "dog"):
    print(f"I have a {animal_type} named {pet_name}")

describe_pet("Spike",)

#task 3

def make_sandwich(*args):
    for arg in args:
        print(arg)

print("Making a sandwich with the following ingredients:")
make_sandwich("Lettuce", "Tomato", "Cheese")

#task 4

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

# Testing the functions
print("Factorial of 5:", factorial(5))
print("10th Fibonacci number:", fibonacci(10))