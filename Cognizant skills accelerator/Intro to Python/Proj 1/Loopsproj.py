import random

number_to_guess = random.randint(1, 10)

attempts = 0
num = int(input("Guess the number: "))

while num != number_to_guess :
    attempts += 1
    if num < number_to_guess:
        print("Too low!")
    elif num > number_to_guess:
        print("Too high!")

    num = int(input("Guess again: "))

attempts += 1
print(f"You got it! The answer is {number_to_guess}! You guessed it in {attempts} attempts!")