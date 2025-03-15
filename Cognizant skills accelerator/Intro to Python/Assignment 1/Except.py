#task 1

def divide_by_number():
    while True:
        try:
            num = int(input("Enter a number: "))
            result = 100 / num
            print(f"100 divided by {num} is {result}")
            break
        except ZeroDivisionError:
            print("Oops! You cannot divide by zero.")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

divide_by_number()

#task 2

try:
    my_list = [1, 2, 3]
    print(my_list[5])  # IndexError
except IndexError:
    print("IndexError occurred! List index out of range.")

try:
    my_dict = {"a": 1, "b": 2}
    print(my_dict["c"])  # KeyError
except KeyError:
    print("KeyError occurred! Key not found in the dictionary.")

try:
    result = "hello" + 5  # TypeError
except TypeError:
    print("TypeError occurred! Unsupported operand types.")

#task 3

def safe_division():
    try:
        a = int(input("Enter the first number: "))
        b = int(input("Enter the second number: "))
        result = a / b
    except ZeroDivisionError:
        print("Oops! You cannot divide by zero.")
    except ValueError:
        print("Invalid input! Please enter valid numbers.")
    else:
        print(f"The result is {result}")
    finally:
        print("This block always executes.")

safe_division()