import logging

# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def get_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a valid number.")
            logging.error("ValueError occurred: Invalid numeric input.")

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Oops! Division by zero is not allowed.")
        logging.error("ZeroDivisionError occurred: division by zero.")
        return None

def calculator():
    while True:
        print("\nWelcome to the Error-Free Calculator!")
        print("Choose an operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '5':
            print("Goodbye!")
            break
        
        if choice not in {'1', '2', '3', '4'}:
            print("Invalid choice! Please select a valid option.")
            continue
        
        num1 = get_number("Enter the first number: ")
        num2 = get_number("Enter the second number: ")
        
        if choice == '1':
            print(f"Result: {add(num1, num2)}")
        elif choice == '2':
            print(f"Result: {subtract(num1, num2)}")
        elif choice == '3':
            print(f"Result: {multiply(num1, num2)}")
        elif choice == '4':
            result = divide(num1, num2)
            if result is not None:
                print(f"Result: {result}")
        
        print("\n--------------------------------------")

if __name__ == "__main__":
    calculator()
