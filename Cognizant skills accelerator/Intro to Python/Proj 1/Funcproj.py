import turtle

def factorial(n):
    """Recursive function to calculate the factorial of a number."""
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Recursive function to find the nth Fibonacci number."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def draw_fractal_tree(branch_length, t):
    """Recursive function to draw a fractal tree using turtle."""
    if branch_length > 5:
        t.forward(branch_length)
        t.right(20)
        draw_fractal_tree(branch_length - 15, t)
        t.left(40)
        draw_fractal_tree(branch_length - 15, t)
        t.right(20)
        t.backward(branch_length)

def fractal_tree():
    """Initialize turtle and draw a fractal tree."""
    screen = turtle.Screen()
    screen.bgcolor("white")
    t = turtle.Turtle()
    t.speed(0)
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    draw_fractal_tree(100, t)
    screen.mainloop()

def main():
    while True:
        print("\nRecursive Function Menu:")
        print("1. Calculate Factorial")
        print("2. Find nth Fibonacci Number")
        print("3. Draw a Fractal Tree (Bonus)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            while True:
                try:
                    num = int(input("Enter a positive integer to find its factorial: "))
                    if num < 0:
                        print("Please enter a non-negative integer.")
                    else:
                        print(f"The factorial of {num} is {factorial(num)}")
                        break
                except ValueError:
                    print("Invalid input! Please enter a valid number.")
        
        elif choice == '2':
            while True:
                try:
                    n = int(input("Enter the position of the Fibonacci number: "))
                    if n < 0:
                        print("Please enter a non-negative integer.")
                    else:
                        print(f"The {n}th Fibonacci number is {fibonacci(n)}")
                        break
                except ValueError:
                    print("Invalid input! Please enter a valid number.")
        
        elif choice == '3':
            print("Drawing a fractal tree...")
            fractal_tree()
        
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()