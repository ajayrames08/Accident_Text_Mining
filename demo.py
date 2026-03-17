## write a program to add two numbers

# Function to add two numbers
def add_numbers(num1, num2):
    return num1 + num2      

# Main function
if __name__ == "__main__":
    # Input: Get two numbers from the user
    number1 = float(input("Enter the first number: "))
    number2 = float(input("Enter the second number: "))

    # Process: Call the function to add the numbers
    result = add_numbers(number1, number2)

    # Output: Display the result
    print(f"The sum of {number1} and {number2} is: {result}")