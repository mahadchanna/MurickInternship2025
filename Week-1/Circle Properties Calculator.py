import math

try:
    radius = float(input("Enter the radius of the circle: "))

    if radius <= 0:
        print("Error: Radius must be a positive number.")
    else:
        area = math.pi * (radius ** 2)
        circumference = 2 * math.pi * radius
        print(f"Area of the circle: {area:.2f}")
        print(f"Circumference of the circle: {circumference:.2f}")

except ValueError:
    print("Error: Please enter a valid number for the radius.")
