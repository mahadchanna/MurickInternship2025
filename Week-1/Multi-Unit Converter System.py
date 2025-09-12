def convert_temperature(value, from_unit, to_unit):

    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5/9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        raise ValueError("Invalid temperature unit")

    if to_unit == "C":
        return celsius
    elif to_unit == "F":
        return (celsius * 9/5) + 32
    elif to_unit == "K":
        return celsius + 273.15
    else:
        raise ValueError("Invalid temperature unit")

def convert_length(value, from_unit, to_unit):

    to_meters = {
        "m": 1,
        "ft": 0.3048,
        "in": 0.0254,
        "km": 1000,
        "mi": 1609.34
    }

    if from_unit not in to_meters or to_unit not in to_meters:
        raise ValueError("Invalid length unit")

    meters = value * to_meters[from_unit]
    return meters / to_meters[to_unit]

def main():
    while True:
        print("\n Unit Converter Menu")
        print("1. Temperature Conversion (C/F/K)")
        print("2. Length Conversion (m, ft, in, km, mi)")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            print("\nTemperature Units: C = Celsius, F = Fahrenheit, K = Kelvin")
            try:
                value = float(input("Enter value: "))
                from_unit = input("From unit (C/F/K): ").strip().upper()
                to_unit = input("To unit (C/F/K): ").strip().upper()
                result = convert_temperature(value, from_unit, to_unit)
                print(f"{value} {from_unit} = {result:.2f} {to_unit}")
            except Exception as e:
                print("Error:", e)

        elif choice == "2":
            print("\nLength Units: m = meters, ft = feet, in = inches, km = kilometers, mi = miles")
            try:
                value = float(input("Enter value: "))
                from_unit = input("From unit (m/ft/in/km/mi): ").strip().lower()
                to_unit = input("To unit (m/ft/in/km/mi): ").strip().lower()
                result = convert_length(value, from_unit, to_unit)
                print(f"{value} {from_unit} = {result:.4f} {to_unit}")
            except Exception as e:
                print("Error:", e)

        elif choice == "3":
            print("Exiting... ")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
