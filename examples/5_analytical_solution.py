import numpy as np


def find_coefficients():
    """
    Step 1: Find the Average (Middle Point)
    First, we find the average number of rooms and the average price.
    This helps us understand where the "center" of our data is.

    Step 2: Find the Slope (How Much Price Changes with Rooms)
    We check how much price increases when the number of rooms increases.
    If more rooms mean much higher prices, the slope is steep.
    If the price changes just a little, the slope is gentle.

    Step 3: Find the Starting Price (When Rooms = 0)
    We calculate where the line crosses the price axis when there are zero rooms.
    This is called the intercept.

    Step 4: Build the Formula
    We put everything together to create a simple price = B0 + B1 × rooms equation.
    Now, we can predict prices for any number of rooms!
    """

    # Given data
    rooms = np.array([1, 2, 4])
    prices = np.array([10000, 20000, 40000])

    # Step 1: Find the Average (Middle Point)
    mean_x = np.mean(rooms)  # Average number of rooms
    mean_y = np.mean(prices)  # Average price

    # Step 2: Find the Slope (How Much Price Changes with Rooms)
    numerator = np.sum((rooms - mean_x) * (prices - mean_y))
    denominator = np.sum((rooms - mean_x) ** 2)
    slope = numerator / denominator  # Slope of the line

    # Step 3: Find the Starting Price (When Rooms = 0)
    intercept = mean_y - slope * mean_x  # Intercept of the line

    print(f"Estimated coefficients: b0 = {intercept:.2f}, b1 = {slope:.2f}")


if __name__ == '__main__':
    find_coefficients()
