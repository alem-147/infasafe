import pandas as pd
import matplotlib.pyplot as plt
import time

# Function to update and display the latest 100 values
def update_and_display_latest_100(csv_filename):
    while True:
        # Load the CSV file
        df = pd.read_csv(csv_filename)

        # Check if there are more than 100 rows
        if len(df) >= 100:
            # Sort the DataFrame by the timestamp column in descending order
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert Timestamp column to datetime format
            df.sort_values(by='Timestamp', ascending=False, inplace=True)

            # Take the latest 100 rows
            latest_100 = df.head(100)

            # Extract Timestamp and MaxVal columns
            timestamps = latest_100['Timestamp']
            max_vals = latest_100['MaxVal']

            # Create a line plot
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, max_vals, marker='o', linestyle='-', color='b')
            plt.xlabel('Timestamp')
            plt.ylabel('MaxVal')
            plt.title('Latest 100 MaxVal Points')
            plt.grid(True)

            # Rotate x-axis labels for better readability (optional)
            plt.xticks(rotation=45)

            # Show the plot
            plt.tight_layout()
            plt.show()

        # Wait for a few seconds before checking again
        time.sleep(10)

if __name__ == "__main__":
    csv_filename = "breath_data.csv"
    update_and_display_latest_100(csv_filename)
