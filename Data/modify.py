import csv

# Function to read the CSV file, process the data, and write to a new CSV file
def process_csv(input_file, output_file):
    updated_data = []
    
    # Read the input CSV file
    with open(input_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Add 20 to the first column and convert to float
            updated_row = [float(row[0]) + 1., float(row[1])]
            updated_data.append(updated_row)
    
    # Write the updated data to the new CSV file
    with open(output_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(updated_data)

# Example usage
input_csv = 'filtered_velocity_140_good_one.csv'
output_csv = 'filtered_velocity_modified.csv'
process_csv(input_csv, output_csv)
