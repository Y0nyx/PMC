import csv
import os
from pipeline.CsvResultRow import CsvResultRow
from common.Constants import *

class CsvManager:
    def __init__(self) -> None:
        self.check_headers()

    def add_new_row(self, row2write: CsvResultRow):
        # Determine the next ID by reading the CSV file and finding the max ID
        next_id = 1
        with open(CSV_FILE_NAME, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                current_id = int(row['id'])
                if current_id >= next_id:
                    next_id = current_id + 1
    
        # Assuming CsvResultRow has a method set_id to set the id field
        row2write.set_id(next_id)

        # Writing to CSV file
        with open(CSV_FILE_NAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL, quotechar='"')
            writer.writerow(row2write.get_writable_row()) 


    def check_headers(self):
        # Check if the file exists
        if not os.path.isfile(CSV_FILE_NAME):
            # Create the file with headers if it does not exist
            with open(CSV_FILE_NAME, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
                writer.writeheader()
            print(f"{CSV_FILE_NAME} has been created with headers.")
        else:
            # Read the CSV file and check for missing fields
            with open(CSV_FILE_NAME, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                headers = reader.fieldnames
                missing_fields = [field for field in CSV_FIELDS if field not in headers]

            # If there are missing fields, add them
            if missing_fields:
                with open(CSV_FILE_NAME, 'r', newline='') as csvfile:
                    rows = list(csv.DictReader(csvfile))

                # Add missing fields to each row
                for row in rows:
                    for field in missing_fields:
                        row[field] = '' # Or any default value you prefer

                # Write the updated rows back to the file
                with open(CSV_FILE_NAME, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

                print(f"Missing fields {missing_fields} have been added to {CSV_FILE_NAME}")
            else:
                print("All expected fields are present in the CSV file.")
