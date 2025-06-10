"""
## UNVEILING SARCASM THROUGH SPEECH: A CNN-BASED APPROACH TO VOCAL FEATURE ANALYSIS ##

This Python file is part of the project undertaken in the course "Deep Learning for Audio Data"
during the winter semester 2023/24 at the Technische Universität Berlin, within the curriculum
of Audio Communication and Technologies M.Sc.

Authors (Group 1):
- Raphaël G. Gillioz
- Blanca Sabater Vilchez
- Pierre S.F. Kolingba-Froidevaux
- Florian Morgner

Description:
Small script to verify the data distribution of the dataset

Direct usage:
python src/data/data_sarcasm_check.py

Date: 29.03.2024

"""

import json

# Define the filename
filename = "data/raw/MUStARD/sarcasm_data.json"

# Initialize a counter
sarcasm_true_count = 0
sarcasm_false_count = 0

# Open the JSON file
with open(filename, "r") as f:
    # Load the JSON data
    data = json.load(f)

    # Recursively search for the "sarcasm" parameter
    def count_sarcasm(data):
        global sarcasm_true_count
        global sarcasm_false_count
        if isinstance(data, dict):
            if "sarcasm" in data and data["sarcasm"] == True:
                sarcasm_true_count += 1
            if "sarcasm" in data and data["sarcasm"] == False:
                sarcasm_false_count += 1
            for key, value in data.items():
                count_sarcasm(value)
        elif isinstance(data, list):
            for item in data:
                count_sarcasm(item)

    # Call the recursive function
    count_sarcasm(data)

# Print the count
print(f"The parameter 'sarcasm' is set to 'true' {sarcasm_true_count} times and 'false' {sarcasm_false_count} times.")
