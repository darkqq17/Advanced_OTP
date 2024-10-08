import json
import ast

# Replace 'input_data.json' with the path to your input JSON file
input_file = 'gpt4o_output.json'
output_file = 'output.json'

with open(input_file, 'r', encoding='utf-8') as infile:
    # Read the file content as a string
    file_content = infile.read()

# Use ast.literal_eval to parse the content into Python data structures
try:
    input_data = ast.literal_eval(file_content)
except Exception as e:
    print(f"Error parsing input data: {e}")
    exit(1)

# Now, convert tuples to lists in the 'output' field
converted_data = []
for item in input_data:
    input_text = item.get('input', '')
    output_list = item.get('output', [])
    converted_output = []
    for entry in output_list:
        if isinstance(entry, tuple):
            converted_output.append(list(entry))
        else:
            converted_output.append(entry)
    converted_data.append({
        'input': input_text,
        'output': converted_output
    })

# Write the converted data to a JSON file
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(converted_data, outfile, indent=2)

print(f"Data has been converted and saved to '{output_file}'.")


# # Replace 'input_data.json' with the path to your input JSON file
# input_file = 'gpt4o_output.json'
# output_file = 'output.json'

# # Function to convert tuples in 'output' to lists
# def convert_tuples_to_lists(data):
#     converted_data = []
#     for item in data:
#         input_text = item.get('input', '')
#         output_list = item.get('output', [])
#         # Convert each tuple in the output list to a list
#         converted_output = []
#         for entry in output_list:
#             if isinstance(entry, tuple):
#                 converted_output.append(list(entry))
#             elif isinstance(entry, list):
#                 # Already a list, no need to convert
#                 converted_output.append(entry)
#             else:
#                 print(f"Warning: Unexpected output entry format: {entry}")
#         converted_item = {
#             'input': input_text,
#             'output': converted_output
#         }
#         converted_data.append(converted_item)
#     return converted_data

# # Read your input data from the JSON file
# with open(input_file, 'r', encoding='utf-8') as infile:
#     input_data = json.load(infile)

# # Convert the data
# converted_data = convert_tuples_to_lists(input_data)

# # Write the converted data to a JSON file
# with open(output_file, 'w', encoding='utf-8') as outfile:
#     json.dump(converted_data, outfile, indent=2)

# print(f"Data has been converted and saved to '{output_file}'.")
