import json
import matplotlib.pyplot as plt
import numpy as np

# Load original data from the Txt file
with open('/home/abrar/LLM_document_search/original_data.txt', 'r') as f:
    original_data_list = json.load(f)

# Load extracted data from the Txt file
with open('/home/abrar/LLM_document_search/extracted_data_gpt_3.5.txt', 'r') as f:
    extracted_data_list = json.load(f)

def calculate_accuracy(original, extracted):
    if len(original) != len(extracted):
        raise ValueError("The number of entries in the original and extracted content do not match.")

    total_elements = 0
    matching_elements = 0

    for original_row, extracted_row in zip(original, extracted):
        for key in original_row:
            total_elements += 1
            if original_row[key] == extracted_row.get(key):
                matching_elements += 1

    accuracy = (matching_elements / total_elements) * 100
    return accuracy

# Calculate accuracies for all papers
accuracies = []
paper_ids = []

for original_paper, extracted_paper in zip(original_data_list, extracted_data_list):
    paper_id = original_paper["paper_id"]
    num_tables = len(original_paper["tables"])
    paper_accuracy = 0

    for original_table, extracted_table in zip(original_paper["tables"], extracted_paper["tables"]):
        table_accuracy = calculate_accuracy(original_table, extracted_table)
        paper_accuracy += table_accuracy

    paper_accuracy /= num_tables
    accuracies.append(paper_accuracy)
    paper_ids.append(f'{paper_id} ({num_tables} table)')

print(accuracies)
print(np.mean(accuracies))
# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.bar(paper_ids, accuracies, color='skyblue')
plt.xlabel('Papers')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of content extraction inside the tabular data using gpt-3.5 without fine tuning prompt')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()
