import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

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

def get_accuracies(original_data_list, extracted_data_list):
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
    
    return paper_ids, accuracies

# Load original data
original_data_path = '/home/abrar/Accuracy_comparision/original_data.txt'
original_data_list = load_data(original_data_path)

# Load extracted data for each model
gpt_35_no_finetune_path = 'extracted_data_gpt-3.5-turbo-16k.json'
gpt_35_finetune_path = 'extracted_data_gpt3.5_with_fine_tune_prompt.json'
gpt_4_path = 'extracted_data_gpt_4o.txt'

extracted_data_35_no_finetune = load_data(gpt_35_no_finetune_path)
extracted_data_35_finetune = load_data(gpt_35_finetune_path)
extracted_data_4 = load_data(gpt_4_path)

# Calculate accuracies for each model
paper_ids, accuracies_gpt35_no_finetune = get_accuracies(original_data_list, extracted_data_35_no_finetune)
_, accuracies_gpt35_finetune = get_accuracies(original_data_list, extracted_data_35_finetune)
_, accuracies_gtp4o = get_accuracies(original_data_list, extracted_data_4)

# Calculate mean accuracies
mean_accuracy_35_no_finetune = np.mean(accuracies_gpt35_no_finetune)
mean_accuracy_35_finetune = np.mean(accuracies_gpt35_finetune)
mean_accuracy_4 = np.mean(accuracies_gtp4o)

# Plotting the accuracies
x = np.arange(len(paper_ids))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

rects1 = ax.bar(x - width, accuracies_gpt35_no_finetune, width, label=f'GPT-3.5 without finetuning\nMean: {mean_accuracy_35_no_finetune:.2f}%', color='skyblue')
rects2 = ax.bar(x, accuracies_gpt35_finetune, width, label=f'GPT-3.5 with finetuning\nMean: {mean_accuracy_35_finetune:.2f}%', color='lightgreen')
rects3 = ax.bar(x + width, accuracies_gtp4o, width, label=f'GPT-4.0\nMean: {mean_accuracy_4:.2f}%', color='salmon')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Papers')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy in Extracting Tabular Data vs. Original PDF Content')
ax.set_xticks(x)
ax.set_xticklabels(paper_ids, rotation=45, ha='right')
ax.set_ylim(0, 100)

# # Attach a text label above each bar in rects, displaying its height.
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(round(height, 2)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# Place the legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

fig.tight_layout()

# Save the plot
plt.savefig("accuracy_comparison.png", bbox_inches='tight')
fig.tight_layout()


