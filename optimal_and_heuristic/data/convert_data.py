import pandas as pd

import os
os.chdir(r"C:\Phd\CUDA test\Test\test 1\EV\EV_delivery\code\code_JD_data")

# File path
file_path = 'data/jd200_1.txt'

# Read the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Locate the DISTANCETIME_SECTION
start = None
for i, line in enumerate(lines):
    if line.strip() == "DISTANCETIME_SECTION":
        start = i + 1
        break

# Extract data starting from the DISTANCETIME_SECTION
distance_data = []
time_data = []
for line in lines[start:]:
    if line.strip() == "":  # Stop at empty line or end of file
        break
    if line.startswith("ID"):  # Skip header
        continue
    parts = line.strip().split(',')
    if len(parts) == 5:
        _, from_node, to_node, distance, spend_tm = parts
        distance_data.append((int(from_node), int(to_node), float(distance)))
        time_data.append((int(from_node), int(to_node), float(spend_tm)))

# Create a distance matrix DataFrame
nodes = sorted(set([i for i, _, _ in distance_data] + [j for _, j, _ in distance_data]))
distance_matrix = pd.DataFrame(index=nodes, columns=nodes, data=float('inf'))
time_matrix = pd.DataFrame(index=nodes, columns=nodes, data=float('inf'))

# Fill the distance matrix
for from_node, to_node, distance in distance_data:
    distance_matrix.loc[from_node, to_node] = distance
    
# Fill the time matrix
for from_node, to_node, spend_tm in time_data:
    time_matrix.loc[from_node, to_node] = spend_tm

# Fill diagonal with 0
for node in nodes:
    distance_matrix.loc[node, node] = 0
    time_matrix.loc[node,node] = 0

# Save to CSV
csv_path = 'data/distance_matrix_jd200_1.csv'
distance_matrix.to_csv(csv_path)

csv_path2 = 'data/time_matrix_jd200_1.csv'
time_matrix.to_csv(csv_path2)






# Locate the NODE_SECTION
node_start = None
for i, line in enumerate(lines):
    if line.strip() == "NODE_SECTION":
        node_start = i + 1
        break

# Extract node data
node_data = []
headers = []
for i, line in enumerate(lines[node_start:], start=node_start):
    if line.strip() == "":
        break
    if i == node_start:
        headers = line.strip().split(',')
        continue
    parts = line.strip().split(',')
    if len(parts) == len(headers):
        node_data.append(parts)

# Create a DataFrame with selected columns
df_nodes = pd.DataFrame(node_data, columns=headers)
selected_columns = ["ID", "type", "lng", "lat", "first_receive_tm", "last_receive_tm", "service_time"]
df_selected_nodes = df_nodes[selected_columns]

# Save to CSV
node_csv_path = "data/combined_data_jd200_1.csv"
df_selected_nodes.to_csv(node_csv_path, index=False)