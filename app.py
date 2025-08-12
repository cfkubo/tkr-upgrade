import gradio as gr
import pandas as pd
import re
from collections import deque
import io
import os # Import the os module for path operations

# Helper function to extract cleaned version numbers from the raw strings
def extract_version(version_string):
    """
    Extracts the clean version number (e.g., '1.27.10') from a version string.
    Handles various formats like 'vSphere Kubernetes releases X.Y.Z (TKr A.B.C ...)'.
    """
    # Regex to find version numbers like X.Y.Z or X.Y.Z.A
    match = re.search(r'(\d+\.\d+\.\d+(?:\.\d+)?)', version_string)
    if match:
        return match.group(1)
    return version_string # Fallback: return original string if no match

# Pathfinding function using Breadth-First Search (BFS)
def find_shortest_path(graph, start, end):
    """
    Finds the shortest path between a start and end node in a graph using BFS.
    Returns a list of versions representing the path, or None if no path exists.
    """
    # If the start and end versions are the same, the path is just the start version itself.
    if start == end:
        return [start]

    # Initialize a queue for BFS with the starting node and its path
    queue = deque([(start, [start])])
    # Keep track of visited nodes to avoid cycles and redundant processing
    visited = {start}

    while queue:
        current_node, path = queue.popleft()

        # Check if the current_node exists in the graph keys to prevent errors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    # If the neighbor is the target end node, we've found the shortest path
                    if neighbor == end:
                        return new_path
                    # Otherwise, add the neighbor to the queue for further exploration
                    queue.append((neighbor, new_path))
    return None # Return None if no path is found after exploring all reachable nodes

# Global variables to store the processed data, accessible by all Gradio functions
COMPATIBILITY_MATRIX = None
UPGRADE_GRAPH = None
ALL_VERSIONS = []

def process_csv_data(file_object):
    """
    Reads the uploaded CSV file, preprocesses it to extract compatibility data,
    and builds the global compatibility matrix and upgrade graph.
    This function is triggered when the user uploads a file or clicks "Process File".
    """
    global COMPATIBILITY_MATRIX, UPGRADE_GRAPH, ALL_VERSIONS

    file_path = None
    status_msg = ""

    if file_object is None:
        # If no file is uploaded, try to load from the default location
        default_file_name = "Upgrade.csv"
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_file_path = os.path.join(script_dir, default_file_name)

        if os.path.exists(default_file_path):
            file_path = default_file_path
            status_msg = f"Attempting to load '{default_file_name}' from the current directory."
        else:
            status_msg = "No file uploaded and 'Upgrade.csv' not found in the current directory. Please upload a file."
            return (
                gr.update(choices=[], interactive=False, value=None),
                gr.update(choices=[], interactive=False, value=None),
                status_msg
            )
    else:
        # If a file is uploaded, use its path
        file_path = file_object.name
        status_msg = "Processing uploaded file."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.strip().split('\n')

        # Ensure there are enough lines for headers and some data
        if len(lines) < 4:
            raise ValueError("CSV file is too short or not in the expected format. Expected at least 4 lines.")

        # Line 3 (index 2 in 0-based list) contains column headers
        # Remove leading comma if present, then split by comma, clean by stripping quotes
        col_header_parts = [h.strip().strip('"') for h in lines[2].split(',')]
        # Filter out the first empty part (due to leading comma) if it exists and get actual headers
        col_headers_raw_full = [part for part in col_header_parts if part and part != '']
        
        if not col_headers_raw_full:
            raise ValueError("Could not parse column headers from the third line of the CSV.")
        
        # Data rows start from line 4 (index 3 in 0-based list)
        data_rows = lines[3:]
        
        row_headers_raw_full = []
        parsed_data = [] # To store the boolean compatibility values

        for line in data_rows:
            parts = [p.strip().strip('"') for p in line.split(',')]
            if parts and parts[0]: # Ensure the row has content and a row header
                row_headers_raw_full.append(parts[0])
                # Compatibility data starts from the second element in the split parts for this row
                # We need to map these to the exact column headers.
                row_compatibility = []
                # Iterate through the expected columns and check corresponding data part
                # Assuming parts[1:] aligns with col_headers_raw_full
                for j in range(len(col_headers_raw_full)):
                    if j + 1 < len(parts): # +1 because parts[0] is row header
                        row_compatibility.append(parts[j+1] == 'Compatible')
                    else:
                        row_compatibility.append(False) # Default to False if data is missing
                parsed_data.append(row_compatibility)
            # Handle cases where a row might be empty or just a comma, skip it
            elif line.strip(): # If line is not completely empty, but parsing failed
                print(f"Skipping malformed data row: '{line}'")


        if not row_headers_raw_full:
            raise ValueError("Could not parse row headers or data from the fourth line onwards. Is the data section empty?")

        # Use raw headers for DataFrame index and columns to avoid ambiguity issues
        COMPATIBILITY_MATRIX = pd.DataFrame(parsed_data, index=row_headers_raw_full, columns=col_headers_raw_full)

        # Build the UPGRADE_GRAPH using cleaned versions
        # First, gather all unique cleaned versions for the graph nodes
        all_unique_cleaned_versions = set()
        for h in row_headers_raw_full:
            all_unique_cleaned_versions.add(extract_version(h))
        for h in col_headers_raw_full:
            all_unique_cleaned_versions.add(extract_version(h))
        
        UPGRADE_GRAPH = {v: [] for v in all_unique_cleaned_versions}

        # Populate the graph by iterating through the COMPATIBILITY_MATRIX (using raw keys)
        for current_raw_ver in COMPATIBILITY_MATRIX.index:
            current_clean_ver = extract_version(current_raw_ver)
            for target_raw_ver in COMPATIBILITY_MATRIX.columns:
                target_clean_ver = extract_version(target_raw_ver)
                if COMPATIBILITY_MATRIX.loc[current_raw_ver, target_raw_ver]:
                    if target_clean_ver not in UPGRADE_GRAPH[current_clean_ver]:
                        UPGRADE_GRAPH[current_clean_ver].append(target_clean_ver)

        ALL_VERSIONS = sorted(list(all_unique_cleaned_versions))
        status_msg = f"Data from '{os.path.basename(file_path)}' loaded successfully! Please select your current and target versions."

    except Exception as e:
        status_msg = f"Error processing file: {e}. Please ensure it's a valid CSV with expected matrix format."
        return (
            gr.update(choices=[], interactive=False, value=None),
            gr.update(choices=[], interactive=False, value=None),
            status_msg
        )

    # Update Gradio dropdowns with available versions and enable them
    return (
        gr.update(choices=ALL_VERSIONS, interactive=True, value=None),
        gr.update(choices=ALL_VERSIONS, interactive=True, value=None),
        status_msg
    )

def find_upgrade_path_ui(current_version, target_version):
    """
    Gradio UI function to find and display the upgrade path based on user selections.
    """
    if COMPATIBILITY_MATRIX is None:
        return "Please upload and load the data first by clicking 'Process File & Populate Versions'.", pd.DataFrame()

    if not current_version or not target_version:
        return "Please select both Current Version and Target Version.", pd.DataFrame()

    # The dropdowns provide cleaned versions, so these should exist in ALL_VERSIONS
    if current_version not in ALL_VERSIONS or target_version not in ALL_VERSIONS:
        return "One or both selected versions are not valid. Please ensure they are from the loaded data.", pd.DataFrame()

    # Find the shortest upgrade path using the pre-built graph
    path = find_shortest_path(UPGRADE_GRAPH, current_version, target_version)

    if path:
        path_str = f"Upgrade path found: {' -> '.join(path)} ✨"

        # Create a detailed DataFrame showing each direct step in the path
        path_df_data = []
        path_columns = ["From Version", "To Version", "Compatible?"]

        # Iterate through the path to get each direct upgrade step
        for i in range(len(path) - 1):
            from_version_clean = path[i]
            to_version_clean = path[i+1]
            
            # Check direct compatibility using the global COMPATIBILITY_MATRIX
            is_compatible_step = False
            
            # Find all raw versions that clean to from_version_clean
            potential_raw_from_versions = [
                raw for raw in COMPATIBILITY_MATRIX.index
                if extract_version(raw) == from_version_clean
            ]
            # Find all raw versions that clean to to_version_clean
            potential_raw_to_versions = [
                raw for raw in COMPATIBILITY_MATRIX.columns
                if extract_version(raw) == to_version_clean
            ]

            # Check if any raw_from -> raw_to pair is compatible in the matrix
            for raw_from in potential_raw_from_versions:
                for raw_to in potential_raw_to_versions:
                    if raw_from in COMPATIBILITY_MATRIX.index and raw_to in COMPATIBILITY_MATRIX.columns:
                        if COMPATIBILITY_MATRIX.loc[raw_from, raw_to]:
                            is_compatible_step = True
                            break # Found a compatible path, no need to check other raw_to
                if is_compatible_step:
                    break # Found a compatible path, no need to check other raw_from
            
            # Add the row data for this step
            path_df_data.append([
                from_version_clean,
                to_version_clean,
                "Yes" if is_compatible_step else "No" # Explicitly show "Yes" or "No"
            ])

        # Create the pandas DataFrame that will be displayed in Gradio
        path_display_df = pd.DataFrame(path_df_data, columns=path_columns)

        return path_str, path_display_df
    else:
        return "No upgrade path found between the selected versions. Please check compatibility data.", pd.DataFrame()


# --- Gradio User Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
    """
    # Tanzu Kubernetes Upgrade Path Finder 🚀
    Welcome! This interactive tool helps you visualize the upgrade paths for Tanzu Kubernetes releases.
    Simply upload your `Upgrade.csv` file, or if it's in the same directory, it will attempt to load automatically!
    """
    )

    with gr.Row():
        # File upload component - now optional for default load
        file_input = gr.File(label="Upload Upgrade.csv File (Optional)", type="filepath", file_count="single")
        # Status message textbox for user feedback
        status_message = gr.Textbox(label="Status", interactive=False, value="Attempting to load default file or upload yours.")

    with gr.Row():
        # Dropdown for selecting the current Kubernetes version
        current_version_dropdown = gr.Dropdown(
            label="Current Version",
            choices=[], # Initially empty, populated after file upload
            interactive=False # Disabled until file is processed
        )
        # Dropdown for selecting the target Kubernetes version
        target_version_dropdown = gr.Dropdown(
            label="Target Version",
            choices=[], # Initially empty, populated after file upload
            interactive=False # Disabled until file is processed
        )

    with gr.Row():
        # Button to trigger file processing and populate dropdowns
        # This button is now useful even if no file is explicitly uploaded, to trigger default load attempt
        load_button = gr.Button("Process File & Populate Versions ✨", variant="primary")
        # Button to find and display the upgrade path
        find_path_button = gr.Button("Find Upgrade Path ➡️", variant="secondary")

    # Textbox to display the summary of the found upgrade path
    output_text = gr.Textbox(label="Upgrade Path Summary", interactive=False, lines=2)
    # DataFrame to display the detailed compatibility for each step in the path
    output_dataframe = gr.DataFrame(label="Path Compatibility Details", wrap=True)

    # --- Event Handlers for Gradio Components ---
    # When a file is uploaded, process it and update dropdowns and status message
    file_input.upload(
        process_csv_data,
        inputs=file_input,
        outputs=[current_version_dropdown, target_version_dropdown, status_message]
    )
    # When the "Process File" button is clicked, do the same as file upload, passing None if no file input
    # This initial call without file_input will trigger the default file load attempt
    load_button.click(
        process_csv_data,
        inputs=file_input, # Pass the file_input value to the function
        outputs=[current_version_dropdown, target_version_dropdown, status_message]
    )
    # When the "Find Upgrade Path" button is clicked, find and display the path
    find_path_button.click(
        find_upgrade_path_ui,
        inputs=[current_version_dropdown, target_version_dropdown],
        outputs=[output_text, output_dataframe]
    )

    # Automatically trigger data processing on app load
    demo.load(
        process_csv_data,
        inputs=None, # Pass None initially to try loading default file
        outputs=[current_version_dropdown, target_version_dropdown, status_message]
    )

# Launch the Gradio application
demo.launch()
