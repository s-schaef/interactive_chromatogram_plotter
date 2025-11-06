import io
import re
import math
import pandas as pd
import streamlit as st
from itertools import cycle
import matplotlib.pyplot as plt

### Function definitions ###

def generate_plots(
    data_dict,
    custom_names,
    x_data_dict,
    plot_configs,
    ylabels_per_sample,            # NEW: dict sample -> ylabel like 'Signal (pA)'
    external_label=False,
    custom_legend=None,
    suptitle_enabled=True,
    suptitle="Formulation",
    supaxes_enabled=False,
    log_y=False,
):
    """Generate matplotlib plots based on configuration."""
    # Filter out empty plot configs
    valid_configs = [config for config in plot_configs if config.get('files')]
    if not valid_configs:
        return None

    # Handle subplot layout
    if len(valid_configs) in [1, 2, 3]:
        fig, axs = plt.subplots(
            1, len(valid_configs),
            figsize=(5 * len(valid_configs), 5),
            squeeze=False,
            sharey=supaxes_enabled, sharex=supaxes_enabled
        )
    elif len(valid_configs) == 4:
        fig, axs = plt.subplots(
            2, 2, figsize=(10, 10),
            squeeze=False,
            sharey=supaxes_enabled, sharex=supaxes_enabled
        )
    else:
        fig, axs = plt.subplots(
            math.ceil(len(valid_configs) / 3), 3,
            figsize=(15, 10), squeeze=False,
            sharey=supaxes_enabled, sharex=supaxes_enabled
        )
    axs = axs.flat

    # Remove unused axes for non-rectangular layouts
    total_subplots = len(axs)
    for i in range(len(valid_configs), total_subplots):
        axs[i].set_visible(False)

    subplot_ylabels = []    # keep computed ylabels per subplot
    any_subplot_mixed = False

    for i, config in enumerate(valid_configs):
        ax = axs[i]

        # Plot traces
        if external_label:
            for filename in config['files']:
                if filename in data_dict:
                    color = st.session_state.get(f"color_{i}_{filename}")
                    ax.plot(x_data_dict[filename], data_dict[filename], color=color)
        else:
            for filename in config['files']:
                if filename in data_dict:
                    color = st.session_state.get(f"color_{i}_{filename}")
                    ax.plot(
                        x_data_dict[filename],
                        data_dict[filename],
                        label=custom_names.get(filename, filename),
                        color=color
                    )
            ax.legend(loc="upper left", fontsize='small')

        if log_y:
            ax.set_yscale('log')

        ax.set_title(config.get('title', f'Plot {i+1}'))

        # --- Determine ylabel for this subplot ---
        used = [f for f in config['files'] if f in data_dict]
        used_ylabels = { ylabels_per_sample.get(f, "Signal") for f in used }

        if len(used_ylabels) == 1:
            subplot_ylabel = next(iter(used_ylabels))
        else:
            subplot_ylabel = "Signal"
            any_subplot_mixed = True
            if used:  # warn only if there are traces
                details = ", ".join(sorted(used_ylabels))
                st.warning(f"Plot {i+1}: mixed units detected ({details}). Using generic 'Signal' y-label.")

        subplot_ylabels.append(subplot_ylabel)

    # --- Axes labels: common vs per-axis ---
    if not supaxes_enabled:
        # Per subplot labels
        for i, ax in enumerate(axs[:len(valid_configs)]):
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(subplot_ylabels[i])
    else:
        # Common x and y labels for the whole figure
        fig.supxlabel("Time (min)")

        # Gather ALL used ylabels across the figure
        all_used_files = []
        for config in valid_configs:
            all_used_files.extend([f for f in config['files'] if f in data_dict])
        all_used_labels = { ylabels_per_sample.get(f, "Signal") for f in all_used_files }

        if len(all_used_labels) == 1:
            common_ylabel = next(iter(all_used_labels))
        else:
            common_ylabel = "Signal"
            any_subplot_mixed = True
            if all_used_files:
                details = ", ".join(sorted(all_used_labels))
                st.warning(f"Common axes: mixed units detected across subplots ({details}). Using generic 'Signal' y-label.")
        fig.supylabel(common_ylabel)

    # --- Suptitle handling ---
    if suptitle_enabled:
        if any_subplot_mixed:
            fig.suptitle("Signal", fontsize=16)
            st.warning("Suptitle set to 'Signal' because mixed units were detected.")
        else:
            if suptitle:
                fig.suptitle(suptitle, fontsize=16)

    plt.tight_layout()
    return fig

def _extract_unit_from_token(token: str) -> str | None:
    """
    Extract a unit from a token like 'Signal (pA)' or 'pA'.
    Returns 'pA', 'mAU', etc., or None if not found.
    """
    if token is None:
        return None
    token = str(token).strip()

    # Prefer content inside parentheses
    m = re.search(r"\(([^)]+)\)", token)
    if m:
        return m.group(1).strip()

    # Otherwise, use the token itself if it's not just 'Signal' or 'Time'
    cleaned = re.sub(r"(?i)\b(signal|time)\b", "", token).strip()
    if cleaned:
        return cleaned
    return None

def _unit_from_ylabel(ylabel: str) -> str | None:
    """
    Given 'Signal (pA)' -> 'pA', 'Signal (mAU)' -> 'mAU', or None if missing.
    """
    return _extract_unit_from_token(ylabel)


def process_txt_file(uploaded_file, header_row_index: int = 42):
    """
    Process a Chromeleon exported .txt file and derive ylabel from the unit row.
    
    Returns:
        df_subset (pd.DataFrame): 2 columns [Time, Signal]
        default_name (str|None): default sample name extracted from header (if present)
        ylabel (str): e.g., 'Signal (pA)' or 'Signal (mAU)'
        error (str|None): error message if any, else None
    """
    # 200MB limit
    if uploaded_file.size > 200 * 1024 * 1024:
        return None, None, None, "File size exceeds 200MB limit."
    
    try:
        # Read full file content once (so we can inspect header lines)
        raw_bytes = uploaded_file.read()
        text = raw_bytes.decode('utf-8', errors='ignore')
        lines = text.splitlines()

        # Extract default name from 6th line if available and starts with "Injection"
        default_name = None
        if len(lines) >= 6:
            sixth_line_parts = lines[5].strip().split('\t')
            if len(sixth_line_parts) >= 2 and sixth_line_parts[0].strip() == "Injection":
                default_name = sixth_line_parts[-1].strip()

        # Reset pointer for pandas to read
        uploaded_file.seek(0)

        # Read the data, using the known header row
        df = pd.read_csv(
            uploaded_file, sep='\t', header=header_row_index,
            thousands=',', engine='python'
        )
        if len(df.columns) != 3:
            return None, None, None, "File should have exactly 3 columns."
        
        # Try to get unit from the 3rd column header first
        third_header = str(df.columns[2])
        unit = _extract_unit_from_token(third_header)

        # If that didn't yield a unit, try the units row:
        # The units row is the row immediately BELOW the header (i.e., above the data)
        if unit is None:
            units_row_idx = header_row_index + 1
            if len(lines) > units_row_idx:
                units_row = lines[units_row_idx].strip().split('\t')
                if len(units_row) >= 3:
                    unit = _extract_unit_from_token(units_row[2])

        ylabel = f"Signal ({unit})" if unit else "Signal"

        # Keep only the first and third columns (Time and Signal)
        df_subset = df.iloc[:, [0, 2]]

        return df_subset, default_name, ylabel, None

    except Exception as e:
        return None, None, None, (
            f"Error processing file: {str(e)}\n"
            "Please ensure the file is a valid Chromeleon exported .txt file."
        )


def process_csv_file(uploaded_file):
    """
    Process a CSV file exported from this app. It supports multiple units
    for the signal (e.g., 'pA - Sample', 'mAU - Sample') and returns a ylabel
    for each sample.

    Returns:
        data_dict (dict[str, pd.Series]): sample -> y-values
        x_data_dict (dict[str, pd.Series]): sample -> time values
        custom_names (dict[str, str]): sample -> display name
        ylabels (dict[str, str]): sample -> ylabel (e.g., 'Signal (pA)')
        error (str|None)
    """
    # 200MB limit
    if uploaded_file.size > 200 * 1024 * 1024:
        return None, None, None, None, "File size exceeds 200MB limit."

    try:
        df = pd.read_csv(uploaded_file)
        data_dict = {}
        x_data_dict = {}
        custom_names = {}
        ylabels = {}

        for column in df.columns:
            col = str(column)

            # Time columns: "Time - Sample"
            if col.startswith("Time - "):
                sample_name = col.split(" - ", 1)[1].strip()
                x_data_dict[sample_name] = df[column]
                continue

            # Signal columns: "{UNIT} - Sample" (e.g., "pA - Sample", "mAU - Sample")
            m = re.match(r"^\s*([^\-]+?)\s*-\s*(.+)\s*$", col)
            if m:
                prefix = m.group(1).strip()
                sample_name = m.group(2).strip()

                # Ignore the Time prefix (handled above)
                if prefix.lower().startswith("time"):
                    continue

                # Treat the prefix as unit (e.g., pA, mAU)
                unit = _extract_unit_from_token(prefix) or prefix

                data_dict[sample_name] = df[column]
                custom_names[sample_name] = sample_name
                ylabels[sample_name] = f"Signal ({unit})"
                continue

        return data_dict, x_data_dict, custom_names, ylabels, None

    except Exception as e:
        return None, None, None, None, f"Error processing CSV file: {str(e)}"

def get_csv_download_data(data_dict, custom_names, x_data_dict, ylabels):
    """Prepare CSV data for download using each sample's unit."""
    output = io.BytesIO()
    new_df = pd.DataFrame()

    for sample in data_dict:
        disp = custom_names.get(sample, sample)
        # Prefer the unit from stored ylabel like 'Signal (pA)' -> 'pA'
        unit = _unit_from_ylabel(ylabels.get(sample, "")) or "Signal"

        new_df[f"Time - {disp}"] = x_data_dict[sample]
        new_df[f"{unit} - {disp}"] = data_dict[sample]

    new_df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

def MatplotlibColorCycler(): #TODO: colors are reset for each new data entry, should be only reset for a new plot
    """Returns an iterator that cycles through matplotlib's default hex colors."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return cycle(colors)
    

### Main Application ###

# Set page config
st.set_page_config(page_title="Chromatogram Plotter", layout="wide")



# Initialize session state
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = {}
if 'x_data_dict' not in st.session_state:
    st.session_state.x_data_dict = {}       
if 'custom_names' not in st.session_state:
    st.session_state.custom_names = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'data_upload'
if 'plot_configs' not in st.session_state:
    st.session_state.plot_configs = []  # List of dicts, each dict contains 'files' and 'title'
if 'csv_file_entries' not in st.session_state:
    st.session_state.csv_file_entries = {}
if 'ylabels' not in st.session_state:
    st.session_state.ylabels = {}  # sample -> 'Signal (pA)' / 'Signal (mAU)' etc.
    

# Page navigation functions
def go_to_data_upload():
    st.session_state.current_page = 'data_upload'

def go_to_visualization():
    st.session_state.current_page = 'visualization'

# Main title
st.title("Chromatogram Plotter")

# Navigation buttons at the top
col1, col2 = st.columns(2)
with col1:
    st.button("Data Upload", on_click=go_to_data_upload, 
              type="primary" if st.session_state.current_page == 'data_upload' else "secondary",
              use_container_width=True)
with col2:
    st.button("Visualization & Export", on_click=go_to_visualization,
              type="primary" if st.session_state.current_page == 'visualization' else "secondary",
              use_container_width=True)

st.divider()

# Display the appropriate page
if st.session_state.current_page == 'data_upload':
    # PAGE 1: DATA UPLOAD
    st.header("Data Upload")
    st.markdown("""
    Please upload one or more chromatogram files exported from Chromeleon in .txt format.
    You can also upload preexisting CSV files exported from this app to continue working on it or to add new data into it.
    """)

    # Initialize all session state variables
    if 'data_dict' not in st.session_state:
        st.session_state.data_dict = {}
    if 'x_data_dict' not in st.session_state:
        st.session_state.x_data_dict = {}
    if 'custom_names' not in st.session_state:
        st.session_state.custom_names = {}
    if 'csv_file_entries' not in st.session_state:
        st.session_state.csv_file_entries = {}
    if 'txt_file_entries' not in st.session_state:
        st.session_state.txt_file_entries = {}
    if 'current_csv_files' not in st.session_state:
        st.session_state.current_csv_files = set()
    if 'current_txt_files' not in st.session_state:   
        st.session_state.current_txt_files = set()

    # 1. CSV FILE UPLOAD SECTION
    st.subheader("Optional: Upload Existing CSV")
    if "csv_uploader_key" not in st.session_state:
        st.session_state["csv_uploader_key"] = 10000
    uploaded_csv_files = st.file_uploader("Upload a preexisting CSV file (optional)", 
                                        type=['csv'], 
                                        accept_multiple_files=True, 
                                        key=st.session_state["csv_uploader_key"])

    # Process CSV files
    if uploaded_csv_files:
        for file in uploaded_csv_files:
            st.session_state.current_csv_files.add(file.name)
            
        # Clean up removed CSV files
        files_to_remove = []
        for csv_filename, entries in list(st.session_state.csv_file_entries.items()):
            if csv_filename not in st.session_state.current_csv_files:
                for entry_key in entries:
                    files_to_remove.append(entry_key)
                st.session_state.csv_file_entries.pop(csv_filename, None)
        
        # Remove data for entries from deleted CSV files
        
        for entry_key in files_to_remove:
            if entry_key not in st.session_state.txt_file_entries:
                st.session_state.data_dict.pop(entry_key, None)
                st.session_state.x_data_dict.pop(entry_key, None)
                st.session_state.custom_names.pop(entry_key, None)
                st.session_state.ylabels.pop(entry_key, None)
        
        # Process uploaded CSV files
        progress_bar_csv = st.progress(0)
        for idx, uploaded_csv in enumerate(uploaded_csv_files):
            new_csv_files_count = 0
            if uploaded_csv.name not in st.session_state.csv_file_entries:
                st.session_state.csv_file_entries[uploaded_csv.name] = []
                
            data_dict_csv, x_data_dict_csv, custom_names_csv, ylabels_csv, error = process_csv_file(uploaded_csv)
            if error:
                st.error(f"Error in CSV file {uploaded_csv.name}: {error}")
            else:
                # Clear previous entries for this file if it's being re-uploaded
                previous_entries = st.session_state.csv_file_entries.get(uploaded_csv.name, [])
                for entry in previous_entries:
                    # st.warning(f"Entry '{entry}' from CSV file {uploaded_csv.name} is being replaced due to re-upload.")
                    st.session_state.data_dict.pop(entry, None)
                    st.session_state.x_data_dict.pop(entry, None)
                    st.session_state.custom_names.pop(entry, None)
                    st.session_state.ylabels.pop(entry, None)
                
                st.session_state.csv_file_entries[uploaded_csv.name] = []
                
                for entry_key, entry_value in data_dict_csv.items():
                    st.session_state.csv_file_entries[uploaded_csv.name].append(entry_key)
                    
                    current_custom_name = custom_names_csv[entry_key]
                    duplicate_exists = False
                    
                    for existing_key, existing_name in st.session_state.custom_names.items():
                        if existing_name == current_custom_name and existing_key != entry_key:
                            duplicate_exists = True
                            st.warning(f"Custom name '{current_custom_name}' from CSV file {uploaded_csv.name} already exists. Latest uploaded entry will be used.")
                            break
                    
                    st.session_state.data_dict[entry_key] = entry_value
                    st.session_state.x_data_dict[entry_key] = x_data_dict_csv[entry_key]
                    st.session_state.custom_names[entry_key] = current_custom_name
                    st.session_state.ylabels[entry_key] = ylabels_csv.get(entry_key, "Signal")
                    
                    if not duplicate_exists:
                        new_csv_files_count += 1
                        
                st.success(f"CSV file {uploaded_csv.name} processed successfully. {new_csv_files_count} unique samples loaded.")
            progress_bar_csv.progress((idx + 1) / len(uploaded_csv_files))

    # 2. TXT FILE UPLOAD SECTION
    st.subheader("Upload Chromeleon Files")
    if "txt_uploader_key" not in st.session_state:
        st.session_state["txt_uploader_key"] = 0
    uploaded_txt_files = st.file_uploader("Upload your .txt files", 
                                        accept_multiple_files=True, 
                                        type=['txt'], 
                                        key=st.session_state["txt_uploader_key"])

    # Process TXT files
    default_names = {}
    new_files_count = 0

    if uploaded_txt_files:
        progress_bar = st.progress(0)     
        # Track current files
        for file in uploaded_txt_files:
            st.session_state.current_txt_files.add(file.name)
        
        # Clean up removed TXT files
        txt_files_to_remove = []
        for txt_filename in list(st.session_state.txt_file_entries.keys()):
            if txt_filename not in st.session_state.current_txt_files:
                txt_files_to_remove.append(txt_filename)
                st.session_state.txt_file_entries.pop(txt_filename, None)
        
        # Remove data for removed TXT files
        for filename in txt_files_to_remove:
            for csv_file, csv_entries in st.session_state.csv_file_entries.items():
                st.text(f"Checking CSV file: {csv_file} with entries {csv_entries}")
                if filename not in csv_entries:
                    st.text(f"Removing data for {filename}")
                    st.session_state.data_dict.pop(filename, None)
                    st.session_state.x_data_dict.pop(filename, None)
                    st.session_state.custom_names.pop(filename, None)
                    st.session_state.ylabels.pop(filename, None)
                else:   
                    st.text(f"Keeping data for {filename} as it exists in CSV entries")



        # Process new TXT files
        for idx, file in enumerate(uploaded_txt_files):
            df, default_name, ylabel, error = process_txt_file(file)
            if error:
                st.error(f"Error in file {file.name}: {error}")
            else:
                is_new_file = file.name not in st.session_state.txt_file_entries
                
                if is_new_file:
                    new_files_count += 1
                    st.session_state.txt_file_entries[file.name] = True
                
                st.session_state.x_data_dict[file.name] = df.iloc[:, 0]
                st.session_state.data_dict[file.name] = df.iloc[:, 1]
                default_names[file.name] = default_name or file.name

                st.session_state.ylabels[file.name] = ylabel or "Signal"

                if file.name not in st.session_state.custom_names:
                    st.session_state.custom_names[file.name] = default_name or file.name
            
            progress_bar.progress((idx + 1) / len(uploaded_txt_files))
        
        if new_files_count > 0:
            st.success(f"{new_files_count} new file(s) processed successfully.")


    # 3. CLEAR ALL DATA BUTTON
    if st.button("🗑️ Clear All Uploaded Data", help="This will remove all uploaded data and custom names."):
        st.session_state["txt_uploader_key"] += 1
        st.session_state["csv_uploader_key"] += 1
        st.session_state.data_dict = {}
        st.session_state.x_data_dict = {}
        st.session_state.custom_names = {}
        st.session_state.csv_file_entries = {}
        st.session_state.txt_file_entries = {}
        st.session_state.ylabels = {}
        st.rerun()
        st.success("All uploaded data cleared.")

    # Custom names input
    if hasattr(st.session_state, 'data_dict') and st.session_state.data_dict:
        st.subheader("Custom Sample Names")
        st.info("💡 Tip: Provide unique, descriptive names for each sample to make them easier to identify.\
                A global legend that allows using the same names for different samples is available in the next step.")
        
        # Display in columns for better organization
        num_cols = 2
        cols = st.columns(num_cols)
        
        for idx, filename in enumerate(st.session_state.data_dict.keys()):
            col = cols[idx % num_cols]
            with col:
                # Get default name from default_names or use existing custom name
                default_value = default_names.get(filename, st.session_state.custom_names.get(filename, filename))
                
                st.session_state.custom_names[filename] = st.text_input(
                    f"{filename}",
                    value=default_value,
                    key=f"name_{filename}"
                )

                        
        
        # Check for duplicate custom names
        custom_names_list = list(st.session_state.custom_names.values())
        if len(custom_names_list) != len(set(custom_names_list)):
            st.warning("Warning: Duplicate custom names detected. Consider using unique names for clarity.")
    
    # Navigation section at the bottom
    st.divider()
    
    if st.session_state.data_dict:
        col1, col2, col3 = st.columns([2, 4, 2])
        with col2:
            st.button("Next: Visualization & Export →", 
                     on_click=go_to_visualization,
                     type="primary",
                     use_container_width=True)
        
        st.success(f"{len(st.session_state.data_dict)} file(s) ready for visualization.")
    else:
        st.info("Please upload chromatogram files to continue.")

elif st.session_state.current_page == 'visualization':
    # PAGE 2: VISUALIZATION & EXPORT
    st.header("Visualization & Export")
    
    # Check if data is available
    if not st.session_state.data_dict:
        st.warning("No data available. Please upload files first.")
        if st.button("← Go to Data Upload"):
            go_to_data_upload()
            st.rerun()
    else:
        # Load data from session state
        data_dict = st.session_state.data_dict
        x_data_dict = st.session_state.x_data_dict
        custom_names = st.session_state.custom_names
        
        # Plot configuration section
        st.subheader("Plot Configuration")
        
        col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
        with col1:
            if st.button("➕ Add Plot", use_container_width=True):
                st.session_state.plot_configs.append({'files': [], 'title': f'F{len(st.session_state.plot_configs)+1}'})
        with col2:
            if st.session_state.plot_configs and st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.plot_configs = []
                st.rerun()
        with col3:
            if st.button("← Back to Data", use_container_width=True):
                go_to_data_upload()
                st.rerun()
        
        # Configure each plot
        if st.session_state.plot_configs:
            for i, config in enumerate(st.session_state.plot_configs):
                color_cycler = MatplotlibColorCycler()

                with st.expander(f"Plot {i+1} Configuration", expanded=True):
                    col1, col2, col3 = st.columns([1, 7, 1, ])
                    with col1:
                        config['title'] = st.text_input(
                            "Plot Title", 
                            value=config.get('title', f"F{i+1}"), 
                            key=f"title_{i}"
                        )

                    with col2:
                        # select files to plot from checkbox list
                        for option in list(data_dict.keys()):
                            col21, col22 = st.columns([5,1])
                            with col21:
                                if option not in config['files']:
                                    if st.checkbox(custom_names.get(option, option), key=f"chk_{i}_{option}"):
                                        config['files'].append(option)
                                else:
                                    if not st.checkbox(custom_names.get(option, option), key=f"chk_{i}_{option}"):
                                        config['files'].remove(option)
                            with col22:
                                if option in config['files']:
                                    st.session_state[f"color_{i}_{option}"] = st.color_picker(
                                        "",
                                        value=next(color_cycler), 
                                        key=f"color_picker_{i}_{option}", 
                                        kwargs={"width": 2 },
                                    )
                                

                    with col3:
                        if st.button("🗑️", key=f"remove_{i}", help="Remove this plot"):
                            st.session_state.plot_configs.pop(i)
                            st.rerun()

            # Generate and display plots
            # if any(config.get('files') for config in st.session_state.plot_configs):
            st.subheader("Generated Plots")
            
            # Plot options
            with st.expander("⚙️ Plot Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    suptitle_enabled = st.toggle("Enable common title", value=False)
                    if suptitle_enabled:
                        suptitle = st.text_input("Common Title (Press ENTER to confirm)", value="Formulation")
                    else:
                        suptitle = ""
                    
                    supaxes_enabled = st.toggle("Enable common axis labels", value=False)
                
                with col2:
                    external_label = st.toggle("Enable external legend")
                    custom_legend = None
                    if external_label:
                        custom_legend = st.text_area(
                            "Each new line is one legend entry. This function is purely cosmetic. It overwrites the existing legend with anything you put in and must be used carefully. Press STRG+ENTER to apply.", 
                            height=100,
                            help="Leave empty to use sample names."
                        )
                    log_y = st.toggle("Enable logarithmic y-axis", value=False)

            # Generate the plot
            
            fig = generate_plots(
                data_dict,
                custom_names,
                x_data_dict,
                st.session_state.plot_configs,
                ylabels_per_sample=st.session_state.ylabels,   
                external_label=external_label,
                custom_legend=custom_legend,
                suptitle_enabled=suptitle_enabled,
                suptitle=suptitle,
                supaxes_enabled=supaxes_enabled,
                log_y=log_y
            )

            
            if fig:
                    st.pyplot(fig)#, width="content")
                    
                    # Download options
                    st.subheader("Download Options")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label="PNG",
                            data=buf.getvalue(),
                            file_name="chromatogram_plot.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with col2:
                        buf_pdf = io.BytesIO()
                        fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
                        buf_pdf.seek(0)
                        st.download_button(
                            label="PDF",
                            data=buf_pdf.getvalue(),
                            file_name="chromatogram_plot.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    with col3:
                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format="svg", bbox_inches='tight')
                        buf_svg.seek(0)     
                        st.download_button(
                            label="SVG",
                            data=buf_svg.getvalue(),
                            file_name="chromatogram_plot.svg",
                            mime="image/svg+xml",
                            use_container_width=True
                        )
                    
                    with col4:
                        csv_data = get_csv_download_data(data_dict, custom_names, x_data_dict, st.session_state.ylabels)
                        st.download_button(
                            label="Data (CSV)",
                            data=csv_data,
                            file_name="chromatogram_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    

            else:
                st.info("Please select files for at least one plot to generate visualizations.")
                st.info("Click 'Add Plot' to start creating visualizations.")
                
                st.subheader("Download Options")
                col1, col2, col3, col4 = st.columns(4)
                with col4:
                    csv_data = get_csv_download_data(data_dict, custom_names, x_data_dict, st.session_state.ylabels)
                    st.download_button(
                        label="Data (CSV)",
                        data=csv_data,
                        file_name="chromatogram_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("Click 'Add Plot' to start creating visualizations.")
            
            st.subheader("Download Options")
            col1, col2, col3, col4 = st.columns(4)
            with col4:
                csv_data = get_csv_download_data(data_dict, custom_names, x_data_dict, st.session_state.ylabels)
                st.download_button(
                    label="Data (CSV)",
                    data=csv_data,
                    file_name="chromatogram_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )