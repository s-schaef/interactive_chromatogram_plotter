import io
import math
import pandas as pd
import streamlit as st
from itertools import cycle
import matplotlib.pyplot as plt

### Function definitions ###

def generate_plots(data_dict, custom_names, x_data_dict, plot_configs, external_label=False, custom_legend=None,
                   suptitle_enabled=True, suptitle="Formulation", supaxes_enabled=True):
    """Generate matplotlib plots based on configuration."""
    # Filter out empty plot configs
    valid_configs = [config for config in plot_configs if config.get('files')]
    
    if not valid_configs:
        return None
    
    # Handle subplot layout
    if len(valid_configs) in [1, 2, 3]:
        fig, axs = plt.subplots(1, len(valid_configs), figsize=(5*len(valid_configs), 5), squeeze=False)
    elif len(valid_configs) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
    else:
        fig, axs = plt.subplots(math.ceil(len(valid_configs)/3), 3, figsize=(15, 10), squeeze=False)
    axs = axs.flat

    # Remove unused axes for non-rectangular layouts
    total_subplots = len(axs)
    for i in range(len(valid_configs), total_subplots):
        axs[i].set_visible(False)
        
    for i, config in enumerate(valid_configs):
        ax = axs[i]
        
        if external_label: # and custom_legend:
            for filename in config['files']:
                if filename in data_dict:
                    ax.plot(x_data_dict[filename], data_dict[filename],
                            color=st.session_state.get(f"color_{i}_{filename}", None)  # Use custom color if provided
                           )
        else:
            for filename in config['files']:
                if filename in data_dict:
                    ax.plot(
                        x_data_dict[filename], 
                        data_dict[filename], 
                        label=custom_names.get(filename, filename),
                        color=st.session_state.get(f"color_{i}_{filename}", None)  # Use custom color if provided
                    )
            ax.legend(loc="upper left", fontsize='small')
            # if config['files'] and not custom_legend:  # Only add legend if there are files
            #     ax.legend(loc="upper left", fontsize='small')
        
        ax.set_title(config.get('title', f'Plot {i+1}'))
    
    if suptitle_enabled and suptitle:
        fig.suptitle(suptitle, fontsize=16)
    
    if supaxes_enabled:
        fig.supxlabel("Time (min)")
        fig.supylabel("Intensity (mAU)")
        
        # Determine the layout shape
        if len(valid_configs) in [1, 2, 3]:
            rows, cols = 1, len(valid_configs)
        elif len(valid_configs) == 4:
            rows, cols = 2, 2
        else:
            rows, cols = math.ceil(len(valid_configs)/3), 3
        
        # Hide tick labels for non-edge axes
        for i, ax in enumerate(axs[:len(valid_configs)]):
            if not ax.get_visible():
                continue
                
            # Calculate row and column position
            row = i // cols
            col = i % cols
            
            # Only show y-tick labels for leftmost axes
            if col > 0:
                ax.tick_params(axis='y', labelleft=False)
            
            # Only show x-tick labels for bottom axes
            if row < (rows - 1):
                ax.tick_params(axis='x', labelbottom=False)
    else:
        all_ylims = []
        for ax in axs:
            if ax.get_visible():
                ax.set_xlabel("Time (min)")
                ax.set_ylabel("Intensity (mAU)")
                
                # Update limits based on the plotted data
                ax.relim()
                ax.autoscale_view()
                
                all_ylims.append(ax.get_ylim())
        
        # Now set consistent y-limits across all axes
        if all_ylims:
            y_min = min(lim[0] for lim in all_ylims)
            y_max = max(lim[1] for lim in all_ylims)
            for ax in axs:
                if ax.get_visible():
                    ax.set_ylim(y_min, y_max)
    
    if external_label and custom_legend:
        labels = [line.strip() for line in custom_legend.splitlines() if line.strip()]
        fig.legend(labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
    elif external_label and not custom_legend:
        # Use custom names for external legend
        unique_files = []
        for config in valid_configs:
            for filename in config['files']:
                if filename not in unique_files:
                    unique_files.append(filename)
        labels = [custom_names.get(f, f) for f in unique_files]
        fig.legend(labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    return fig

def process_txt_file(uploaded_file):
    """Process a Chromelion exported .txt file."""
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
        return None, None, "File size exceeds 200MB limit."
   
    try:
        # Read the first few lines to extract the default name
        lines = []
        for i, line in enumerate(uploaded_file):
            if i < 7:  # Read enough lines to get to the 6th line
                lines.append(line.decode('utf-8'))
            else:
                break
       
        # Reset file pointer
        uploaded_file.seek(0)
       
        # Extract default name from 6th line, 2nd column (if available)
        default_name = None
        if len(lines) >= 6:
            sixth_line_parts = lines[5].strip().split('\t')
            if len(sixth_line_parts) >= 2 and sixth_line_parts[0].strip() == "Injection":
                default_name = sixth_line_parts[1].strip()

        # Read the file into a DataFrame    
        df = pd.read_csv(uploaded_file, sep='\t', header=42)
        if len(df.columns) != 3:
            return None, None, "File should have exactly 3 columns."
       
        return df.iloc[:, [0, 2]], default_name, None  # Return 1st and 3rd columns
    except Exception as e:
        return None, None, f"Error processing file: {str(e)}\nPlease ensure the file is a valid Chromelion exported .txt file."

def process_csv_file(uploaded_file):
    """Process a CSV file exported from this app."""
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
        return None, None, None, "File size exceeds 200MB limit."
    try:
        df = pd.read_csv(uploaded_file)
        data_dict = {}
        x_data_dict = {}
        custom_names = {}
        
        for column in df.columns:
            if column.startswith("Time - "):
                sample_name = column[7:]  # Remove "Time - " prefix
                x_data_dict[sample_name] = df[column]
            elif column.startswith("pA - "):
                sample_name = column[5:]  # Remove "pA - " prefix
                data_dict[sample_name] = df[column]
                custom_names[sample_name] = sample_name
        
        return data_dict, x_data_dict, custom_names, None
    except Exception as e:
        return None, None, None, f"Error processing CSV file: {str(e)}"

def get_csv_download_data(data_dict, custom_names, x_data_dict):
    """Prepare CSV data for download."""
    output = io.BytesIO()
    new_df = pd.DataFrame() 

    for col in data_dict:
        new_df[f"Time - {custom_names.get(col, col)}"] = x_data_dict[col]
        new_df[f"pA - {custom_names.get(col, col)}"] = data_dict[col]

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
    Please upload one or more chromatogram files exported from Chromelion in .txt format.
    You can also upload preexisting CSV files exported from this app to continue working on it or to add new data into it.
    """)

    # File upload section
    # Upload preexisting CSV file 
    st.subheader("Optional: Upload Existing CSV")
    if "csv_uploader_key" not in st.session_state:
        st.session_state["csv_uploader_key"] = 10000  # Initialize key for CSV uploader
    uploaded_csv_files = st.file_uploader("Upload a preexisting CSV file (optional)", type=['csv'], accept_multiple_files=True, key=st.session_state["csv_uploader_key"])
 
    if uploaded_csv_files:
        progress_bar_csv = st.progress(0)
        for idx, uploaded_csv in enumerate(uploaded_csv_files):
            new_csv_files_count = 0
            data_dict_csv, x_data_dict_csv, custom_names_csv, error = process_csv_file(uploaded_csv)
            if error:
                st.error(f"Error in CSV file {uploaded_csv.name}: {error}")
            else:
                for entry_key, entry_value in data_dict_csv.items():
                    # Check if file already exists
                    current_custom_name = custom_names_csv[entry_key]
                    
                    if current_custom_name not in st.session_state.custom_names.values():
                        new_csv_files_count += 1
                    else:
                        st.warning(f"Custom name {current_custom_name} from CSV file {uploaded_csv.name} already exists. Latest uploaded entry will be used.")

                    # Direct assignment instead of update
                    st.session_state.data_dict[entry_key] = entry_value
                    st.session_state.x_data_dict[entry_key] = x_data_dict_csv[entry_key]
                    st.session_state.custom_names[entry_key] = current_custom_name
                st.success(f"CSV file {uploaded_csv.name} processed successfully. {new_csv_files_count} samples loaded.")
            #progress_bar_csv.progress((idx + 1) / new_csv_files_count)

    
    # Upload new txt files    
    st.subheader("Upload Chromelion Files")
    if "txt_uploader_key" not in st.session_state:
        st.session_state["txt_uploader_key"] = 0  # Initialize key for txt uploader

    uploaded_txt_files = st.file_uploader("Upload your .txt files", accept_multiple_files=True, type=['txt'], key=st.session_state["txt_uploader_key"])

    # delete all  data
    if st.button("🗑️ Clear All Uploaded Data", help="This will remove all uploaded data and custom names."):
        st.session_state["txt_uploader_key"] += 1  # Change key to reset uploader
        st.session_state["csv_uploader_key"] += 1  # Change key to reset uploader
        # also reset data and custom names
        st.session_state.data_dict = {}
        st.session_state.x_data_dict = {}
        st.session_state.custom_names = {}
        st.experimental_rerun()  # Rerun to reset the uploader
        st.success("All uploaded data cleared.")

    # Process uploaded txt files
    default_names = {}
    new_files_count = 0

    # Track current files to detect removals
    current_files = set()

    if uploaded_txt_files:
        progress_bar = st.progress(0)
        
        # First, identify all current files
        for file in uploaded_txt_files:
            current_files.add(file.name)
        
        # Clean up removed files before processing new ones
        files_to_remove = []
        if hasattr(st.session_state, 'data_dict'):
            for filename in st.session_state.data_dict.keys():
                if filename not in current_files:
                    files_to_remove.append(filename)
            
            for filename in files_to_remove:
                # Remove data for files that were deleted via the "x" button
                st.session_state.data_dict.pop(filename, None)
                st.session_state.x_data_dict.pop(filename, None)
                st.session_state.custom_names.pop(filename, None)
        
        # Now process the uploaded files
        for idx, file in enumerate(uploaded_txt_files):
            df, default_name, error = process_txt_file(file)
            if error:
                st.error(f"Error in file {file.name}: {error}")
            else:
                # Check if this is a new file in this session
                is_new_file = True
                
                # If we're re-running the app and the file was already processed before
                if file.name in st.session_state.data_dict:
                    # Check if the content is the same (optional, can be complex)
                    # For simplicity, we'll just assume it's a re-upload
                    is_new_file = False
                    # Remove the old data to avoid the warning
                    st.session_state.data_dict.pop(file.name, None)
                    st.session_state.x_data_dict.pop(file.name, None)
                    # Keep the custom name if it exists
                
                if is_new_file:
                    new_files_count += 1
                
                # Process the file
                st.session_state.x_data_dict[file.name] = df.iloc[:, 0]
                st.session_state.data_dict[file.name] = df.iloc[:, 1]
                default_names[file.name] = default_name or file.name
                
                # Initialize custom name if not present
                if file.name not in st.session_state.custom_names:
                    st.session_state.custom_names[file.name] = default_name or file.name
            
            progress_bar.progress((idx + 1) / len(uploaded_txt_files))
        
        if new_files_count > 0:
            st.success(f"{new_files_count} new file(s) processed successfully.")

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
                st.rerun()
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
                with st.expander(f"Plot {i+1} Configuration", expanded=True):
                    color_cycler = MatplotlibColorCycler()

                    col1, col2, col3 = st.columns([2, 7, 1, ])
                    with col1:
                        config['title'] = st.text_input(
                            "Plot Title", 
                            value=config.get('title', f"F{i+1}"), 
                            key=f"title_{i}"
                        )

                    with col2:
                        # select files to plot from checkbox list
                        col21, col22 = st.columns([5,1])
                        with col21:
                            for option in list(data_dict.keys()):
                                if option not in config['files']:
                                    if st.checkbox(custom_names.get(option, option), key=f"chk_{i}_{option}"):
                                        config['files'].append(option)
                                        with col22:
                                            # ask for custom color to be later used in the plot
                                            #TODO: colors are reset for each new data entry, should be only reset for a new plot
                                            #FIXME: color is not saved in session state/ does not make it into the plot
                                            st.color_picker("Pick a color (optional)", value=next(color_cycler) ,key=f"color_{i}_{option}, size=5, ")
                                            # FIXME: below three lines are not tested at all!!! 
                                            color = st.session_state.get(f"color_{i}_{option}", None)
                                            if color:
                                                st.session_state[f"color_{i}_{option}"] = color 

                                        
                                else:
                                    if not st.checkbox(custom_names.get(option, option), key=f"chk_{i}_{option}"):
                                        config['files'].remove(option)
                            


                    with col3:
                        if st.button("🗑️", key=f"remove_{i}", help="Remove this plot"):
                            st.session_state.plot_configs.pop(i)
                            st.rerun()

            # Generate and display plots
            if any(config.get('files') for config in st.session_state.plot_configs):
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

                # Generate the plot
                fig = generate_plots(
                    data_dict,
                    custom_names, 
                    x_data_dict, 
                    st.session_state.plot_configs, 
                    external_label=external_label, 
                    custom_legend=custom_legend,
                    suptitle_enabled=suptitle_enabled,
                    suptitle=suptitle,
                    supaxes_enabled=supaxes_enabled,
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
                        csv_data = get_csv_download_data(data_dict, custom_names, x_data_dict)
                        st.download_button(
                            label="Data (CSV)",
                            data=csv_data,
                            file_name="chromatogram_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    

            else:
                st.info("Please select files for at least one plot to generate visualizations.")
        else:
            st.info("Click 'Add Plot' to start creating visualizations.")
            
            st.subheader("Download Options")
            col1, col2, col3, col4 = st.columns(4)
            with col4:
                csv_data = get_csv_download_data(data_dict, custom_names, x_data_dict)
                st.download_button(
                    label="Data (CSV)",
                    data=csv_data,
                    file_name="chromatogram_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )