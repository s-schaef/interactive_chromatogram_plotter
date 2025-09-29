import io
import math
import pandas as pd
import streamlit as st
from itertools import cycle
import matplotlib.pyplot as plt

### function definitions
def generate_plots(data_dict, custom_names, x_data_dict, plot_configs, external_label=False, custom_legend=None,
                   suptitle_enabled=True, suptitle="Formulation", supaxes_enabled=True):
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
        fig, axs = plt.subplots(math.ceil(len(valid_configs)/3), 3, figsize=(15, 10), squeeze=True)
    axs = axs.flat

    # remove unused axis in case of 5 plots
    if len(valid_configs) == 5:
        axs[-1].set_visible(False)
        
    for i, config in enumerate(valid_configs):
        ax = axs[i]
        
        if external_label and custom_legend:
            for filename in config['files']:
                if filename in data_dict:
                    ax.plot(x_data_dict[filename], data_dict[filename])
        else:
            for filename in config['files']:
                if filename in data_dict:
                    ax.plot(
                        x_data_dict[filename], 
                        data_dict[filename], 
                        label=custom_names.get(filename, filename)
                    )
            ax.legend(loc="upper left", fontsize='small')
        ax.set_title(config['title'])
    
    if suptitle_enabled:
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
        for i, ax in enumerate(axs):
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
        fig.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    return fig

def process_txt_file(uploaded_file): 
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
        return None, None, f"Error processing file: {str(e)} \n Please ensure the file is a valid Chromelion exported .txt file."

def process_csv_file(uploaded_file): #fix key error
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
    output = io.BytesIO()
    new_df = pd.DataFrame() 

    for col in data_dict:
        new_df[f"Time - {custom_names.get(col, col)}"] = x_data_dict[col]
        new_df[f"pA - {custom_names.get(col, col)}"] = data_dict[col]

    new_df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()


### Streamlit app layout and logic
# Set page config
st.set_page_config(page_title="Chromatogram Plotter", layout="wide")

# Main title
st.title("Chromatogram Plotter")
# Initialize session state
if 'plot_configs' not in st.session_state:
    st.session_state.plot_configs = []
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = {}
if 'x_data_dict' not in st.session_state:
    st.session_state.x_data_dict = {}
if 'custom_names' not in st.session_state:
    st.session_state.custom_names = {}



# Tabs for different sections
tab1, tab2 = st.tabs(["Data Upload", "Visualization & Export"])
with tab1:
    st.header("Upload Chromatogram Files")
    st.markdown("""
    Please upload one or more chromatogram files exported from Chromelion in .txt format.
    You can also upload a preexisting CSV file with the same structure (e.g., exported from this app),
    to continue working on it or to add data into it.
    """)

    # File upload section
    # upload preexisting csv file 
    uploaded_csv = st.file_uploader("Upload a preexisting CSV file (optional)", type=['csv'])
    if uploaded_csv:
        data_dict_csv, x_data_dict_csv, custom_names_csv, error = process_csv_file(uploaded_csv)
        if error:
            st.error(f"Error in CSV file: {error}")
        else:
            # Merge with existing session state data
            st.session_state.data_dict.update(data_dict_csv)
            st.session_state.x_data_dict.update(x_data_dict_csv)
            st.session_state.custom_names.update(custom_names_csv)
            st.success("CSV file processed and data merged successfully.")
    # upload new txt files    
    uploaded_txt_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=['txt'])
    
    # # Process uploaded txt files
    # data_dict = {}
    # x_data_dict = {} #  = None
    default_names = {}

    if uploaded_txt_files:
        for file in uploaded_txt_files:
            df, default_name, error = process_txt_file(file)
            if error:
                st.error(f"Error in file {file.name}: {error}")
            else:
                #if x_data is None:
                st.session_state.x_data_dict[file.name] = df.iloc[:, 0]
                st.session_state.data_dict[file.name] = df.iloc[:, 1]
                default_names[file.name] = default_name or file.name

    # Custom names input
    custom_names = {}
    if st.session_state.data_dict:
        st.header("Custom Sample Names")
        for filename in st.session_state.data_dict.keys():
            st.session_state.custom_names[filename] = st.text_input(
                f"Custom name for {filename}",
                value=default_names[filename],
                key=f"name_{filename}"
            )
        #TODO: document avoid duplicate names, move to visualization tab etc. 
    
    st.info("After uploading files and setting custom names, please switch to the 'Visualization & Export' tab on the top of the page.")


with tab2:
    # Load data from session state
    data_dict = st.session_state.get('data_dict', {})
    x_data_dict = st.session_state.get('x_data_dict', {})
    custom_names = st.session_state.get('custom_names', {})

    # Update session state with new uploads
    if uploaded_txt_files:
        st.session_state.data_dict.update(data_dict)
        st.session_state.x_data_dict.update(x_data_dict)
        st.session_state.custom_names.update(custom_names)
        data_dict = st.session_state.data_dict
        x_data_dict = st.session_state.x_data_dict
        custom_names = st.session_state.custom_names
    # Plot configuration
    if data_dict:
        st.header("Plot Configuration")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Add Plot"):
                st.session_state.plot_configs.append({'files': []})
            
            if st.session_state.plot_configs and st.button("Clear All Plots"):
                st.session_state.plot_configs = []
                st.rerun()
        
        # Configure each plot
        for i, config in enumerate(st.session_state.plot_configs):
            with st.expander(f"Plot {i+1} Configuration", expanded=True):
                col1, col2, col3 = st.columns([1, 5, 1])
                with col1:
                    config['title'] = st.text_input(f"Rename Plot {i+1}", value=f"F{i+1}", key=f"title_{i}")

                with col2:
                    config['files'] = st.multiselect(
                        f"Select files for Plot {i+1}", 
                        options=list(data_dict.keys()),
                        default=config.get('files', []),
                        format_func=lambda x: custom_names.get(x, x),
                        key=f"plot_{i}"
                    )
                with col3:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.plot_configs.pop(i)
                        st.rerun()

    # Generate and display plots
        if st.session_state.plot_configs and any(config.get('files') for config in st.session_state.plot_configs):
            st.header("Generated Plots")
            external_label = st.toggle("Enable external legend")
            custom_legend = None
            if external_label:
                custom_legend = st.text_area("Custom Legend Text (one entry per line)", height=100)

            fig = generate_plots(
                data_dict, custom_names, x_data_dict, 
                st.session_state.plot_configs, 
                external_label=external_label, 
                custom_legend=custom_legend,
                suptitle=st.text_input("Common Title", value="Formulation") if st.toggle("Enable common title", value=True) else "",
                supaxes_enabled=st.toggle("Enable common axis labels", value=True),
            )
            
            if fig:
                st.pyplot(fig)
                
                # Download options
                col1, col2, col3 = st.columns(3)
                with col1:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot (PNG)",
                        data=buf.getvalue(),
                        file_name="chromatogram_plot.png",
                        mime="image/png"
                    )
                
                with col2:
                    buf_pdf = io.BytesIO()
                    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
                    buf_pdf.seek(0)
                    st.download_button(
                        label="Download Plot (PDF)",
                        data=buf_pdf.getvalue(),
                        file_name="chromatogram_plot.pdf",
                        mime="application/pdf"
                    )
                
                with col3:
                    buf_svg = io.BytesIO()
                    fig.savefig(buf_svg, format="svg", bbox_inches='tight')
                    buf_svg.seek(0)     
                    st.download_button(
                        label="Download Plot (SVG)",
                        data=buf_svg.getvalue(),
                        file_name="chromatogram_plot.svg",
                        mime="image/svg+xml"
                    )   


    # Data export section
    if data_dict:
        st.header("Export Data")
        csv_data = get_csv_download_data(data_dict, custom_names, x_data_dict)
        st.download_button(
            label="Download Processed Data (CSV)",
            data=csv_data,
            file_name="processed_chromatogram_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload chromatogram files to begin processing.")

# Add sidebar with instructions
# with st.sidebar:
#     st.header("Instructions")
#     st.markdown("""
#     1. **Upload Files**: Select one or more .txt chromatogram files
#     2. **Customize Names**: Edit sample names if needed
#     3. **Create Plots**: Click 'Add Plot' and select files to display
#     4. **Export**: Download plots or processed data
    
#     **File Requirements:**
#     Only Chromelion exported .txt files can be used.
#     """)
    
#     st.header("About")
#     st.markdown("""
#     Chromatogram Plotter v1.1\n
#     Developed by Stefan Schaefer\n 
#     2025   
#     """)