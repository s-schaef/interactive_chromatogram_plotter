import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import io

# Set page config
st.set_page_config(page_title="Chromatogram Plotter", layout="wide")

# Main title
st.title("Chromatogram Plotter")

# Initialize session state
if 'plot_configs' not in st.session_state:
    st.session_state.plot_configs = []

#### data functions
def process_file(uploaded_file):
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
        return None, None, f"Error processing file: {str(e)}"

# File upload section
uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=['txt'])

# Process uploaded files
data_dict = {}
x_data = None
default_names = {}

if uploaded_files:
    for file in uploaded_files:
        df, default_name, error = process_file(file)
        if error:
            st.error(f"Error in file {file.name}: {error}")
        else:
            if x_data is None:
                x_data = df.iloc[:, 0]
            data_dict[file.name] = df.iloc[:, 1]
            default_names[file.name] = default_name or file.name

# Custom names input
custom_names = {}
if data_dict:
    st.header("Custom Sample Names")
    for filename in data_dict.keys():
        custom_names[filename] = st.text_input(
            f"Custom name for {filename}",
            value=default_names[filename],
            key=f"name_{filename}"
        )

### plotting function
def generate_plots(data_dict, custom_names, x_data, plot_configs):
    # Filter out empty plot configs
    valid_configs = [config for config in plot_configs if config.get('files')]
    
    if not valid_configs:
        return None
    
    if len(valid_configs) in [1, 2, 3]:
        fig, axs = plt.subplots(1, len(valid_configs), figsize=(5*len(valid_configs), 5), squeeze=False, sharey=True, sharex=True)
    elif len(valid_configs) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), squeeze=False, sharey=True, sharex=True)
    elif len(valid_configs) > 4:
        fig, axs = plt.subplots(math.ceil(len(valid_configs)/3), 3, figsize=(15, 10), squeeze=True, sharey=True, sharex=True) # squeeze necessary? 
    axs = axs.flat  # Flatten in case of multiple rows
    for i, config in enumerate(valid_configs):
        ax = axs[i]#, 0]
        for filename in config['files']:
            if filename in data_dict:  # Check if file exists
                ax.plot(x_data, data_dict[filename], label=custom_names.get(filename, filename))
        ax.set_title(config['title'])
        # ax.set_xlabel("Time (min)")
        # ax.set_ylabel("Intensity (mAU)")
        if not external_label: #not st.session_state['external_label']:
            ax.legend(loc="upper left", fontsize='small')
        #ax.grid(True, alpha=0.3) # uncomment to add grid back in
    fig.suptitle("Formulation", fontsize=16)
    fig.supxlabel("Time (min)")
    fig.supylabel("Intensity (mAU)")
    if external_label: 
        box = ax.get_position()
        fig.legend(bbox_to_anchor=(box.x0+box.width+0.1, 0.5), loc='center left')
    plt.tight_layout()
    return fig

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
            col1, col2, col3 = st.columns([2, 5, 1])
            with col1:
                config['title'] = st.text_input(f"Rename Plot {i+1}", value=f"Plot {i+1}", key=f"title_{i}")

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
        st.session_state['external_label'] = False # Initialize external label state
        external_label = st.toggle("External Legend")

        fig = generate_plots(data_dict, custom_names, x_data, st.session_state.plot_configs)
        
        if fig:
            st.pyplot(fig)
            
            # Download options
            col1, col2 = st.columns(2)
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

### export functions
def get_csv_download_data(data_dict, custom_names, x_data):
    output = io.BytesIO()
    df = pd.DataFrame(data_dict)
    df.columns = [custom_names.get(col, col) for col in df.columns]
    df.insert(0, "Time (min)", x_data)
    df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

# Data export section
if data_dict:
    st.header("Export Data")
    csv_data = get_csv_download_data(data_dict, custom_names, x_data)
    st.download_button(
        label="Download Processed Data (CSV)",
        data=csv_data,
        file_name="processed_chromatogram_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload chromatogram files to begin processing.")

# Add sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. **Upload Files**: Select one or more .txt chromatogram files
    2. **Customize Names**: Edit sample names if needed
    3. **Create Plots**: Click 'Add Plot' and select files to display
    4. **Export**: Download plots or processed data
    
    **File Requirements:**
    - Tab-separated .txt files
    - 3 columns (unused second column)
    - Data starts at row 43
    """)
    
    st.header("About")
    st.markdown("""
    Chromatogram Plotter v1.0
    Developed by Stefan Schaefer 
    2025   
    """)