import io
import math
import pandas as pd
import streamlit as st
# from itertools import cycle
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from matplotlib import colors as mcolors


### Function definitions ###

def generate_plots(data_dict, custom_names, x_data_dict, plot_configs, external_label=False, custom_legend=None,
                   suptitle_enabled=True, suptitle="Formulation", supaxes_enabled=True):
    """Generate interactive Plotly plots based on configuration."""
    # Get matplotlib default colors
    mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Filter out empty plot configs
    valid_configs = [config for config in plot_configs if config.get('files')]
    
    if not valid_configs:
        return None
    
    # Handle subplot layout
    if len(valid_configs) in [1, 2, 3]:
        rows, cols = 1, len(valid_configs)
    elif len(valid_configs) == 4:
        rows, cols = 2, 2
    else:
        rows = math.ceil(len(valid_configs)/3)
        cols = 3
    
    # Create subplot figure
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[config.get('title', f'Plot {i+1}') for i, config in enumerate(valid_configs)],
        shared_xaxes=supaxes_enabled,
        shared_yaxes=supaxes_enabled
    )
    
    # Add traces to subplots
    color_idx = 0
    for i, config in enumerate(valid_configs):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        for filename in config['files']:
            if filename in data_dict:
                trace = go.Scatter(
                    x=x_data_dict[filename],
                    y=data_dict[filename],
                    name=custom_names.get(filename, filename),
                    line=dict(width=2, color=mpl_colors[color_idx % len(mpl_colors)]),
                    showlegend=not external_label or i == 0,  # Show legend only once if external
                )
                fig.add_trace(trace, row=row, col=col)
                color_idx += 1
    
    # Update layout
    fig.update_layout(
        title=suptitle if suptitle_enabled else None,
        height=400 * rows,
        width=800 if cols < 3 else 1200,
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right" if external_label else "left",
            x=1.15 if external_label else 1.02
        ),
        clickmode='event+select'
    )
    
    # Update axes labels if common axes are enabled
    if supaxes_enabled:
        fig.update_xaxes(title_text="Time (min)")
        fig.update_yaxes(title_text="Intensity (mAU)")
    
    # Add color picker buttons for each trace
    buttons = []
    for i in range(len(fig.data)):
        buttons.append(
            dict(
                args=[{"line.color": [[None] * i + [f"{{colorPicker}}"] + [None] * (len(fig.data) - i - 1)]}],
                label=f"Change color: {fig.data[i].name}",
                method="restyle",
                name=f"color_{i}",
                type="colorPicker",
                value=fig.data[i].line.color
            )
        )
    
    # Add the color picker menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0,
                y=1.1,
                xanchor="left",
                yanchor="top",
            )
        ]
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="Time: %{x:.2f}<br>Intensity: %{y:.2f}<extra></extra>",
    )
    
    # Disable zoom
    fig.update_layout(
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )
    
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


### Main Application ###

# Set page config
st.set_page_config(page_title="Chromatogram Plotter", layout="wide")

# Initialize session state
if 'plot_configs' not in st.session_state:
    st.session_state.plot_configs = []
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = {}
if 'x_data_dict' not in st.session_state:
    st.session_state.x_data_dict = {}
if 'custom_names' not in st.session_state:
    st.session_state.custom_names = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'data_upload'

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
    You can also upload a preexisting CSV file with the same structure (e.g., exported from this app),
    to continue working on it or to add data into it.
    """)

    # File upload section
    # Upload preexisting CSV file 
    st.subheader("Optional: Upload Existing CSV")
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
            st.success(f"CSV file processed successfully. {len(data_dict_csv)} samples loaded.")
    
    # Upload new txt files    
    st.subheader("Upload Chromelion Files")
    uploaded_txt_files = st.file_uploader("Upload your .txt files", accept_multiple_files=True, type=['txt'])
    
    # Process uploaded txt files
    default_names = {}
    new_files_count = 0

    if uploaded_txt_files:
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_txt_files):
            df, default_name, error = process_txt_file(file)
            if error:
                st.error(f"Error in file {file.name}: {error}")
            else:
                # Check if file already exists
                if file.name not in st.session_state.data_dict:
                    new_files_count += 1
                
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
    if st.session_state.data_dict:
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
        col1, col2, col3 = st.columns([2, 1, 2])
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
                    col1, col2, col3 = st.columns([2, 8, 1])
                    with col1:
                        config['title'] = st.text_input(
                            "Plot Title", 
                            value=config.get('title', f"F{i+1}"), 
                            key=f"title_{i}"
                        )

                    with col2:
                        # select files to plot from checkbox list
                        for option in list(data_dict.keys()):
                            if option not in config['files']:
                                if st.checkbox(custom_names.get(option, option), key=f"chk_{i}_{option}"):
                                    config['files'].append(option)
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
                        suptitle_enabled = st.toggle("Enable common title", value=True)
                        if suptitle_enabled:
                            suptitle = st.text_input("Common Title", value="Formulation")
                        else:
                            suptitle = ""
                        
                        supaxes_enabled = st.toggle("Enable common axis labels", value=True)
                    
                    with col2:
                        external_label = st.toggle("Enable external legend")
                        custom_legend = None
                        if external_label:
                            custom_legend = st.text_area(
                                "Custom Legend Text (one entry per line). Press STRG+ENTER to apply.", 
                                height=100,
                                help="Leave empty to use sample names."
                            )

                # Generate the plot
                fig = generate_plots(
                    data_dict, custom_names, x_data_dict, 
                    st.session_state.plot_configs, 
                    external_label=external_label, 
                    custom_legend=custom_legend,
                    suptitle_enabled=suptitle_enabled,
                    suptitle=suptitle,
                    supaxes_enabled=supaxes_enabled,
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
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