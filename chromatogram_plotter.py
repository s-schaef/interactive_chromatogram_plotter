import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
 
# Set page config
st.set_page_config(page_title="Chromatogram Plotter", layout="wide")
 
# Main title
st.title("Chromatogram Plotter")
 
#### data functions
def process_file(uploaded_file):
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
        return None, "File size exceeds 200MB limit."
   
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
        df = pd.read_csv(uploaded_file, sep='\t', header=42) #TODO: check for off by one in function documentation
        if len(df.columns) != 3:
            return None, default_name, "File should have exactly 3 columns."
       
        return df.iloc[:, [0, 2]], default_name, None  # Return 1st and 3rd columns
    except Exception as e:
        return None, f"Error processing file: {str(e)}"
 
# File upload section
uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=['txt'])
 
# Process uploaded files
data_dict = {}
x_data = None
default_names = {}
 
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
for filename in data_dict.keys():
    custom_names[filename] = st.text_input(
        f"Custom name for {filename}",
        value=default_names[filename]
    )
 
### plotting function
def generate_plots(data_dict, custom_names, x_data, plot_configs):
    fig, axs = plt.subplots(len(plot_configs), 1, figsize=(10, 5*len(plot_configs)), squeeze=False)
   
    for i, config in enumerate(plot_configs):
        ax = axs[i, 0]
        for filename in config['files']:
            ax.plot(x_data, data_dict[filename], label=custom_names[filename])
        ax.set_title(f"Plot {i+1}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
   
    plt.tight_layout()
    return fig
 
# Plot configuration
st.header("Plot Configuration")
plot_configs = []
add_plot = st.button("Add Plot")
 
if add_plot:
    plot_configs.append({'files': []})
 
for i, config in enumerate(plot_configs):
    st.subheader(f"Plot {i+1}")
    config['files'] = st.multiselect(f"Select files for Plot {i+1}", options=list(data_dict.keys()), key=f"plot_{i}")
 
# Generate plot
if plot_configs:
    fig = generate_plots(data_dict, custom_names, x_data, plot_configs)
    st.pyplot(fig)
 
    # Download options
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    btn = st.download_button(
        label="Download Plot",
        data=buf.getvalue(),
        file_name="plot.png",
        mime="image/png"
    )
 
### export functions
def get_csv_download_link(data_dict, custom_names, x_data):
    output = io.StringIO()
    df = pd.DataFrame(data_dict)
    df.columns = [custom_names[col] for col in df.columns]
    df.insert(0, "X", x_data)
    df.to_csv(output, index=False)
    b64 = base64.b64encode(output.getvalue().encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
 
if data_dict:
    st.markdown(get_csv_download_link(data_dict, custom_names, x_data), unsafe_allow_html=True)