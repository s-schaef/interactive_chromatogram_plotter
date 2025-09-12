# Chromatogram Plotter

A Streamlit web application for visualizing and analyzing chromatography data files. Upload multiple chromatogram files, customize sample names, create multiple plots, and export your processed data.

Access the application directly through Streamlit Cloud - no installation required!


## Features

- **Multiple File Upload**: Process multiple chromatogram text files simultaneously
- **Automatic Sample Name Detection**: Extracts default sample names from file metadata
- **Custom Naming**: Override default names with custom labels for each sample
- **Flexible Plotting**: Create multiple plots with different file combinations
- **Data Export**: Download processed data as CSV or save plots as PNG images
- **Large File Support**: Handles files up to 200MB

## File Format Requirements

### Input File Structure
- **Format**: Tab-separated values (`.txt` extension)
- **Header**: Data should start at row 43 (42 header rows)
- **Sample Name**: Located in row 6, column 2 (optional, after "Injection" label)



## How to Use

1. **Upload Files**
   - Click "Browse files" or drag and drop your chromatogram `.txt` files
   - Multiple files can be uploaded at once

2. **Customize Names**
   - Review the automatically detected sample names
   - Enter custom names in the text fields if desired

3. **Configure Plots**
   - Click "Add Plot" to create a new plot panel
   - Select which files to display on each plot using the multiselect dropdown
   - Create multiple plots to compare different sample combinations

4. **Export Results**
   - Download plots as PNG images using the "Download Plot" button
   - Export all processed data as a CSV file with custom sample names

## Technical Details

### Dependencies
- `streamlit` - Web application framework
- `pandas` - Data processing and manipulation
- `matplotlib` - Plot generation
- Python 3.7+

### File Processing
- Automatically extracts sample names from file metadata
- Validates file format and size constraints
- Handles missing or malformed data gracefully

### Data Structure
- X-axis data is taken from the first uploaded file
- All files must have the same number of data points
- Y-axis data from each file is plotted against the common X-axis

## Limitations

- Maximum file size: 200MB per file
- File format: Only `.txt` files with tab-separated values
- Column requirement: Exactly 3 columns per file
- Data alignment: All files must have the same X-axis range and number of points

## Error Handling

The application will display error messages for:
- Files exceeding the 200MB size limit
- Files with incorrect number of columns
- Files that cannot be parsed as tab-separated values
- Missing or corrupted data


## Support
For questions, suggestions, or issues, please contact Stefan Schaefer. 

Version History
v1.0 (Current)
Initial release
Multiple file processing
Internal Sanofi use only

Developed by Stefan Schaefer
Last Updated: September 2025