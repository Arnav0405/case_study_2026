# 🎵 Indian Music Analytics Dashboard

A comprehensive interactive Streamlit dashboard analyzing Indian songs across multiple languages and platforms.

## Features

The dashboard includes five main sections:

### 📊 Overview

- **Dataset Summary**: Total songs, unique artists, languages, and year range
- **Language Distribution**: Pie chart and bar chart showing song distribution across languages
- **Key Metrics**: Average popularity, stream counts, and language statistics

### 🎤 Artists & Language

- **Top Artists by Language**: Interactive visualization of top performing artists per language
- **Adjustable Rankings**: Slider to adjust the number of top artists displayed (1-10)
- **Language Statistics**: Comprehensive table showing average and total popularity/streams per language

### 📈 Trends

- **Stream Trends Over Time**: Line charts showing how average streams evolved across years
- **Multi-Language Comparison**: Select multiple languages to compare trends side-by-side
- **Year-wise Analysis**: Track how different languages performed over time

### 🎼 Audio Features

- **Radar Charts**: Visualization of audio features for top 3 songs per language
- **Features Analyzed**:
  - Danceability
  - Acousticness
  - Energy
  - Liveness
  - Loudness
  - Speechiness

### 📉 Detailed Analysis

- **Scatter Plot**: Average popularity vs average streams by language
- **Correlation Heatmap**: Correlations between popularity and streams metrics
- **Dual-Axis Chart**: Total popularity and streams comparison by language
- **Correlation Coefficients**: Overall and per-language correlation statistics

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure all data files are present in the `data/` directory:
   - Assamese_songs.csv
   - Bengali_songs.csv
   - Bhojpuri_songs.csv
   - Gujarati_songs.csv
   - Haryanvi_songs.csv
   - Hindi_songs.csv
   - Kannada_songs.csv
   - Malayalam_songs.csv
   - Marathi_songs.csv
   - Odia_songs.csv
   - Old_songs.csv
   - Punjabi_songs.csv
   - Rajasthani_songs.csv
   - Tamil_songs.csv
   - Telugu_songs.csv
   - Urdu_songs.csv

## Running the Dashboard

```bash
streamlit run main.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Usage

1. **Navigation**: Use the sidebar radio buttons to switch between different sections
2. **Interactivity**:
   - Adjust sliders to filter top artists
   - Select languages from multiselect dropdowns
   - Hover over charts for detailed tooltips
3. **Data Export**: Download visualizations using the camera icon in the top-right of each chart

## Data Processing

The application:

- Loads all CSV files from the `data/` directory
- Merges unique songs from Old_songs.csv into Hindi_songs
- Converts date strings to datetime format
- Calculates aggregate statistics by language
- Caches data for optimal performance

## Technologies Used

- **Streamlit**: Interactive web framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **NumPy**: Numerical computations

## Project Structure

```
case-study/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
├── data/                  # CSV data files
│   ├── Assamese_songs.csv
│   ├── Bengali_songs.csv
│   └── ... (other language CSVs)
├── arnav_visualisations.ipynb  # Source Jupyter notebook
└── asd.ipynb              # Additional analysis notebook
```

## Key Insights

The dashboard reveals:

- Distribution of songs across 16 Indian language datasets
- Correlation patterns between popularity and streaming metrics
- Artist performance rankings by language
- Temporal trends in streaming behavior
- Audio feature characteristics of popular songs

## Notes

- Invalid dates are automatically handled with `pd.to_datetime(..., errors='coerce')`
- Missing values in audio features are handled appropriately
- Loudness values are normalized to 0-1 range for consistent visualization
- All charts are responsive and scale based on available space
