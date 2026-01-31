import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from math import pi
import os
from functools import lru_cache
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Indian Music Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# Load and cache data
@lru_cache(maxsize=1)
def load_all_csv_files(data_dir='data') -> dict:
    """Load all CSV files from data directory"""
    dataframes = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        file_key = csv_file.replace('.csv', '')
        dataframes[file_key] = pd.read_csv(file_path)
    
    return dataframes

@st.cache_data
def prepare_data(dfs):
    """Prepare and process data"""
    # Filter language dataframes
    language_dfs = {k: v for k, v in dfs.items() if k not in ['spotify_data clean', 'Old_songs']}
    language_dfs = dict(sorted(language_dfs.items()))
    
    # Add unique old songs to Hindi
    old_songs_df = dfs.get('Old_songs', pd.DataFrame())
    if not old_songs_df.empty:
        old_songs_set = set(old_songs_df['song_name'].dropna().str.lower())
        all_language_songs = set()
        for df in language_dfs.values():
            all_language_songs.update(df['song_name'].dropna().str.lower())
        
        unique_to_old = old_songs_df[old_songs_df['song_name'].str.lower().isin(old_songs_set - all_language_songs)]
        if len(unique_to_old) > 0:
            unique_to_old_copy = unique_to_old.copy()
            unique_to_old_copy['language'] = 'Hindi'
            language_dfs['Hindi_songs'] = pd.concat([language_dfs['Hindi_songs'], unique_to_old_copy], ignore_index=True)
    
    # Process dates
    for lang, df in language_dfs.items():
        df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')
    
    # Create combined dataframe
    combined_data = []
    for lang, df in language_dfs.items():
        lang_name = lang.replace('_songs', '')
        temp_df = df[['popularity', 'Stream']].copy()
        temp_df['language'] = lang_name
        combined_data.append(temp_df)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Calculate language stats
    language_stats = combined_df.groupby('language').agg({
        'popularity': ['mean', 'sum'],
        'Stream': ['mean', 'sum']
    }).reset_index()
    
    language_stats.columns = ['language', 'avg_popularity', 'total_popularity', 'avg_streams', 'total_streams']
    language_stats = language_stats.sort_values('avg_popularity', ascending=False)
    
    return language_dfs, combined_df, language_stats

@st.cache_data
def prepare_language_timeseries(dfs, language_key):
    """
    Prepare time series data for a specific language
    Returns DataFrame with 'ds' (date) and 'y' (popularity score)
    """
    if language_key not in dfs:
        return None
    
    df = dfs[language_key].copy()
    
    # Convert released_date to datetime
    df['released_date'] = pd.to_datetime(df['released_date'], format='%d-%m-%Y', errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['released_date'])
    
    # Group by month and calculate average popularity
    df['year_month'] = df['released_date'].dt.to_period('M')
    monthly_data = df.groupby('year_month').agg({
        'popularity': 'mean',
        'Stream': 'sum'
    }).reset_index()
    
    # Convert period back to timestamp
    monthly_data['ds'] = monthly_data['year_month'].dt.to_timestamp()
    monthly_data['y'] = monthly_data['popularity']
    
    # Select only required columns for Prophet
    prophet_df = monthly_data[['ds', 'y']].sort_values('ds')
    
    return prophet_df

@st.cache_data
def build_prophet_forecasts(language_dfs, forecast_periods=36):
    """Build Prophet models and forecast for each language"""
    major_languages = [
        'Hindi', 'Tamil', 'Telugu', 'Punjabi', 'Bengali', 
        'Kannada', 'Malayalam', 'Marathi', 'Gujarati', 'Urdu'
    ]
    
    language_data = {}
    forecasts = {}
    models_info = {}
    
    for lang in major_languages:
        lang_key = f"{lang}_songs"
        ts_data = prepare_language_timeseries(language_dfs, lang_key)
        if ts_data is not None and len(ts_data) > 10:
            language_data[lang] = ts_data
            
            # Initialize and fit the model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(ts_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
            
            # Make predictions
            forecast = model.predict(future)
            
            forecasts[lang] = forecast
            models_info[lang] = {
                'last_date': ts_data['ds'].max(),
                'first_date': ts_data['ds'].min(),
                'data_points': len(ts_data)
            }
    
    return language_data, forecasts, models_info

def main():
    st.title("🎵 Indian Music Analytics Dashboard")
    st.markdown("Comprehensive analysis of Indian songs across multiple languages and platforms")
    
    # Load data
    dfs = load_all_csv_files('data')
    language_dfs, combined_df, language_stats = prepare_data(dfs)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Section", 
        ["📊 Overview", "🎤 Artists & Language", "📈 Trends", "🎼 Audio Features", "📉 Detailed Analysis", "🔮 Forecasting"])
    
    # ============ OVERVIEW PAGE ============
    if page == "📊 Overview":
        st.header("Dataset Overview")
        
        # Calculate summary statistics
        total_songs = sum(len(df) for df in language_dfs.values())
        all_artists = set()
        for df in language_dfs.values():
            if 'singer' in df.columns:
                all_artists.update(df['singer'].dropna().unique())
        num_artists = len(all_artists)
        num_languages = len(language_dfs)
        
        all_years = []
        for df in language_dfs.values():
            years = df['released_date'].dt.year.dropna()
            all_years.extend(years.tolist())
        
        min_year = int(min(all_years)) if all_years else None
        max_year = int(max(all_years)) if all_years else None
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Songs", f"{total_songs:,}")
        with col2:
            st.metric("Unique Artists", f"{num_artists:,}")
        with col3:
            st.metric("Languages", num_languages)
        with col4:
            st.metric("Year Range", f"{min_year}-{max_year}")
        with col5:
            st.metric("Avg Popularity", f"{combined_df['popularity'].mean():.2f}")
        
        st.markdown("---")
        
        # Language distribution pie chart
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("Language Distribution")
            language_counts = {}
            for lang, df in language_dfs.items():
                lang_name = lang.replace('_songs', '')
                language_counts[lang_name] = len(df)
            
            language_counts = dict(sorted(language_counts.items(), key=lambda item: item[1], reverse=False))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette('husl', len(language_counts))
            wedges, texts, autotexts = ax.pie(
                language_counts.values(), 
                labels=language_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 9}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('light')
            
            ax.set_title('Language Distribution Across All Songs', fontsize=12, fontweight='light')
            st.pyplot(fig, width='content')
        
        with col2:
            st.subheader("Language Count")
            lang_df = pd.DataFrame(list(language_counts.items()), columns=['Language', 'Songs'])
            lang_df = lang_df.sort_values('Songs', ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(lang_df['Language'], lang_df['Songs'], color=sns.color_palette('viridis', len(lang_df)))
            ax.set_xlabel('Number of Songs')
            ax.set_title('Songs per Language')
            st.pyplot(fig, width='content')
    
    # ============ ARTISTS & LANGUAGE PAGE ============
    elif page == "🎤 Artists & Language":
        st.header("Artists & Language Analysis")
        
        # Top artists per language
        st.subheader("Top Artists per Language (by Popularity)")
        
        top_n = st.slider("Select number of top artists per language", 1, 10, 3)
        
        rows = []
        for lang, df in language_dfs.items():
            artist_col = 'singer'
            if 'popularity' not in df.columns:
                continue
            
            artist_rank = (
                df[[artist_col, 'popularity']]
                .dropna()
                .groupby(artist_col, as_index=False)['popularity']
                .sum()
                .sort_values('popularity', ascending=False)
                .head(top_n)
                .assign(language=lang.replace('_songs', ''))
                .rename(columns={artist_col: 'artist'})
            )
            rows.append(artist_rank)
        
        if rows:
            rank_df = pd.concat(rows, ignore_index=True)
            pivot_df = rank_df.pivot(index='language', columns='artist', values='popularity').fillna(0)
            pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]
            
            fig, ax = plt.subplots(figsize=(14, 8))
            pivot_df.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
            ax.set_title(f'Top {top_n} Artists per Language (by Popularity)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Popularity Score')
            ax.set_ylabel('Language')
            ax.legend(title='Artist', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, width='content')
        
        st.markdown("---")
        
        # Language stats table
        st.subheader("Language Statistics")
        display_stats = language_stats.copy()
        display_stats.columns = ['Language', 'Avg Popularity', 'Total Popularity', 'Avg Streams', 'Total Streams']
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(display_stats, width='content')
        with col2:
            st.metric("Highest Avg Popularity", display_stats.iloc[0]['Language'])
            st.metric("Most Total Streams", display_stats.loc[display_stats['Total Streams'].idxmax(), 'Language'])
    
    # ============ TRENDS PAGE ============
    elif page == "📈 Trends":
        st.header("Stream Trends Over Time")
        
        # Select languages to display
        selected_langs = st.multiselect(
            "Select languages to display",
            [lang.replace('_songs', '') for lang in language_dfs.keys()],
            default=[list(language_dfs.keys())[0].replace('_songs', '')])
        
        if selected_langs:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            for lang, df in sorted(language_dfs.items()):
                lang_display = lang.replace('_songs', '')
                if lang_display not in selected_langs:
                    continue
                
                df_clean = df.dropna(subset=['released_date', 'Stream'])
                if len(df_clean) == 0:
                    continue
                
                df_clean['year'] = df_clean['released_date'].dt.year
                yearly_popularity = df_clean.groupby('year', as_index=False)['Stream'].mean()
                
                ax.plot(yearly_popularity['year'], yearly_popularity['Stream'], 
                       marker='o', linewidth=2.5, label=lang_display, markersize=6)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Streams', fontsize=12, fontweight='bold')
            ax.set_title('Stream Trends Over Time by Language', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Genre trends
        st.subheader("Genre Trends Over Time")
        
        # Check if genre_time.png exists
        if os.path.exists('genre_time.png'):
            st.image('genre_time.png', use_column_width=True, caption="Genre Distribution Trends Over Time")
        else:
            st.info("Genre trends visualization not available. Ensure genre_time.png is in the project directory.")
    
    # ============ AUDIO FEATURES PAGE ============
    elif page == "🎼 Audio Features":
        st.header("Audio Features Analysis")
        
        audio_features = ['danceability', 'acousticness', 'energy', 'liveness', 'loudness', 'speechiness']
        
        # Select a language
        selected_lang = st.selectbox(
            "Select a language to view audio features",
            [lang.replace('_songs', '') for lang in language_dfs.keys()])
        
        if selected_lang:
            # Find the dataframe for selected language
            lang_key = None
            for k in language_dfs.keys():
                if k.replace('_songs', '') == selected_lang:
                    lang_key = k
                    break
            
            if lang_key:
                df = language_dfs[lang_key]
                top_3_songs = df.nlargest(3, 'popularity')[['song_name'] + audio_features].copy()
                
                if len(top_3_songs) > 0 and all(f in top_3_songs.columns for f in audio_features):
                    # Normalize loudness
                    if 'loudness' in top_3_songs.columns:
                        loudness_min = df['loudness'].min()
                        loudness_max = df['loudness'].max()
                        if loudness_max != loudness_min:
                            top_3_songs['loudness'] = (top_3_songs['loudness'] - loudness_min) / (loudness_max - loudness_min)
                    
                    # Create radar charts
                    num_songs = min(3, len(top_3_songs))
                    cols = st.columns(num_songs)
                    
                    num_vars = len(audio_features)
                    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
                    angles += angles[:1]
                    
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    
                    for col_idx, (col, (_, song)) in enumerate(zip(cols, top_3_songs.iterrows())):
                        with col:
                            fig = plt.figure(figsize=(5, 5))
                            ax = fig.add_subplot(111, projection='polar')
                            
                            values = song[audio_features].tolist()
                            values += values[:1]
                            
                            ax.plot(angles, values, 'o-', linewidth=2, color=colors[col_idx])
                            ax.fill(angles, values, alpha=0.25, color=colors[col_idx])
                            
                            ax.set_xticks(angles[:-1])
                            ax.set_xticklabels(audio_features, size=8)
                            ax.set_ylim(0, 1)
                            ax.set_title(f"#{col_idx + 1}: {song['song_name'][:25]}", size=10, fontweight='bold', pad=15)
                            ax.grid(True)
                            
                            st.pyplot(fig, width='content')
    
    # ============ DETAILED ANALYSIS PAGE ============
    elif page == "📉 Detailed Analysis":
        st.header("Detailed Analysis")
        
        # Popularity vs Streams Analysis
        st.subheader("Popularity vs Streams Correlation Analysis")
        
        colors = sns.color_palette('bright', len(language_stats))

        # Create bubble sizes
        bubble_sizes = (language_stats['total_streams'] / language_stats['total_streams'].max()) * 1000

        # Create scatter plot with colors
        for idx, row in language_stats.iterrows():
            plt.scatter(row['avg_popularity'], 
                    row['avg_streams'],
                    s=bubble_sizes.iloc[idx], 
                    alpha=0.3, 
                    color=colors[idx],
                    edgecolors='black', 
                    linewidth=2)
        
        legend_elements = [Patch(facecolor=colors[idx], edgecolor='black', label=row['language']) 
                        for idx, row in language_stats.iterrows()]

        plt.xlabel('Average Popularity', fontsize=13, fontweight='light')
        plt.ylabel('Average Streams', fontsize=13, fontweight='light')
        plt.title('Language Comparison: Popularity vs Streams\n(Bubble Size = Total Streams)', 
                fontsize=15, fontweight='light')
        plt.grid(True, alpha=0.3)
        plt.legend(handles=legend_elements, title='Language', bbox_to_anchor=(1.05, 1), 
                loc='upper left', fontsize=10, title_fontsize=11, frameon=True, shadow=True)
        plt.tight_layout()
        st.pyplot(plt.gcf(), width='content')
        
        st.markdown("---")
        
        # Dual axis chart
        st.subheader("Total Popularity vs Total Streams by Language")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        x = range(len(language_stats))
        width = 0.4
        
        bars1 = ax.bar([i - width/2 for i in x], language_stats['total_popularity'], 
                      width, label='Total Popularity', alpha=0.8, color='skyblue')
        
        ax2 = ax.twinx()
        bars2 = ax2.bar([i + width/2 for i in x], language_stats['total_streams'], 
                       width, label='Total Streams', alpha=0.8, color='coral')
        
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Popularity', fontsize=12, fontweight='bold', color='skyblue')
        ax2.set_ylabel('Total Streams', fontsize=12, fontweight='bold', color='coral')
        ax.set_title('Total Popularity vs Total Streams by Language', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(language_stats['language'], rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig, width='content')
        
        st.markdown("---")
        
        # Correlation statistics
        st.subheader("Correlation Coefficients")
        
        overall_corr = combined_df[['popularity', 'Stream']].corr()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Correlation (Popularity vs Streams)", 
                     f"{overall_corr.iloc[0, 1]:.4f}")
        
        with col2:
            st.info("👉 Per-language correlations shown below")
        
        # Create correlation table for each language
        corr_data = []
        for lang, df in sorted(language_dfs.items()):
            lang_name = lang.replace('_songs', '')
            df_clean = df[['popularity', 'Stream']].dropna()
            if len(df_clean) > 1:
                corr = df_clean['popularity'].corr(df_clean['Stream'])
                corr_data.append({'Language': lang_name, 'Correlation': f"{corr:.4f}"})
        
        if corr_data:
            corr_table = pd.DataFrame(corr_data)
            st.dataframe(corr_table, width='content')
    
    # ============ FORECASTING PAGE ============
    elif page == "🔮 Forecasting":
        st.header("Prophet Forecasting: Popularity Trends (2026-2029)")
        st.markdown("Using Facebook Prophet to forecast song popularity trends for major Indian languages")
        
        # Build forecasts (cached)
        with st.spinner("Building Prophet forecasting models..."):
            language_data, forecasts, models_info = build_prophet_forecasts(language_dfs)
        
        if not forecasts:
            st.error("No forecast data available. Ensure data files have sufficient historical data.")
            return
        
        # ===== FORECAST SUMMARY TABLE =====
        st.subheader("📊 Forecast Summary (Next 3 Years)")
        
        summary_data = []
        for lang, forecast in forecasts.items():
            last_actual_date = models_info[lang]['last_date']
            future_forecast = forecast[forecast['ds'] > last_actual_date]
            
            if len(future_forecast) > 0:
                avg_popularity = future_forecast['yhat'].mean()
                min_popularity = future_forecast['yhat'].min()
                max_popularity = future_forecast['yhat'].max()
                trend = future_forecast['yhat'].iloc[-1] - future_forecast['yhat'].iloc[0]
                
                summary_data.append({
                    'Language': lang,
                    'Avg Predicted Popularity': round(avg_popularity, 2),
                    'Min Predicted': round(min_popularity, 2),
                    'Max Predicted': round(max_popularity, 2),
                    'Overall Trend': round(trend, 2),
                    'Trend Direction': '📈 Increasing' if trend > 0 else '📉 Decreasing'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Avg Predicted Popularity', ascending=False)
        
        st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("---")
        
        # ===== TREND DIRECTION VISUALIZATION =====
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("📊 Popularity Trend Direction (2026-2029)")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            trend_colors = ['green' if x > 0 else 'red' for x in summary_df['Overall Trend']]
            bars = ax.barh(summary_df['Language'], summary_df['Overall Trend'], 
                          color=trend_colors, edgecolor='black', linewidth=1.5, alpha=0.7)
            
            for idx, (bar, trend) in enumerate(zip(bars, summary_df['Overall Trend'])):
                width = bar.get_width()
                ax.text(width + (0.5 if width > 0 else -0.5), bar.get_y() + bar.get_height()/2.,
                       f'{trend:.2f}',
                       ha='left' if width > 0 else 'right', 
                       va='center', fontsize=10, fontweight='bold')
            
            ax.axvline(x=0, color='black', linewidth=2)
            ax.set_xlabel('Trend (Change in Popularity)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Language', fontsize=12, fontweight='bold')
            ax.set_title('Popularity Trend Direction', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Key Insights")
            
            # Top increasing language
            top_increasing = summary_df[summary_df['Overall Trend'] > 0].nlargest(1, 'Overall Trend')
            if len(top_increasing) > 0:
                st.success(f"**Top Increasing:** {top_increasing.iloc[0]['Language']}")
                st.metric("Trend Score", f"+{top_increasing.iloc[0]['Overall Trend']:.2f}")
            
            # Top decreasing language
            top_decreasing = summary_df[summary_df['Overall Trend'] < 0].nsmallest(1, 'Overall Trend')
            if len(top_decreasing) > 0:
                st.error(f"**Top Decreasing:** {top_decreasing.iloc[0]['Language']}")
                st.metric("Trend Score", f"{top_decreasing.iloc[0]['Overall Trend']:.2f}")
            
            # Most stable
            summary_df_copy = summary_df.copy()
            summary_df_copy['abs_trend'] = summary_df_copy['Overall Trend'].abs()
            most_stable = summary_df_copy.nsmallest(1, 'abs_trend')
            if len(most_stable) > 0:
                st.info(f"**Most Stable:** {most_stable.iloc[0]['Language']}")
                st.metric("Trend Score", f"{most_stable.iloc[0]['Overall Trend']:.2f}")
        
        st.markdown("---")
        
        # ===== AVERAGE PREDICTED POPULARITY BAR CHART =====
        st.subheader("📊 Average Predicted Popularity by Language (2026-2029)")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(summary_df)))
        bars = ax.bar(summary_df['Language'], summary_df['Avg Predicted Popularity'], 
                     color=colors_bar, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_title('Average Predicted Popularity Score by Language', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Popularity Score', fontsize=12, fontweight='bold')
        ax.set_xticklabels(summary_df['Language'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ===== DETAILED FORECAST PLOTS =====
        st.subheader("📈 Detailed Language Forecasts")
        st.markdown("Select languages to view detailed forecast plots with historical data and future predictions")
        
        # Multi-select for languages
        selected_forecast_langs = st.multiselect(
            "Select languages to display detailed forecasts",
            list(forecasts.keys()),
            default=list(forecasts.keys())[:3] if len(forecasts) >= 3 else list(forecasts.keys())
        )
        
        if selected_forecast_langs:
            for lang in selected_forecast_langs:
                st.markdown(f"### {lang}")
                
                forecast = forecasts[lang]
                actual_data = language_data[lang]
                last_actual_date = models_info[lang]['last_date']
                
                # Split forecast
                historical = forecast[forecast['ds'] <= last_actual_date]
                future = forecast[forecast['ds'] > last_actual_date]
                
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Plot actual data
                ax.scatter(actual_data['ds'], actual_data['y'], 
                          color='black', s=30, alpha=0.6, label='Actual Data', zorder=3)
                
                # Plot historical forecast
                ax.plot(historical['ds'], historical['yhat'], 
                       color='blue', linewidth=2.5, label='Historical Fit', alpha=0.8)
                ax.fill_between(historical['ds'], 
                               historical['yhat_lower'], 
                               historical['yhat_upper'],
                               alpha=0.2, color='blue', label='Confidence Interval (Historical)')
                
                # Plot future forecast
                ax.plot(future['ds'], future['yhat'], 
                       color='red', linewidth=2.5, linestyle='--', label='Future Forecast', alpha=0.9)
                ax.fill_between(future['ds'], 
                               future['yhat_lower'], 
                               future['yhat_upper'],
                               alpha=0.2, color='red', label='Confidence Interval (Future)')
                
                # Add vertical line at prediction start
                ax.axvline(x=last_actual_date, color='green', 
                          linestyle=':', linewidth=2.5, label='Forecast Start', alpha=0.7)
                
                ax.set_title(f'{lang} - Popularity Forecast (Next 3 Years)', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Popularity Score', fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=10, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data Points", models_info[lang]['data_points'])
                with col2:
                    st.metric("Date Range", f"{models_info[lang]['first_date'].strftime('%Y-%m')}")
                with col3:
                    st.metric("Last Actual", f"{last_actual_date.strftime('%Y-%m')}")
                with col4:
                    future_avg = future['yhat'].mean()
                    st.metric("Avg Future Pop", f"{future_avg:.2f}")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
