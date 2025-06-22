
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, jarque_bera,skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.stats import f_oneway


#  Load the Dataset
# Load the dataset
try:
    df = pd.read_csv('instoreweekly.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Table_04.csv not found. Please ensure the file is in the correct directory.")
    exit()

IMPORTANT_COLS = [
    'wm_year', 'wm_store_id', 'wm_city', 'wm_region', 'dept_nbr', 'dept_subcatg_nbr',
    'fineline_nbr', 'brand_type', 'brand_id', 'brand_desc', 'vendor_id', 'vendor_desc',
    'product_type', 'id', 'upc_nbr', 'item_nbr', 'upc_desc', 'sell_size', 'uom',
    'item_status', 'base_unit_retail_amt', 'wm_reg_price', 'wm_live_price',
    'wm_price_type', 'banner_name', 'store_no', 'comp_reg', 'comp_live',
    'reg_price_gap', 'live_price_gap', 'reg_price_gap_perc', 'live_price_gap_perc',
    'item_win', 'item_draw', 'item_loss', 'comp_reg_equalized_price',
    'comp_live_equalized_price', 'match_flag', 'equalization_factor', 'wm_week'
]

existing_cols = [col for col in IMPORTANT_COLS if col in df.columns]
print(f"Loading {len(existing_cols)} specified columns.")
df = df[existing_cols]


#  Prepare the Data

numeric_cols = [
    'base_unit_retail_amt', 'wm_reg_price', 'wm_live_price', 'comp_reg', 'comp_live',
    'reg_price_gap', 'live_price_gap', 'reg_price_gap_perc', 'live_price_gap_perc',
    'comp_reg_equalized_price', 'comp_live_equalized_price', 'equalization_factor',
    'wm_week', 'sell_size'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()


#  Level 1: EDA Basics

# # 1.1 Basic Data Overview

print("="*60)
print("LEVEL 1: BASIC EDA CHECKS")
print("="*60)

print("\n📋 1.1 DATA OVERVIEW & STRUCTURE")
print("-" * 40)
print(f"Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Numeric Columns: {len(numeric_cols)}")
print(f"Categorical Columns: {len(categorical_cols)}")
print("\nColumn Types:")
print(df.dtypes.value_counts())

df.head(3)

df.info()


# # 1.2 Missing Data Assessment
print("\n🔍 1.2 MISSING DATA ASSESSMENT")
print("-" * 40)

missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_percent
}).sort_values('Missing_Percentage', ascending=False)

print("All columns with missing data:")
print(missing_df[missing_df.Missing_Count > 0])

if missing_df.Missing_Count.sum() > 0:
    plt.figure(figsize=(max(12, len(missing_df) * 0.3), 6))
    missing_cols = missing_df[missing_df.Missing_Count > 0]
    plt.bar(range(len(missing_cols)), missing_cols.Missing_Percentage)
    plt.title('Missing Data Percentage by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(range(len(missing_cols)), missing_cols.index, rotation=90, ha='right')
    plt.tight_layout()
    plt.show()


# # 1.3 Descriptive Statistics
print("\n📊 1.3 BASIC DESCRIPTIVE STATISTICS")
print("-" * 40)

if numeric_cols:
    print("NUMERIC VARIABLES SUMMARY:")
    print(df[numeric_cols].describe().round(2))

print("\nCATEGORICAL VARIABLES SUMMARY:")
key_categorical = ['wm_region', 'brand_type', 'banner_name', 'item_win', 'wm_price_type']
for col in key_categorical:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head())


# #  1.4 Data Quality Checks
print("\n🛡️ 1.4 BASIC DATA QUALITY CHECKS")
print("-" * 40)

duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates:,}")

price_cols = ['base_unit_retail_amt', 'wm_reg_price', 'wm_live_price', 'comp_reg', 'comp_live']
for col in price_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        print(f"Negative values in {col}: {negative_count}")

print("\nExtreme outliers (>3σ):")
for col in numeric_cols:
    if df[col].dtype in ['int64', 'float64']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        outliers = ((df[col] - mean_val).abs() > 3 * std_val).sum()
        if outliers > 0:
            print(f"{col}: {outliers} outliers")


# # 1.5 Numeric and Categorical Distributions
# ----- NUMERIC DISTRIBUTIONS -----
print("\n📈 1.5 NUMERIC DISTRIBUTIONS")
price_cols = ['base_unit_retail_amt', 'wm_reg_price', 'comp_reg', 'live_price_gap']
available_price_cols = [col for col in price_cols if col in df.columns]

if available_price_cols:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(available_price_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('Price / Gap')
        axes[i].set_ylabel('Frequency')

    # Hide unused axes
    for j in range(len(available_price_cols), 8):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ----- CATEGORICAL DISTRIBUTIONS -----
print("\n📊 1.5 CATEGORICAL DISTRIBUTIONS")
cat_cols = ['wm_region', 'brand_type', 'wm_price_type', 'banner_name']
available_cat_cols = [col for col in cat_cols if col in df.columns]

if available_cat_cols:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(available_cat_cols):
        value_counts = df[col].value_counts().head(10)
        axes[i].bar(range(len(value_counts)), value_counts.values)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')

    # Hide unused axes
    for j in range(len(available_cat_cols), 8):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


#  Level 2: EDA Analysis

# 2.1 TEMPORAL ANALYSIS
print("\n📅 2.1 TEMPORAL ANALYSIS")
print("-" * 40)

if 'wm_week' in df.columns:
    weekly_data = df.groupby('wm_week').agg({
        'base_unit_retail_amt': ['mean', 'count'],
        'comp_reg': 'mean',
        'reg_price_gap': 'mean'
    }).round(2)

    print("Weekly trends (first 10 weeks):")
    print(weekly_data.head(10))

    if 'base_unit_retail_amt' in df.columns:
        plt.figure(figsize=(14, 4))
        weekly_prices = df.groupby('wm_week')['base_unit_retail_amt'].mean()
        plt.plot(weekly_prices.index, weekly_prices.values, marker='o', linewidth=2)
        plt.title('Average Price Trend Over Time')
        plt.xlabel('Week Number')
        plt.ylabel('Average Price')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# # 2.2 Competitive Analysis
print("\n🏆 2.2 COMPETITIVE ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['base_unit_retail_amt', 'comp_reg']):
    corr_coef = df['base_unit_retail_amt'].corr(df['comp_reg'])
    print(f"Price correlation (WM's vs Competitor): {corr_coef:.3f}")

    if 'reg_price_gap' in df.columns:
        gap_stats = df['reg_price_gap'].describe()
        print("\nPrice Gap Statistics:")
        print(gap_stats.round(2))

        if 'item_win' in df.columns:
            win_rate = (df['item_win'] == 'Y').mean() * 100
            loss_rate = (df['item_loss'] == 'Y').mean() * 100
            draw_rate = (df['item_draw'] == 'Y').mean() * 100
            print(f"\nWin Rate: {win_rate:.1f}%")
            print(f"Loss Rate: {loss_rate:.1f}%")
            print(f"Draw Rate: {draw_rate:.1f}%")

if 'banner_name' in df.columns and 'comp_reg' in df.columns:
    # The data is grouped by 'banner_name', so the plot will use these names.
    competitor_analysis = df.groupby('banner_name').agg({
        'comp_reg': ['mean', 'count', 'std'],
        'reg_price_gap': 'mean'
    }).round(2)
    print("\nCompetitor Analysis by Banner:")
    print(competitor_analysis)

    # Plotting the average price gap by competitor
    plt.figure(figsize=(14, 4))
    # The plot uses the index of the `competitor_analysis` DataFrame for the x-axis labels,
    # which is the 'banner_name' from your data.
    competitor_analysis[('reg_price_gap', 'mean')].plot(kind='bar', color='skyblue')
    plt.title('Average Price Gap by Competitor')
    plt.ylabel('Average Price Gap')
    plt.xlabel('Banner Name') # Changed label for clarity
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout() # Adjust layout to make room for labels
    plt.show()


# 2.3 SEGMENTATION ANALYSIS
print("\n🎯 2.3 SEGMENTATION ANALYSIS")
print("-" * 40)

# --- Regional Analysis ---
if 'wm_region' in df.columns and 'base_unit_retail_amt' in df.columns:
    regional_analysis = df.groupby('wm_region').agg({
        'base_unit_retail_amt': ['mean', 'median', 'count', 'std'],
        'reg_price_gap': 'mean'
    }).round(2)

    print("Regional Analysis:")
    print(regional_analysis)

    # --- Improved Plots for Regional Analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle('Regional Segmentation Analysis', fontsize=16)

    # Plot 1: Price Distribution by Region (Box Plot)
    sns.boxplot(x='wm_region', y='base_unit_retail_amt', data=df, ax=axes[0], palette="Set2")
    axes[0].set_title('Price Distribution by Region')
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('Base Unit Retail Price')
    axes[0].tick_params(axis='x', rotation=0)

    # Plot 2: Average Price Gap by Region (Bar Plot)
    regional_analysis[('reg_price_gap', 'mean')].plot(kind='bar', ax=axes[1], color=sns.color_palette("Set2"))
    axes[1].set_title('Average Price Gap by Region')
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel('Average Price Gap')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


# --- Brand Type Analysis ---
if 'brand_type' in df.columns:
    brand_analysis = df.groupby('brand_type').agg({
        'base_unit_retail_amt': ['mean', 'count'],
        'reg_price_gap': 'mean'
    }).round(2)

    print("\nBrand Type Analysis:")
    print(brand_analysis)

    # --- Plots for Brand Type Analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('Brand Type Segmentation Analysis', fontsize=16)

    # Plot 1: Average Price by Brand Type
    sns.barplot(x=brand_analysis.index, y=('base_unit_retail_amt', 'mean'), data=brand_analysis, ax=axes[0], palette="Pastel1")
    axes[0].set_title('Average Price by Brand Type')
    axes[0].set_xlabel('Brand Type')
    axes[0].set_ylabel('Average Retail Price')

    # Plot 2: Average Price Gap by Brand Type
    sns.barplot(x=brand_analysis.index, y=('reg_price_gap', 'mean'), data=brand_analysis, ax=axes[1], palette="Pastel2")
    axes[1].set_title('Average Price Gap by Brand Type')
    axes[1].set_xlabel('Brand Type')
    axes[1].set_ylabel('Average Price Gap')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# 2.4 STATISTICAL RELATIONSHIPS
print("\n📊 2.4 STATISTICAL RELATIONSHIPS")
print("-" * 40)

# --- Create a combined figure for statistical plots ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Statistical Relationship Analysis', fontsize=20)


# --- Plot 1: Correlation Matrix ---
price_numeric_cols = [col for col in numeric_cols
                      if any(keyword in col.lower() for keyword in ['price', 'gap', 'amt', 'metric'])]

if len(price_numeric_cols) > 1:
    correlation_matrix = df[price_numeric_cols].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, fmt='.2f', ax=axes[0])
    axes[0].set_title('Price Variables Correlation Matrix', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    print("Strong correlations (|r| > 0.5):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"- {correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_val:.3f}")


# --- Plot 2: T-test and Distribution Visualization ---
if all(col in df.columns for col in ['brand_type', 'base_unit_retail_amt']):
    pl_prices = df[df['brand_type'] == 'PL']['base_unit_retail_amt'].dropna()
    nb_prices = df[df['brand_type'] == 'NB']['base_unit_retail_amt'].dropna()

    if len(pl_prices) > 10 and len(nb_prices) > 10:
        # Perform T-test
        t_stat, p_value = stats.ttest_ind(pl_prices, nb_prices)
        print(f"\nT-test (PL vs NB prices):")
        print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

        # Create Violin Plot
        sns.violinplot(x='brand_type', y='base_unit_retail_amt', data=df, ax=axes[1], palette='Pastel1', order=['PL', 'NB'])
        sns.stripplot(x='brand_type', y='base_unit_retail_amt', data=df, ax=axes[1], color=".25", size=3, order=['PL', 'NB'])

        axes[1].set_title('Price Distribution: Private Label vs. National Brand', fontsize=14)
        axes[1].set_xlabel('Brand Type')
        axes[1].set_ylabel('Base Unit Retail Price')

        # Annotate with T-test results
        significance = 'Statistically Significant' if p_value < 0.05 else 'Not Significant'
        axes[1].text(0.5, 0.9, f'T-statistic: {t_stat:.2f}\nP-value: {p_value:.3f}\n({significance})',
                     horizontalalignment='center', transform=axes[1].transAxes,
                     fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

# Adjust layout and display the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 2.5 Distribution Analysis
print("\n📈 2.5 DISTRIBUTION ANALYSIS")
print("-" * 40)

# Adjusted key_vars based on your dataframe's available columns
key_vars = ['base_unit_retail_amt', 'reg_price_gap', 'comp_reg', 'live_price_gap']

# --- Plotting Code Addition ---
plottable_vars = [var for var in key_vars if var in df.columns and pd.api.types.is_numeric_dtype(df[var])]
num_plots = len(plottable_vars)

if num_plots > 0:
    # Determine grid size for the plots
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # --- Modified Loop for Analysis and Plotting ---
    for i, var in enumerate(plottable_vars):
        ax = axes[i]
        data = df[var].dropna()

        # Adjusted the length check for your smaller dataset size
        if len(data) > 1:
            # --- Analysis Logic ---
            # Note: With very few data points, these stats may not be stable.
            s = skew(data)
            k = kurtosis(data)
            # Normality test requires at least 8 samples
            if len(data) >= 8:
                _, p_value = normaltest(data)
                normality_result = 'No' if p_value < 0.05 else 'Possibly'
                p_value_text = f"{p_value:.3f}"
            else:
                p_value_text = "N/A (<8 samples)"
                normality_result = "N/A"


            print(f"\n{var} Distribution Analysis:")
            print(f"  Skewness: {s:.3f}")
            print(f"  Kurtosis: {k:.3f}")
            print(f"  Normality test p-value: {p_value_text}")
            print(f"  Normal distribution: {normality_result}")

            # --- Plotting ---
            sns.histplot(data, kde=True, ax=ax, bins=5, color=sns.color_palette("viridis", num_plots)[i])
            ax.set_title(f"Distribution of '{var}'", fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Frequency')
            
            # Add statistical annotations to the plot
            stats_text = (f"Skew: {s:.2f}\n"
                          f"Kurtosis: {k:.2f}\n"
                          f"P-Val: {p_value_text}")
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.8))
        else:
             print(f"\nNot enough data to analyze or plot '{var}'.")
             ax.text(0.5, 0.5, "Not enough data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_title(f"Distribution of '{var}'", fontsize=12)

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Distribution Analysis of Key Variables', fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()

else:
    print("No key variables found in the DataFrame to analyze or plot.")

print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")


#  Level 3: EDA Analysis Advanced

# # 3.1 Advanced Temporal Exploration

# 3.1 ADVANCED TEMPORAL EXPLORATION
print("\n⏰ 3.1 ADVANCED TEMPORAL EXPLORATION")
print("-" * 40)

# Check if the necessary columns exist in the DataFrame
if 'wm_week' in df.columns and 'base_unit_retail_amt' in df.columns:
    # Group by week and calculate the mean price
    weekly_series = df.groupby('wm_week')['base_unit_retail_amt'].mean()

    # Ensure there's enough data for meaningful analysis (at least one full year)
    if len(weekly_series) >= 52:
        # --- Time Series Analysis ---
        weeks_numeric = np.arange(len(weekly_series))
        # Calculate coefficients for the linear trend line (slope and intercept)
        trend_coeffs = np.polyfit(weeks_numeric, weekly_series.values, 1)
        trend_line = trend_coeffs[0] * weeks_numeric + trend_coeffs[1]
        print(f"Weekly trend coefficient (slope): {trend_coeffs[0]:.4f}")

        # Calculate a 4-week rolling average to smooth the series
        rolling_avg = weekly_series.rolling(window=4).mean()

        # --- Seasonality Analysis ---
        weekly_series_df = weekly_series.reset_index()
        # Use modulo to find the week of the year for identifying seasonality
        weekly_series_df['week_of_year'] = (weekly_series_df['wm_week'] - 1) % 52 + 1
        seasonal_pattern = weekly_series_df.groupby('week_of_year')['base_unit_retail_amt'].mean()
        
        # Find peak and trough of the seasonal cycle
        peak_week = seasonal_pattern.idxmax()
        peak_value = seasonal_pattern.max()
        trough_week = seasonal_pattern.idxmin()
        trough_value = seasonal_pattern.min()

        # --- Enhanced Plotting ---
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

        # Plot 1: Time Series with Trend and Rolling Average
        ax1 = axes[0]
        ax1.plot(weekly_series.index, weekly_series.values, label='Weekly Avg Price', color='cornflowerblue', alpha=0.8)
        ax1.plot(weekly_series.index, rolling_avg, label='4-Week Rolling Avg', color='orange', linestyle='--')
        ax1.plot(weekly_series.index, trend_line, label='Linear Trend', color='crimson', linestyle=':')
        ax1.set_title('Weekly Price Time Series with Trend Analysis', fontsize=14, weight='bold')
        ax1.set_xlabel('Week Number')
        ax1.set_ylabel('Average Price')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot 2: Enhanced Seasonal Pattern
        ax2 = axes[1]
        ax2.plot(seasonal_pattern.index, seasonal_pattern.values, marker='.', linestyle='-', label='Average Seasonal Price', color='seagreen')
        # Highlight the peak and trough points
        ax2.plot(peak_week, peak_value, 'o', color='red', markersize=10, label=f'Peak Week: {peak_week}')
        ax2.plot(trough_week, trough_value, 'o', color='blue', markersize=10, label=f'Trough Week: {trough_week}')
        ax2.set_title('Seasonal Pattern (52-Week Cycle)', fontsize=14, weight='bold')
        ax2.set_xlabel('Week of Year')
        ax2.set_ylabel('Average Price')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.suptitle('Advanced Temporal Exploration of Price', fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # --- Volatility Calculation ---
        weekly_cv = df.groupby('wm_week')['base_unit_retail_amt'].apply(
            lambda x: x.std() / x.mean() if x.mean() != 0 else 0
        )
        print(f"\nAverage weekly coefficient of variation (volatility): {weekly_cv.mean():.3f}")

    else:
        print("Not enough weekly data (less than 52 weeks) for a full temporal analysis.")
else:
    print("Columns 'wm_week' or 'base_unit_retail_amt' not found in the DataFrame.")


# # 3.2 Multi-Dimensional Analysis
print("\n🔄 3.2 MULTI-DIMENSIONAL ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['wm_region', 'brand_type', 'base_unit_retail_amt']):
    agg_dict = {
        'base_unit_retail_amt': ['mean', 'count', 'std']
    }
    # Add reg_price_gap to aggregation if it exists
    if 'reg_price_gap' in df.columns:
        agg_dict['reg_price_gap'] = 'mean'
        
    if 'item_win' in df.columns:
        agg_dict['item_win'] = lambda x: (x == 'Y').mean()

    three_way = df.groupby(['wm_region', 'brand_type']).agg(agg_dict).round(2)
    print("Three-way Analysis (Region × Brand Type):")
    print(three_way.head(10))

    # Create an enhanced heatmap if the price gap data is available
    if 'reg_price_gap' in df.columns:
        # --- Enhanced Heatmap for Interaction Analysis ---
        sns.set_theme(style="white", font_scale=1.1)
        plt.figure(figsize=(6, 4))

        # Pivot data to get regions as rows and brand types as columns
        interaction_data = df.pivot_table(
            values='reg_price_gap',
            index='wm_region',
            columns='brand_type',
            aggfunc='mean'
        )

        # Create the heatmap with improved aesthetics and annotations
        heatmap = sns.heatmap(
            interaction_data,
            annot=True,          # Display data values on the heatmap
            fmt=".2f",           # Format annotations to two decimal places
            cmap='RdBu_r',       # Use a diverging colormap (Red-Blue reversed)
            center=0,            # Center the colormap at zero to highlight positive/negative gaps
            linewidths=.5,       # Add subtle lines between cells for clarity
            cbar_kws={'label': 'Average Price Gap'} # Add a label to the color bar
        )

        # Set a more descriptive title, subtitle, and labels
        plt.title('Price Gap Interaction: Region vs. Brand Type', fontsize=16, weight='bold', pad=20)
        plt.suptitle('Heatmap of Average Price Gap', fontsize=12, y=0.93)
        plt.xlabel('Brand Type', weight='bold')
        plt.ylabel('Walmart Region', weight='bold')
        
        # Ensure axis labels are not rotated for better readability
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)

        # Adjust layout to prevent title overlap and show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Perform panel data analysis if the required columns are present
if all(col in df.columns for col in ['wm_store_id', 'wm_week', 'base_unit_retail_amt']):
    store_means = df.groupby('wm_store_id')['base_unit_retail_amt'].mean()
    
    # Calculate between-store and within-store variance
    between_store_var = store_means.var()
    within_store_var = df.groupby('wm_store_id')['base_unit_retail_amt'].var().mean()
    
    print(f"\nPanel Data Variation Analysis:")
    print(f"Between-store variance: {between_store_var:.3f}")
    print(f"Within-store variance: {within_store_var:.3f}")
    
    # Calculate and print the variance ratio if within-store variance is non-zero
    if within_store_var > 0:
        print(f"Variance ratio (between/within): {between_store_var / within_store_var:.3f}")


#  # 3.3 Advanced Pattern Recognition
print("\n🔍 3.3 ADVANCED PATTERN RECOGNITION")
print("-" * 40)

# Check if necessary columns for cycle analysis are present
if all(col in df.columns for col in ['wm_week', 'base_unit_retail_amt']):
    weekly_prices = df.groupby('wm_week')['base_unit_retail_amt'].mean()

    # Ensure there's enough data for meaningful analysis (at least 3 months)
    if len(weekly_prices) >= 12:
        # --- Peak and Trough Detection ---
        # Find peaks (local maxima) and troughs (local minima) in the price series
        # The 'distance' parameter helps prevent finding too many minor fluctuations
        peaks, _ = find_peaks(weekly_prices.values, distance=4, height=weekly_prices.mean())
        troughs, _ = find_peaks(-weekly_prices.values, distance=4, height=-weekly_prices.mean())

        print(f"Price peaks detected (above average): {len(peaks)}")
        print(f"Price troughs detected (below average): {len(troughs)}")

        # Calculate the average cycle length if multiple peaks are found
        if len(peaks) > 1:
            avg_peak_distance = np.mean(np.diff(peaks))
            print(f"Average cycle length (peak-to-peak): {avg_peak_distance:.1f} weeks")

        # --- Enhanced Visualization of Price Cycles ---
        sns.set_theme(style="whitegrid", palette="muted")
        plt.figure(figsize=(16, 8))

        # Plot the main weekly price line
        plt.plot(weekly_prices.index, weekly_prices.values, linewidth=2.5, alpha=0.8, label='Average Weekly Price')

        # Highlight peaks with distinct markers
        if len(peaks) > 0:
            plt.scatter(weekly_prices.index[peaks], weekly_prices.values[peaks], 
                        color=sns.color_palette("bright")[2], s=150, marker='^', 
                        label='Peaks', zorder=5, edgecolors='black', linewidth=1)
        # Highlight troughs
        if len(troughs) > 0:
            plt.scatter(weekly_prices.index[troughs], weekly_prices.values[troughs], 
                        color=sns.color_palette("bright")[0], s=150, marker='v',
                        label='Troughs', zorder=5, edgecolors='black', linewidth=1)
        
        # Add a horizontal line for the mean price to provide context
        mean_price = weekly_prices.mean()
        plt.axhline(mean_price, color='grey', linestyle='--', linewidth=1.5, 
                    label=f'Mean Price: ${mean_price:.2f}')

        # Improve plot titles and labels for clarity
        plt.title('Advanced Price Cycle Analysis', fontsize=18, weight='bold', pad=20)
        plt.suptitle('Identifying Significant Peaks and Troughs in Weekly Average Price', y=0.92)
        plt.xlabel('Week Number', fontsize=12, weight='bold')
        plt.ylabel('Average Price ($)', fontsize=12, weight='bold')
        
        # Enhance legend and grid
        plt.legend(loc='upper left', frameon=True, fontsize=11, title='Legend')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Final layout adjustment and display
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()

# Check for columns needed for correlation analysis
if all(col in df.columns for col in ['wm_week', 'base_unit_retail_amt', 'comp_reg']):
    # Prepare weekly data for Walmart and competitor
    weekly_wm = df.groupby('wm_week')['base_unit_retail_amt'].mean()
    weekly_comp = df.groupby('wm_week')['comp_reg'].mean()
    
    # Calculate week-over-week price changes
    wm_changes = weekly_wm.diff()
    comp_changes = weekly_comp.diff()

    # Ensure there's enough data to calculate correlations
    if len(wm_changes.dropna()) > 10 and len(comp_changes.dropna()) > 10:
        # --- Lagged Correlation Analysis ---
        correlation = wm_changes.corr(comp_changes)
        print(f"\nPrice Change Correlation (Walmart vs. Competitor):")
        print(f"Concurrent Correlation: {correlation:.3f}")

        # Lagged correlation: Does Walmart's price change predict the competitor's next week?
        wm_leads_corr = comp_changes.corr(wm_changes.shift(1))
        # Lagged correlation: Does the competitor's price change predict Walmart's next week?
        comp_leads_corr = wm_changes.corr(comp_changes.shift(1))
        
        print(f"Walmart leads by 1 week (Walmart's current change vs. Competitor's last week): {comp_leads_corr:.3f}")
        print(f"Competitor leads by 1 week (Competitor's current change vs. Walmart's last week): {wm_leads_corr:.3f}")


# # 3.4 Advanced Statistical Analysis
print("\n📊 3.4 ADVANCED STATISTICAL ANALYSIS")
print("-" * 40)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
price_vars = [col for col in numeric_cols 
              if any(keyword in col.lower() for keyword in ['price', 'gap', 'amt'])]

if len(price_vars) >= 3:
    price_data = df[price_vars].dropna()
    if len(price_data) > 50:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(price_data)
        pca = PCA()
        pca.fit(scaled_data)
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        print("PCA Analysis on Price Variables:")
        print(f"Variables included: {price_vars}")
        print("Explained variance by component:")
        for i, var in enumerate(explained_var[:5]):
            print(f"  PC{i+1}: {var:.3f} ({cumulative_var[i]:.3f} cumulative)")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_var)+1), explained_var)
        plt.title('Explained Variance by Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Ratio')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if all(col in df.columns for col in ['wm_region', 'brand_type', 'base_unit_retail_amt']):
    region_groups = [g['base_unit_retail_amt'].dropna() for _, g in df.groupby('wm_region')]
    if len(region_groups) > 1 and all(len(g) > 0 for g in region_groups):
        f_stat, p_value = f_oneway(*region_groups)
        print(f"\nOne-way ANOVA - Region Effect on Price:")
        print(f"F-statistic: {f_stat:.3f}, P-value: {p_value:.3f}")

    brand_groups = [g['base_unit_retail_amt'].dropna() for _, g in df.groupby('brand_type')]
    if len(brand_groups) > 1 and all(len(g) > 0 for g in brand_groups):
        f_stat, p_value = f_oneway(*brand_groups)
        print(f"\nOne-way ANOVA - Brand Type Effect on Price:")
        print(f"F-statistic: {f_stat:.3f}, P-value: {p_value:.3f}")


# # 3.5 Advanced Visualization
print("\n📈 3.5 ADVANCED VISUALIZATION (STACKED)")
print("-" * 40)

if all(col in df.columns for col in ['wm_region', 'wm_week', 'base_unit_retail_amt']):
    # --- Use a professional and visually appealing seaborn style ---
    sns.set_theme(style="whitegrid", palette="viridis")

    regions = df['wm_region'].unique()[:6]
    n_regions = len(regions)
    
    # --- Configure for a single column layout to stack plots vertically ---
    n_cols = 1
    n_rows = n_regions
    
    # --- Adjust figsize for a taller layout and create subplots ---
    # A shared Y-axis is kept for direct price level comparison.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), constrained_layout=True, sharey=True)
    
    # --- Handle the case of a single plot where 'axes' is not an array ---
    if n_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # --- Calculate the overall average price for a contextual reference line ---
    overall_avg_price = df[df['wm_region'].isin(regions)]['base_unit_retail_amt'].mean()

    for i, region in enumerate(regions):
        region_data = df[df['wm_region'] == region]
        weekly_avg = region_data.groupby('wm_week')['base_unit_retail_amt'].mean()
        
        # Plot the region's price trend
        axes[i].plot(weekly_avg.index, weekly_avg.values, linewidth=2.5, label=f'{region} Avg. Price')
        
        # Add a reference line for the overall average price
        axes[i].axhline(overall_avg_price, color='red', linestyle='--', linewidth=1.5, label=f'Overall Avg. (${overall_avg_price:.2f})')

        # --- Enhanced Aesthetics and Labels for clarity ---
        axes[i].set_title(f'Price Trend - {region}', fontsize=14, weight='bold')
        axes[i].set_xlabel('Week Number', fontsize=12)
        axes[i].set_ylabel('Average Price ($)', fontsize=12)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].legend(fontsize=10)

    # --- Add a clear, descriptive main title ---
    # The 'y' parameter is adjusted to prevent overlap with the top plot
    fig.suptitle('Small Multiples: Regional Price Trends vs. Overall Average', fontsize=20, weight='bold')
    plt.show()

print("\n✅ LEVEL 3 ADVANCED ANALYSIS COMPLETED")







