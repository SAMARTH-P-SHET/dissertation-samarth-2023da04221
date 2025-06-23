# =================================================================================
# SCRIPT: store_pricing_analysis_spark.py
# AUTHOR: SAMARTH P SHET
# DESCRIPTION: This script performs an in-depth analysis of product pricing data
#              for execution on a GCP Dataproc cluster. It reads from a Hive table,
#              processes the data using PySpark, and stores all analysis outputs
#              (summaries and visualizations) in a Google Cloud Storage bucket.
# =================================================================================

import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, BooleanType, DateType
from google.cloud import storage

# =================================================================================
# --- CONFIGURATION ---
# =================================================================================
# Fill in your project-specific details here.

# HIVE settings
HIVE_DATABASE = "your_hive_database"
HIVE_TABLE = "store_pricing"

# GCS settings for output
GCS_BUCKET = "your-gcs-bucket-name"
GCS_OUTPUT_PREFIX = "analysis/store_pricing_eda/" # A folder-like prefix in your bucket

# =================================================================================
# --- HELPER FUNCTIONS ---
# =================================================================================

def save_text_to_gcs(bucket_name, destination_blob_name, content):
    """Uploads a string to the GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(content)
        print(f"✅ Successfully saved text output to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"❌ Failed to save text to GCS: {e}")


def save_plot_to_gcs(fig, bucket_name, destination_blob_name):
    """Saves a matplotlib figure to a GCS bucket."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(buf, content_type='image/png')
        
        plt.close(fig) # Close the figure to free up memory
        print(f"✅ Successfully saved plot to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"❌ Failed to save plot to GCS: {e}")

# =================================================================================
# --- SPARK INITIALIZATION ---
# =================================================================================

spark = SparkSession.builder \
    .appName("Store Pricing EDA") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext
print(f"Spark Session started. Version: {spark.version}")

# =================================================================================
# --- 1. DATA LOADING AND PREPARATION ---
# =================================================================================
print("\n--- 1. Data Loading and Preparation ---")

# Load the dataset from the specified Hive table.
table_name = f"{HIVE_DATABASE}.{HIVE_TABLE}"
print(f"Loading data from Hive table: {table_name}...")
df = spark.read.table(table_name)

# Define a list of columns that should contain numeric price-related data.
price_columns = [
    'live_price_1_amt', 'base_unit_price_amt', 'defined_reg_price_1_amt',
    'multi_save_price_amt', 'live_price_amt', 'defined_reg_price_amt',
    'upc_live_price_amt', 'upc_reg_price_amt', 'wm_wk_nbr'
]

# Convert columns to numeric types, handling errors by turning them into nulls.
for col_name in price_columns:
    if col_name in df.columns:
        df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

# Create new derived columns for analysis.
if 'live_price_amt' in df.columns and 'defined_reg_price_amt' in df.columns:
    df = df.withColumn('price_gap', F.col('defined_reg_price_amt') - F.col('live_price_amt'))
    df = df.withColumn('discount_pct', 
        F.when(F.col('defined_reg_price_amt') != 0, (F.col('price_gap') / F.col('defined_reg_price_amt')) * 100)
         .otherwise(0)
    )

if 'multi_save_ind' in df.columns:
    df = df.withColumn('on_promotion', F.col('multi_save_ind') == 1)

# --- Feature Engineering ---
print("Starting feature engineering...")
# 1. Add a 'competitor_price_amt' for price gap analysis (using Spark's random function)
df = df.withColumn('competitor_price_amt', F.round(F.col('live_price_amt') * (0.9 + F.rand() * 0.2), 2))
# 2. Recalculate price_gap with competitor price
df = df.withColumn('price_gap', F.col('live_price_amt') - F.col('competitor_price_amt'))
# 3. Add time-based features (assuming 'date' column is of DateType or TimestampType)
if 'date' in df.columns:
    df = df.withColumn("date", F.to_date(F.col("date"))) # Ensure it's a date type
    df = df.withColumn('month', F.month(F.col('date')))
    df = df.withColumn('day_of_week', F.date_format(F.col('date'), 'EEEE'))
# 4. Add price_cents for Benford's Law analysis
df = df.withColumn('price_cents', (F.col('live_price_amt') * 100 % 100).cast(IntegerType()))

df.cache() # Cache the dataframe for faster access in subsequent stages
print("Data loaded and prepared successfully.")

# =================================================================================
# --- LEVEL 1: EDA BASIC CHECKS ---
# =================================================================================
print("\n--- LEVEL 1: BASIC EDA ANALYSIS ---")
output_report = "LEVEL 1: BASIC EDA ANALYSIS\n" + "="*60 + "\n"

# --- 1.1 Data Structure Overview ---
print("\n📋 1.1 DATA STRUCTURE OVERVIEW")
rows = df.count()
cols = len(df.columns)
unique_products = df.select(F.countDistinct('item_nbr')).first()[0]
unique_stores = df.select(F.countDistinct('store_nbr')).first()[0]
unique_weeks = df.select(F.countDistinct('wm_wk_nbr')).first()[0]
unique_provinces = df.select(F.countDistinct('prov_cd')).first()[0]
week_range = df.select(F.min('wm_wk_nbr'), F.max('wm_wk_nbr')).first()

structure_summary = f"""
📋 1.1 DATA STRUCTURE OVERVIEW
----------------------------------------
Dataset Dimensions: {rows:,} rows × {cols} columns
Key Dimensions:
  Unique Products: {unique_products:,}
  Unique Stores: {unique_stores:,}
  Unique Weeks: {unique_weeks}
  Unique Provinces: {unique_provinces}
  Week Range: {week_range['min(wm_wk_nbr)']} to {week_range['max(wm_wk_nbr)']}
Data Types:
{df.dtypes}
"""
print(structure_summary)
output_report += structure_summary

# --- 1.2 Geographic Coverage Analysis ---
print("\n🗺️ 1.2 GEOGRAPHIC COVERAGE")
province_summary_df = df.groupBy('prov_cd').count().toPandas()
region_summary_df = df.groupBy('region_nm').count().toPandas()
stores_by_province_df = df.groupBy('prov_cd').agg(F.countDistinct('store_nbr')).toPandas()

geo_summary = f"""
🗺️ 1.2 GEOGRAPHIC COVERAGE
----------------------------------------
Province Distribution:
{province_summary_df.to_string(index=False)}

Region Distribution:
{region_summary_df.to_string(index=False)}

Stores by Province:
{stores_by_province_df.to_string(index=False)}
"""
print(geo_summary)
output_report += geo_summary

# --- 1.3 Basic Pricing Overview ---
print("\n💰 1.3 BASIC PRICING OVERVIEW")
price_cols = [col for col in df.columns if 'price' in col.lower() or 'amt' in col.lower()]
price_summary_df = df.select(price_cols).describe().toPandas()
promo_rate = df.select(F.mean(F.col('on_promotion').cast('double'))).first()[0] * 100
gap_stats_df = df.select('price_gap').describe().toPandas()

pricing_summary = f"""
💰 1.3 BASIC PRICING OVERVIEW
----------------------------------------
Price Variable Summary:
{price_summary_df.round(2).to_string(index=False)}

Promotional Activity:
  Products on promotion: {promo_rate:.1f}%

Price Gap Statistics (Live - Competitor):
{gap_stats_df.round(2).to_string(index=False)}
"""
print(pricing_summary)
output_report += pricing_summary

# --- 1.4 Data Quality Assessment ---
print("\n🛡️ 1.4 DATA QUALITY CHECKS")
missing_counts = df.select([F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]).first().asDict()
missing_percent = {k: (v / rows * 100) for k, v in missing_counts.items()}

quality_summary = "\n🛡️ 1.4 DATA QUALITY CHECKS\n" + "-" * 40 + "\n"
quality_summary += "Missing Data Summary (Top 10):\n"
missing_df = pd.DataFrame({
    'Missing_Count': pd.Series(missing_counts),
    'Missing_Percentage': pd.Series(missing_percent)
}).sort_values('Missing_Percentage', ascending=False)
quality_summary += missing_df[missing_df.Missing_Count > 0].head(10).to_string() + "\n"

quality_summary += "\nData Quality Issues:\n"
for col in price_cols:
    negative_count = df.filter(F.col(col) < 0).count()
    if negative_count > 0:
        quality_summary += f"  - Negative values in {col}: {negative_count}\n"
    zero_count = df.filter(F.col(col) == 0).count()
    if zero_count > 0:
        quality_summary += f"  - Zero values in {col}: {zero_count}\n"

duplicates = rows - df.distinct().count()
quality_summary += f"  - Duplicate rows: {duplicates}\n"
print(quality_summary)
output_report += quality_summary

# Save the combined text report to GCS
save_text_to_gcs(GCS_BUCKET, f"{GCS_OUTPUT_PREFIX}reports/level_1_eda_summary.txt", output_report)

# --- 1.5 Basic Visualizations ---
print("\n📈 1.5 BASIC VISUALIZATIONS")

# For plotting distributions, it's often best to sample the data if it's very large.
# Let's take a 10% sample for visualization to avoid overwhelming the driver node.
sample_df = df.sample(fraction=0.1, seed=42)

# Plot 1: Live Price Distribution
live_prices_pd = sample_df.select('live_price_amt').toPandas()['live_price_amt'].dropna()
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(live_prices_pd, bins=50, alpha=0.7, edgecolor='black')
ax1.set_title('Live Price Distribution (10% Sample)')
ax1.set_xlabel('Price ($)')
ax1.set_ylabel('Frequency')
save_plot_to_gcs(fig1, GCS_BUCKET, f"{GCS_OUTPUT_PREFIX}plots/live_price_distribution.png")

# Plot 2: Price Gap Distribution
price_gaps_pd = sample_df.select('price_gap').toPandas()['price_gap'].dropna()
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.hist(price_gaps_pd, bins=50, alpha=0.7, edgecolor='black', color='orange')
ax2.set_title('Price Gap Distribution (Live - Competitor) (10% Sample)')
ax2.set_xlabel('Price Gap ($)')
ax2.set_ylabel('Frequency')
save_plot_to_gcs(fig2, GCS_BUCKET, f"{GCS_OUTPUT_PREFIX}plots/price_gap_distribution.png")

# Plot 3: Records by Province (using full dataset as it's an aggregation)
province_counts_pd = df.groupBy('prov_cd').count().toPandas()
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.bar(province_counts_pd['prov_cd'], province_counts_pd['count'])
ax3.set_title('Total Records by Province')
ax3.set_xlabel('Province')
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)
save_plot_to_gcs(fig3, GCS_BUCKET, f"{GCS_OUTPUT_PREFIX}plots/records_by_province.png")

# =================================================================================
# --- FINALIZATION ---
# =================================================================================
df.unpersist() # Unpersist the DataFrame to free up memory
print("\n✅ LEVEL 1 BASIC ANALYSIS COMPLETED")
print("All outputs saved to GCS bucket.")


# Level 2: EDA Medium Analysis

# 2.1 Geographic Price Analysis
print("\n🗺️ 2.1 GEOGRAPHIC PRICE ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['prov_cd', 'live_price_amt']):
    # Provincial analysis
    provincial_stats_agg = {
        'live_price_amt': ['mean', 'median', 'std', 'count']
    }
    if 'price_gap' in df.columns:
        provincial_stats_agg['price_gap'] = 'mean'
    if 'on_promotion' in df.columns:
        provincial_stats_agg['on_promotion'] = 'mean'
        
    provincial_stats = df.groupby('prov_cd').agg(provincial_stats_agg).round(2)
    
    print("Provincial Price Analysis:")
    print(provincial_stats)
    
    # Statistical test for provincial differences
    provinces = df['prov_cd'].unique()
    if len(provinces) > 1:
        province_groups = [df[df['prov_cd'] == prov]['live_price_amt'].dropna() 
                         for prov in provinces]
        
        if all(len(group) > 0 for group in province_groups):
            f_stat, p_value = stats.f_oneway(*province_groups)
            print(f"\nANOVA Test for Provincial Price Differences:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  P-value: {p_value:.3f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
    # Box plot by province
    plt.figure(figsize=(8, 4))
    df.boxplot(column='live_price_amt', by='prov_cd', ax=plt.gca())
    plt.title('Price Distribution by Province')
    plt.suptitle('')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# 2.2 Temporal Price Analysis
print("\n⏰ 2.2 TEMPORAL PRICE ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['wm_wk_nbr', 'live_price_amt']):
    # Weekly aggregation
    weekly_stats_agg = {
        'live_price_amt': ['mean', 'std', 'count']
    }
    if 'price_gap' in df.columns:
        weekly_stats_agg['price_gap'] = 'mean'
    if 'on_promotion' in df.columns:
        weekly_stats_agg['on_promotion'] = 'mean'
        
    weekly_stats = df.groupby('wm_wk_nbr').agg(weekly_stats_agg).round(2)
    
    print("Weekly Trends (first 10 weeks):")
    print(weekly_stats.head(10))
    
    # Time series visualization
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # Time series visualization - vertical layout
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    
    # Average price over time
    weekly_price = df.groupby('wm_wk_nbr')['live_price_amt'].mean()
    axes[0].plot(weekly_price.index, weekly_price.values, marker='o', linewidth=2)
    axes[0].set_title('Average Price Over Time')
    axes[0].set_ylabel('Average Price ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Price volatility over time
    weekly_std = df.groupby('wm_wk_nbr')['live_price_amt'].std()
    axes[1].plot(weekly_std.index, weekly_std.values, marker='o', linewidth=2, color='orange')
    axes[1].set_title('Price Volatility Over Time')
    axes[1].set_ylabel('Price Std Dev')
    axes[1].grid(True, alpha=0.3)
    
    # Promotional activity over time
    if 'on_promotion' in df.columns:
        weekly_promo = df.groupby('wm_wk_nbr')['on_promotion'].mean() * 100
        axes[2].plot(weekly_promo.index, weekly_promo.values, marker='o', linewidth=2, color='green')
        axes[2].set_title('Promotional Activity Over Time')
        axes[2].set_ylabel('% on Promotion')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_visible(False)  # Hide if not used
    
    # Average discount over time
    if 'price_gap' in df.columns:
        weekly_gap = df.groupby('wm_wk_nbr')['price_gap'].mean()
        axes[3].plot(weekly_gap.index, weekly_gap.values, marker='o', linewidth=2, color='red')
        axes[3].set_title('Average Discount Over Time')
        axes[3].set_xlabel('Week')
        axes[3].set_ylabel('Avg Discount ($)')
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].set_visible(False)  # Hide if not used
    
    plt.tight_layout()
    plt.show()


# 2.3 Promotion Impact Analysis
print("\n🎉 2.3 PROMOTION IMPACT ANALYSIS")
print("-" * 40)

if 'on_promotion' in df.columns:
    # Promotion summary
    promo_counts = df['on_promotion'].value_counts()
    promo_perc = df['on_promotion'].value_counts(normalize=True) * 100
    
    print("Promotion Summary:")
    print(f"  On Promotion: {promo_counts.get(True, 0)} obs ({promo_perc.get(True, 0):.1f}%)")
    print(f"  Not on Promotion: {promo_counts.get(False, 0)} obs ({promo_perc.get(False, 0):.1f}%)")

    # --- PLOT 1: Promotion Status Pie Chart ---
    plt.figure(figsize=(8, 6))
    
    # Dynamically generate labels and colors to prevent ValueError
    labels = []
    colors = []
    for status in promo_counts.index:
        if status:
            labels.append('On Promotion')
            colors.append('#ff9999')
        else:
            labels.append('Not on Promotion')
            colors.append('#66b3ff')

    plt.pie(promo_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Promotion Status Distribution')
    plt.ylabel('') # Hides the 'on_promotion' label on the y-axis
    plt.show()

# Weekly promotion analysis
if 'wm_wk_nbr' in df.columns:
    weekly_promo = df.groupby('wm_wk_nbr')['on_promotion'].mean() * 100
    print(f"\nWeekly Promotion Rates:")
    print(f"  Highest promotion week: {weekly_promo.idxmax()} ({weekly_promo.max():.1f}%)")
    print(f"  Lowest promotion week: {weekly_promo.idxmin()} ({weekly_promo.min():.1f}%)")
    print(f"  Average weekly variation: {weekly_promo.std():.1f}%")

    # --- PLOT 2: Weekly Promotion Rate Line Plot ---
    plt.figure(figsize=(15, 4))
    sns.lineplot(x=weekly_promo.index, y=weekly_promo.values, marker='o')
    plt.title('Weekly Promotion Rate (%)')
    plt.xlabel('Week Number')
    plt.ylabel('Promotion Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# Discount analysis
if 'price_gap' in df.columns and 'on_promotion' in df.columns:
    promo_discounts = df[df['on_promotion']]['price_gap']
    non_promo_gaps = df[~df['on_promotion']]['price_gap']
    
    print(f"\nDiscount Analysis:")
    print(f"  Average discount when on promotion: ${promo_discounts.mean():.2f}")
    print(f"  Average gap when not on promotion: ${non_promo_gaps.mean():.2f}")
    
    # --- PLOT 3: Discount Distribution Box Plot ---
    plt.figure(figsize=(14, 4))
    sns.boxplot(x='on_promotion', y='price_gap', data=df)
    plt.title('Price Gap Distribution by Promotion Status')
    plt.xlabel('On Promotion')
    plt.ylabel('Price Gap (Discount)')
    plt.xticks([False, True], ['Not on Promotion', 'On Promotion'])
    plt.show()

    # Statistical test
    if len(promo_discounts) > 10 and len(non_promo_gaps) > 10:
        # Drop NaN values for the statistical test
        clean_promo_discounts = promo_discounts.dropna()
        clean_non_promo_gaps = non_promo_gaps.dropna()
        if len(clean_promo_discounts) > 1 and len(clean_non_promo_gaps) > 1:
            t_stat, p_value = stats.ttest_ind(clean_promo_discounts, clean_non_promo_gaps, equal_var=False) # Welch's t-test
            print(f"\n  Statistical difference in price gaps: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.3f})")


# 2.4 Store Performance Analysis
print("\n🏪 2.4 STORE PERFORMANCE ANALYSIS")
print("-" * 40)

num_top_stores_to_show = 10
num_ranking_stores_to_show = 5

# --- CONFIGURATION ---
item_column = 'item_nbr'

if 'store_nbr' in df.columns and 'live_price_amt' in df.columns and item_column in df.columns:
    # Store-level analysis based on unique item count
    store_stats_agg = {
        'live_price_amt': ['mean', 'std'],
        item_column: ['nunique']  # Count unique items per store
    }
    if 'price_gap' in df.columns:
        store_stats_agg['price_gap'] = 'mean'
    if 'on_promotion' in df.columns:
        store_stats_agg['on_promotion'] = 'mean'
        
    store_stats = df.groupby('store_nbr').agg(store_stats_agg).round(2)
    
    # Flatten column names for easier access
    store_stats.columns = ['_'.join(col).strip() for col in store_stats.columns]
    item_count_col = f'{item_column}_nunique'
    
    print(f"Store Performance Summary (top {num_top_stores_to_show} by unique item count):")
    top_stores = store_stats.nlargest(num_top_stores_to_show, item_count_col)
    #print(top_stores)
    
    # --- PLOT 4: Top Stores by Unique Item Count ---
    plt.figure(figsize=(15, 4))
    sns.barplot(x=item_count_col, y=top_stores.index.astype(str), data=top_stores, orient='h', palette='viridis')
    plt.title(f'Top {num_top_stores_to_show} Stores by Number of Unique Items')
    plt.xlabel('Number of Unique Items')
    plt.ylabel('Store Number')
    plt.show()

    # Store price ranking (remains the same)
    avg_prices = df.groupby('store_nbr')['live_price_amt'].mean().sort_values()
    
    print(f"\nStore Price Ranking:")
    
    # --- PLOT 5: Store Price Ranking ---
    if len(avg_prices) >= num_ranking_stores_to_show * 2:
        plot_data = pd.concat([avg_prices.head(num_ranking_stores_to_show), avg_prices.tail(num_ranking_stores_to_show)])
        colors = ['#66b3ff'] * num_ranking_stores_to_show + ['#ff9999'] * num_ranking_stores_to_show
        
        plt.figure(figsize=(12, 4))
        plot_data.plot(kind='barh', color=colors)
        
        plt.title(f'Store Price Ranking (Lowest {num_ranking_stores_to_show} vs. Highest {num_ranking_stores_to_show} Average Price)')
        plt.xlabel('Average Live Price (amt)')
        plt.ylabel('Store Number')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Create legend
        lowest_patch = mpatches.Patch(color='#66b3ff', label=f'Lowest {num_ranking_stores_to_show} Priced Stores')
        highest_patch = mpatches.Patch(color='#ff9999', label=f'Highest {num_ranking_stores_to_show} Priced Stores')
        plt.legend(handles=[lowest_patch, highest_patch])
        
        plt.show()
    else:
        print("\nSkipping price ranking plot as there are not enough stores to compare.")

    # Store consistency analysis (remains the same)
    store_volatility = df.groupby('store_nbr')['live_price_amt'].std().sort_values()
    print(f"\nStore Price Consistency (lowest volatility = most consistent, top {num_ranking_stores_to_show}):")
    print(store_volatility.head(num_ranking_stores_to_show).round(2))
else:
    print(f"Could not perform store analysis. Required columns ('store_nbr', 'live_price_amt', '{item_column}') not found in the DataFrame.")


# 2.5 Statistical Relationship Analysis
print("\n📊 2.5 STATISTICAL RELATIONSHIPS")
print("-" * 40)
            
# Regional price differences
if all(col in df.columns for col in ['region_nm', 'live_price_amt']):
    regions = df['region_nm'].unique()
    if len(regions) > 1:
        # --- PLOT 6: Regional Price Distribution Boxplot ---
        plt.figure(figsize=(15, 4))
        sns.boxplot(x='region_nm', y='live_price_amt', data=df, palette='Set2')
        plt.title('Price Distribution by Region')
        plt.xlabel('Region')
        plt.ylabel('Live Price (amt)')
        plt.xticks(rotation=45)
        plt.show()

        region_groups = [df[df['region_nm'] == region]['live_price_amt'].dropna() 
                       for region in regions]
        
        if all(len(group) > 0 for group in region_groups):
            f_stat, p_value = stats.f_oneway(*region_groups)
            print(f"\nRegional Price Differences (ANOVA):")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  P-value: {p_value:.3f}")
            print(f"  Significant regional differences: {'Yes' if p_value < 0.05 else 'No'}")

print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")


# Correlation analysis
if 'price_cols' in locals():
    print(f"✅ 'price_cols' is defined.")
    print(f"   Content: {price_cols}")
    print(f"   Number of columns: {len(price_cols)}")
else:
    print(f"❌ 'price_cols' is not defined. Please define it as a list of column names.")
# --- End of Debugging Code ---


if 'price_cols' in locals() and len(price_cols) > 1:
    price_data = df[price_cols].dropna()
    
    # --- Start of Debugging Code ---
    # Check how many rows are left after dropping missing values
    print(f"   Number of rows in price_data (after dropna): {len(price_data)}")
    # --- End of Debugging Code ---

    if len(price_data) > 50:
        print("✅ Conditions met. Generating plot...")
        correlation_matrix = price_data.corr()
        
        plt.figure(figsize=(15, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5)
        plt.title('Price Variables Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Strong correlations
        print("Strong correlations (|r| > 0.7):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    print(f"  {correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_val:.3f}")
    else:
        print("❌ Plotting skipped: Not enough data (50 or fewer rows) after removing missing values.")
else:
    print("❌ Plotting skipped: 'price_cols' is not defined or has fewer than 2 columns.")


# 2.5 Statistical Relationship Analysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("\n📊 2.5 STATISTICAL RELATIONSHIPS")
print("-" * 40)

# Define the price-related columns available in the current DataFrame
price_cols = ['live_price_amt', 'price_gap']

# Correlation analysis
if len(price_cols) > 1:
    price_data = df[price_cols].dropna()
    
    if len(price_data) > 1:
        correlation_matrix = price_data.corr()
        
        # --- FIXED CORRELATION PLOT ---
        plt.figure(figsize=(6, 4))  # Adjusted size
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            linewidths=0.5,
            cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.2}  # Horizontal colorbar
        )
        plt.title('Price Variables Correlation Matrix', pad=20)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        
        # Strong correlations
        print("Strong correlations (|r| > 0.7):")
        if not correlation_matrix.empty:
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        print(f"  {correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_val:.3f}")
    else:
        print("Not enough data to compute a correlation matrix for price variables.")
        
# Regional price differences
if all(col in df.columns for col in ['region_nm', 'live_price_amt']):
    regions = df['region_nm'].unique()
    if len(regions) > 1:
        # --- PLOT 6: Regional Price Distribution Boxplot ---
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='region_nm', y='live_price_amt', data=df, palette='Set2')
        plt.title('Price Distribution by Region')
        plt.xlabel('Region')
        plt.ylabel('Live Price (amt)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        region_groups = [df[df['region_nm'] == region]['live_price_amt'].dropna() 
                         for region in regions]
        
        if all(len(group) > 0 for group in region_groups):
            f_stat, p_value = stats.f_oneway(*region_groups)
            print(f"\nRegional Price Differences (ANOVA):")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  P-value: {p_value:.3f}")
            print(f"  Significant regional differences: {'Yes' if p_value < 0.05 else 'No'}")
else:
    print("\nSkipping regional price analysis: 'region_nm' column not found in DataFrame.")

print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")



# Level 3: EDA Analysis

# 3.1 Store Clustering Analysis
print("\n🎯 3.1 STORE CLUSTERING ANALYSIS")
print("-" * 40)

if 'store_nbr' in df.columns and len(price_cols) > 0:
    # Create store-level features
    agg_dict = {
        'live_price_amt': ['mean', 'std'],
    }
    if 'price_gap' in df.columns:
        agg_dict['price_gap'] = 'mean'
    if 'on_promotion' in df.columns:
        agg_dict['on_promotion'] = 'mean'
    
    store_features = df.groupby('store_nbr').agg(agg_dict).round(3)
    
    # Flatten column names
    store_features.columns = ['_'.join(col).strip() for col in store_features.columns]
    store_features = store_features.dropna()
    
    if len(store_features) > 10:
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(store_features)
        
        # K-means clustering
        n_clusters = min(5, len(store_features) // 3)  # Reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add clusters back to store features
        store_features['cluster'] = clusters
        store_features
        
        print(f"Store Clustering Results ({n_clusters} clusters):")
        cluster_summary = store_features.groupby('cluster').agg({
            c: 'mean' for c in store_features.columns if c != 'cluster'
        })
        cluster_summary['Num_Stores'] = store_features['cluster'].value_counts()
        print(cluster_summary.round(2))
        
        # PCA for visualization
        if scaled_features.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                                c=clusters, cmap='viridis', alpha=0.7, s=60)
            plt.colorbar(scatter, label='Cluster')
            plt.title('Store Clustering Visualization (PCA Space)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"PCA Variance Explained: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")


# 3.2 Advanced Temporal Pattern Analysis
print("\n⏰ 3.2 ADVANCED TEMPORAL PATTERNS")
print("-" * 40)

if all(col in df.columns for col in ['wm_wk_nbr', 'live_price_amt']):
    weekly_data = df.groupby('wm_wk_nbr')['live_price_amt'].mean()
    
    if len(weekly_data) >= 20:  # Need sufficient data for pattern analysis
        # Trend analysis
        weeks = np.arange(len(weekly_data))
        trend_coef = np.polyfit(weeks, weekly_data.values, 1)[0]
        print(f"Overall price trend: ${trend_coef:.4f} per week")
        
        # Seasonality detection (if full year available)
        if len(weekly_data) >= 52:
            # Simple seasonal decomposition
            weekly_df = weekly_data.reset_index()
            weekly_df['week_of_year'] = weekly_df['wm_wk_nbr'] % 52
            seasonal_pattern = weekly_df.groupby('week_of_year')['live_price_amt'].mean()
            
            # Seasonal strength
            seasonal_var = seasonal_pattern.var()
            total_var = weekly_data.var()
            seasonal_strength = seasonal_var / total_var
            
            print(f"Seasonal strength: {seasonal_strength:.3f} ({seasonal_strength*100:.1f}% of total variation)")
            
            # Plot seasonal pattern
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(weekly_data.index, weekly_data.values, linewidth=2)
            plt.title('Weekly Price Time Series')
            plt.xlabel('Week Number')
            plt.ylabel('Average Price ($)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(seasonal_pattern.index, seasonal_pattern.values, marker='o', linewidth=2)
            plt.title('Seasonal Pattern (52-week cycle)')
            plt.xlabel('Week of Year')
            plt.ylabel('Average Price ($)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        # Price volatility analysis
        price_changes = weekly_data.pct_change().dropna()
        
        # Find high volatility periods (changes > 2 std deviations)
        threshold = price_changes.std() * 2
        high_vol_weeks = price_changes[abs(price_changes) > threshold]
        
        if len(high_vol_weeks) > 0:
            print(f"\nHigh Volatility Periods:")
            print(f"  Number of volatile weeks: {len(high_vol_weeks)}")
            print(f"  Average change in volatile weeks: {high_vol_weeks.mean()*100:.1f}%")
            most_volatile_week = high_vol_weeks.abs().idxmax()
            print(f"  Most volatile week: {most_volatile_week} ({high_vol_weeks[most_volatile_week]*100:.1f}% change)")


# 3.3 Advanced Geographic Analysis
print("\n🗺️ 3.3 ADVANCED GEOGRAPHIC ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['prov_cd', 'store_nbr', 'live_price_amt']):
    # Price coordination within provinces
    coordination_analysis = {}
    
    for province in df['prov_cd'].unique():
        prov_data = df[df['prov_cd'] == province]
        
        if len(prov_data['store_nbr'].unique()) > 1:
            # Calculate price correlation between stores in same province
            store_pivot = prov_data.pivot_table(
                index='wm_wk_nbr', 
                columns='store_nbr', 
                values='live_price_amt'
            )
            
            if store_pivot.shape[1] > 1:
                correlations = store_pivot.corr()
                avg_correlation = correlations.values[np.triu_indices_from(correlations.values, k=1)].mean()
                coordination_analysis[province] = avg_correlation
                
    if coordination_analysis:
        print("Price Coordination Analysis (Average correlation between stores):")
        for prov, corr in sorted(coordination_analysis.items(), key=lambda item: item[1], reverse=True):
            print(f"  {prov}: {corr:.3f}")
            
    # Geographic price gradient analysis
    if 'region_nm' in df.columns:
        regional_prices = df.groupby('region_nm')['live_price_amt'].mean().sort_values()
        price_range = regional_prices.max() - regional_prices.min()
        
        print(f"\nRegional Price Analysis:")
        print(f"  Price range across regions: ${price_range:.2f}")
        print(f"  Lowest price region: {regional_prices.index[0]} (${regional_prices.iloc[0]:.2f})")
        print(f"  Highest price region: {regional_prices.index[-1]} (${regional_prices.iloc[-1]:.2f})")


# 3.4 Advanced Pricing Strategy Analysis
print("\n💡 3.4 PRICING STRATEGY ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['store_nbr', 'live_price_amt', 'on_promotion']):
    strategy_agg = {'live_price_amt': 'mean', 'on_promotion': 'mean'}
    if 'price_gap' in df.columns:
        strategy_agg['price_gap'] = 'mean'
        
    store_strategies = df.groupby('store_nbr').agg(strategy_agg).round(3)
    
    # Classify stores by strategy
    avg_price = store_strategies['live_price_amt'].median()
    avg_promo = store_strategies['on_promotion'].median()
    
    strategies = []
    for store_id in store_strategies.index:
        price_level = 'High' if store_strategies.loc[store_id, 'live_price_amt'] > avg_price else 'Low'
        promo_level = 'High' if store_strategies.loc[store_id, 'on_promotion'] > avg_promo else 'Low'
        
        if price_level == 'High' and promo_level == 'Low':
            strategy = 'Premium'
        elif price_level == 'Low' and promo_level == 'Low':
            strategy = 'EDLP'  # Everyday Low Price
        elif price_level == 'High' and promo_level == 'High':
            strategy = 'Hi-Lo'  # High-Low pricing
        else:
            strategy = 'Discount'
        strategies.append(strategy)
        
    store_strategies['strategy'] = strategies
    
    strategy_summary = store_strategies['strategy'].value_counts()
    print("Store Pricing Strategy Classification:")
    for strategy, count in strategy_summary.items():
        pct = count / len(store_strategies) * 100
        print(f"  {strategy}: {count} stores ({pct:.1f}%)")
        
    # Strategy performance analysis
    if 'price_gap' in df.columns:
        performance_agg = {'live_price_amt': 'mean', 'on_promotion': 'mean', 'price_gap': 'mean'}
        strategy_performance = store_strategies.groupby('strategy').agg(performance_agg).round(2)
        
        print("\nStrategy Performance Analysis:")
        print(strategy_performance)


# 3.5 Store Network Analysis
print("\n🔗 3.5 STORE NETWORK ANALYSIS")
print("-" * 40)

if all(col in df.columns for col in ['store_nbr', 'wm_wk_nbr', 'live_price_amt']):
    # Create store-week price matrix
    price_matrix = df.pivot_table(
        index='wm_wk_nbr',
        columns='store_nbr', 
        values='live_price_amt'
    )
    
    if price_matrix.shape[1] > 2:  # Need at least 3 stores
        # Calculate store similarity (correlation)
        store_correlations = price_matrix.corr()
        
        # Network statistics
        corr_values = store_correlations.values[np.triu_indices_from(store_correlations.values, k=1)]
        avg_correlation = np.nanmean(corr_values)
        max_correlation = np.nanmax(corr_values)
        min_correlation = np.nanmin(corr_values)
        
        print(f"Store Network Analysis:")
        print(f"  Average store correlation: {avg_correlation:.3f}")
        print(f"  Highest correlation: {max_correlation:.3f}")
        print(f"  Lowest correlation: {min_correlation:.3f}")
        
        # Find most and least connected stores
        store_avg_corr = store_correlations.mean()
        most_connected = store_avg_corr.idxmax()
        least_connected = store_avg_corr.idxmin()
        
        print(f"  Most connected store: {most_connected} (avg corr: {store_avg_corr[most_connected]:.3f})")
        print(f"  Least connected store: {least_connected} (avg corr: {store_avg_corr[least_connected]:.3f})")
        
        # Visualize correlation network
        if len(store_correlations) <= 20:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(store_correlations, dtype=bool))
            sns.heatmap(store_correlations, mask=mask, annot=False, cmap='RdBu_r', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Store Price Correlation Network')
            plt.tight_layout()
            plt.show()

print("\n✅ LEVEL 3 ADVANCED ANALYSIS COMPLETED")


# Analyze pricing consistency across stores for same products
print("\n" + "-"*60)
print("ADDITIONAL ANALYSIS: PRODUCT PRICING CONSISTENCY")
print("-" * 60)
if all(col in df.columns for col in ['store_nbr', 'live_price_amt', 'item_nbr']):
    # Calculate coefficient of variation for each product across stores
    product_consistency = df.groupby('item_nbr')['live_price_amt'].agg(['mean', 'std']).reset_index()
    product_consistency['cv'] = product_consistency['std'] / product_consistency['mean']
    product_consistency = product_consistency.dropna()
    
    print(f"Average coefficient of variation: {product_consistency['cv'].mean():.3f}")
    print(f"Most consistent products (lowest CV):")
    print(product_consistency.nsmallest(5, 'cv')[['item_nbr', 'cv']])
    print(f"Least consistent products (highest CV):")
    print(product_consistency.nlargest(5, 'cv')[['item_nbr', 'cv']])

# Analyze seasonal patterns in promotional activity
print("\n" + "-"*60)
print("ADDITIONAL ANALYSIS: SEASONAL PROMOTION PATTERNS")
print("-" * 60)
if all(col in df.columns for col in ['wm_wk_nbr', 'on_promotion']):
    # Assume 52-week cycle for seasonality
    df_temp = df.copy()
    df_temp['season_week'] = df_temp['wm_wk_nbr'] % 52
    
    seasonal_promo = df_temp.groupby('season_week')['on_promotion'].mean() * 100
    
    print(f"Peak promotion weeks (Top 5):")
    print(seasonal_promo.nlargest(5))
    print(f"\nLow promotion weeks (Bottom 5):")
    print(seasonal_promo.nsmallest(5))

print("\n\n🎉 SCRIPT EXECUTION COMPLETED!")
print("="*80)


# 5.1 - PRODUCT PRICE VOLATILITY
print("\n" + "="*60)
print("NEW ANALYSIS: 5.1 - PRODUCT PRICE VOLATILITY")
print("="*60)

if 'item_nbr' in df.columns and 'live_price_amt' in df.columns:
    print("Analyzing product price stability...")
    
    # Group by item and calculate mean, std dev, and count of prices
    product_volatility = df.groupby('item_nbr')['live_price_amt'].agg(['mean', 'std', 'count']).copy()
    
    # Calculate Coefficient of Variation (CV) = std / mean
    # CV is a standardized measure of dispersion, useful for comparing volatility
    product_volatility['volatility_cv'] = (product_volatility['std'] / product_volatility['mean']).fillna(0)
    
    # Filter out products with very few data points to ensure meaningful stats
    product_volatility = product_volatility[product_volatility['count'] > 5]
    
    if not product_volatility.empty:
        # --- Identify Most and Least Volatile Products ---
        most_volatile = product_volatility.nlargest(10, 'volatility_cv')
        least_volatile = product_volatility.nsmallest(10, 'volatility_cv')
        
        print("\n--- Top 10 Most Volatile Products (by Coefficient of Variation) ---")
        print(most_volatile[['mean', 'std', 'volatility_cv']].round(3))
        
        print("\n--- Top 10 Least Volatile (Most Stable) Products ---")
        print(least_volatile[['mean', 'std', 'volatility_cv']].round(3))
        
        # --- Visualization ---
        plt.figure(figsize=(14, 6))
        
        # Plotting most volatile
        plt.subplot(1, 2, 1)
        sns.barplot(x=most_volatile.index, y=most_volatile['volatility_cv'], palette='Reds_d', order=most_volatile.index)
        plt.title('Top 10 Most Volatile Products')
        plt.xlabel('Item Number')
        plt.ylabel('Price Volatility (CV)')
        plt.xticks(rotation=45)
        
        # Plotting least volatile
        plt.subplot(1, 2, 2)
        sns.barplot(x=least_volatile.index, y=least_volatile['volatility_cv'], palette='Greens_d', order=least_volatile.index)
        plt.title('Top 10 Most Stable Products')
        plt.xlabel('Item Number')
        plt.ylabel('Price Volatility (CV)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()


# 5.2 - CATEGORY-SPECIFIC ANALYSIS
print("\n" + "="*60)
print("NEW ANALYSIS: 5.2 - CATEGORY-SPECIFIC ANALYSIS")
print("="*60)

if 'item_nbr' in df.columns:
    # --- Synthetic Category Creation ---
    # In a real scenario, this mapping would come from a business data source.
    print("Creating synthetic product categories for demonstration...")
    categories = ["Electronics", "Groceries", "Apparel", "Home Goods", "Toys"]
    
    def assign_category(item_id):
        """Assigns a category based on the item number."""
        return categories[item_id % len(categories)]

    df['product_category'] = df['item_nbr'].apply(assign_category)
    
    print("Categories created. Sample:")
    print(df[['item_nbr', 'product_category']].head())

    # --- Perform Analysis per Category ---
    print("\n--- Analyzing metrics per product category ---")
    
    # Define aggregations
    category_agg = {
        'live_price_amt': 'mean',
        'on_promotion': 'mean', # This will give the proportion of items on sale
        'price_gap': 'mean',
        'item_nbr': 'nunique' # Count unique items in category
    }
    
    # Group by the new category column
    category_analysis = df.groupby('product_category').agg(category_agg).rename(columns={
        'live_price_amt': 'Avg_Price',
        'on_promotion': 'Promotion_Rate',
        'price_gap': 'Avg_Price_Gap',
        'item_nbr': 'Num_Unique_Items'
    }).sort_values(by='Avg_Price', ascending=False)
    
    # Convert promotion rate to percentage
    category_analysis['Promotion_Rate'] *= 100
    
    print("\nCategory Analysis Summary:")
    print(category_analysis.round(2))
    
    # --- Visualization of Category Analysis ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Category-Specific Analysis', fontsize=16)

    # Plot 1: Average Price per Category
    sns.barplot(x=category_analysis.index, y=category_analysis['Avg_Price'], ax=axes[0], palette='coolwarm')
    axes[0].set_title('Average Price')
    axes[0].set_ylabel('Avg. Price (CAD)')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Promotion Rate per Category
    sns.barplot(x=category_analysis.index, y=category_analysis['Promotion_Rate'], ax=axes[1], palette='viridis')
    axes[1].set_title('Promotion Rate')
    axes[1].set_ylabel('On Promotion (%)')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Average Price Gap per Category
    sns.barplot(x=category_analysis.index, y=category_analysis['Avg_Price_Gap'], ax=axes[2], palette='plasma')
    axes[2].set_title('Average Price Gap to Competitor')
    axes[2].set_ylabel('Avg. Price Difference (CAD)')
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\n\n✅ ALL ANALYSES, INCLUDING PRODUCT & CATEGORY DEEP DIVES, ARE COMPLETE!")