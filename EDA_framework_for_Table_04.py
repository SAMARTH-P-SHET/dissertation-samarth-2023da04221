# 🎯 Complete Three-Tier Framework
# Level 1: Basic EDA Checks
# 
# Data Overview: Structure, dimensions, data types
# Missing Data: Patterns and percentages
# Descriptive Stats: Mean, median, distributions
# Quality Checks: Duplicates, outliers, impossible values
# Simple Visualizations: Histograms, bar charts, box plots
# 
# Level 2: Medium-Level Analysis
# 
# Temporal Analysis: Weekly trends, seasonality, moving averages
# Competitive Analysis: Price gaps, win/loss rates, market positioning
# Segmentation: Regional, brand, product category analysis
# Statistical Relationships: Correlations, t-tests, ANOVA
# Distribution Analysis: Normality tests, group comparisons
# 
# Level 3: Advanced EDA
# 
# Advanced Temporal: Cycle detection, trend decomposition, lead-lag analysis
# Multi-dimensional: Three-way analysis, interaction effects, panel data
# Pattern Recognition: Price cycles, competitive responses, anomaly detection
# Advanced Statistics: PCA, variance decomposition, ANOVA
# Complex Visualization: Small multiples, parallel coordinates, heatmaps


# Three-Tier EDA Implementation for Retail Pricing Data
# Master's Level Analysis: Basic → Medium → Advanced

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, jarque_bera
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

class ThreeTierEDA:
    """
    Three-tier EDA implementation for retail pricing data
    Level 1: Basic Checks
    Level 2: Medium Analysis  
    Level 3: Advanced Exploration
    """
    
    def __init__(self, data_path):
        """Initialize with data loading"""
        self.df = self.load_data(data_path)
        self.prepare_data()
        
    def load_data(self, data_path):
        """Load the dataset"""
        # Adjust delimiter based on your file format
        df = pd.read_csv(data_path, delimiter='\t', low_memory=False)
        return df
    
    def prepare_data(self):
        """Basic data preparation"""
        # Convert numeric columns
        numeric_cols = ['base_unit_retail_amt', 'comp_reg', 'comp_live', 
                       'reg_price_gap', 'live_price_gap', 'ken_week']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert week to datetime if possible
        if 'ken_week' in self.df.columns:
            self.df['week_date'] = pd.to_datetime(self.df['ken_week'], errors='coerce')
            
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

    # ========================================
    # LEVEL 1: BASIC EDA CHECKS
    # ========================================
    
    def level_1_basic_checks(self):
        """Complete Level 1 Basic EDA Analysis"""
        print("="*60)
        print("LEVEL 1: BASIC EDA CHECKS")
        print("="*60)
        
        self.basic_data_overview()
        self.basic_missing_data_analysis()
        self.basic_descriptive_statistics()
        self.basic_data_quality_checks()
        self.basic_visualizations()
        
        print("\n✅ LEVEL 1 BASIC CHECKS COMPLETED")
        
    def basic_data_overview(self):
        """1.1 Data Overview & Structure"""
        print("\n📋 1.1 DATA OVERVIEW & STRUCTURE")
        print("-" * 40)
        
        print(f"Dataset Dimensions: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Numeric Columns: {len(self.numeric_cols)}")
        print(f"Categorical Columns: {len(self.categorical_cols)}")
        
        print("\nColumn Types:")
        print(self.df.dtypes.value_counts())
        
        print("\nFirst 3 rows:")
        print(self.df.head(3))
        
    def basic_missing_data_analysis(self):
        """1.2 Missing Data Assessment"""
        print("\n🔍 1.2 MISSING DATA ASSESSMENT")
        print("-" * 40)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Show top 15 columns with missing data
        print("Top columns with missing data:")
        print(missing_df[missing_df.Missing_Count > 0].head(15))
        
        # Missing data visualization
        if missing_df.Missing_Count.sum() > 0:
            plt.figure(figsize=(12, 6))
            top_missing = missing_df[missing_df.Missing_Count > 0].head(15)
            plt.bar(range(len(top_missing)), top_missing.Missing_Percentage)
            plt.title('Missing Data Percentage by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(range(len(top_missing)), top_missing.index, rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
    def basic_descriptive_statistics(self):
        """1.3 Basic Descriptive Statistics"""
        print("\n📊 1.3 BASIC DESCRIPTIVE STATISTICS")
        print("-" * 40)
        
        # Numeric variables
        if self.numeric_cols:
            print("NUMERIC VARIABLES SUMMARY:")
            numeric_summary = self.df[self.numeric_cols].describe()
            print(numeric_summary.round(2))
            
        # Categorical variables  
        print("\nCATEGORICAL VARIABLES SUMMARY:")
        key_categorical = ['ken_region', 'brand_type', 'banner_name', 'item_win']
        
        for col in key_categorical:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts().head())
                
    def basic_data_quality_checks(self):
        """1.4 Basic Data Quality Checks"""
        print("\n🛡️ 1.4 BASIC DATA QUALITY CHECKS")
        print("-" * 40)
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates:,}")
        
        # Negative prices (should not exist)
        price_cols = ['base_unit_retail_amt', 'comp_reg', 'comp_live']
        for col in price_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                print(f"Negative values in {col}: {negative_count}")
                
        # Extreme outliers (>3 standard deviations)
        print("\nExtreme outliers (>3σ):")
        for col in self.numeric_cols:
            if self.df[col].dtype in ['int64', 'float64']:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                outliers = ((self.df[col] - mean_val).abs() > 3 * std_val).sum()
                if outliers > 0:
                    print(f"{col}: {outliers} outliers")
                    
    def basic_visualizations(self):
        """1.5 Simple Visualizations"""
        print("\n📈 1.5 BASIC VISUALIZATIONS")
        print("-" * 40)
        
        # Price distributions
        price_cols = ['base_unit_retail_amt', 'comp_reg', 'comp_live']
        available_price_cols = [col for col in price_cols if col in self.df.columns]
        
        if available_price_cols:
            fig, axes = plt.subplots(1, len(available_price_cols), figsize=(15, 5))
            if len(available_price_cols) == 1:
                axes = [axes]
                
            for i, col in enumerate(available_price_cols):
                data = self.df[col].dropna()
                if len(data) > 0:
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel('Price')
                    axes[i].set_ylabel('Frequency')
                    
            plt.tight_layout()
            plt.show()
            
        # Categorical distributions
        cat_cols = ['ken_region', 'brand_type']
        available_cat_cols = [col for col in cat_cols if col in self.df.columns]
        
        if available_cat_cols:
            fig, axes = plt.subplots(1, len(available_cat_cols), figsize=(15, 5))
            if len(available_cat_cols) == 1:
                axes = [axes]
                
            for i, col in enumerate(available_cat_cols):
                value_counts = self.df[col].value_counts().head(10)
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                
            plt.tight_layout()
            plt.show()

    # ========================================
    # LEVEL 2: MEDIUM-LEVEL ANALYSIS
    # ========================================
    
    def level_2_medium_analysis(self):
        """Complete Level 2 Medium Analysis"""
        print("\n" + "="*60)
        print("LEVEL 2: MEDIUM-LEVEL ANALYSIS")
        print("="*60)
        
        self.medium_temporal_analysis()
        self.medium_competitive_analysis()
        self.medium_segmentation_analysis()
        self.medium_statistical_relationships()
        self.medium_distribution_analysis()
        
        print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")
        
    def medium_temporal_analysis(self):
        """2.1 Temporal Analysis"""
        print("\n📅 2.1 TEMPORAL ANALYSIS")
        print("-" * 40)
        
        if 'ken_week' in self.df.columns:
            # Weekly aggregation
            weekly_data = self.df.groupby('ken_week').agg({
                'base_unit_retail_amt': ['mean', 'count'],
                'comp_reg': 'mean',
                'reg_price_gap': 'mean'
            }).round(2)
            
            print("Weekly trends (first 10 weeks):")
            print(weekly_data.head(10))
            
            # Time series plot
            if 'base_unit_retail_amt' in self.df.columns:
                plt.figure(figsize=(15, 6))
                weekly_prices = self.df.groupby('ken_week')['base_unit_retail_amt'].mean()
                plt.plot(weekly_prices.index, weekly_prices.values, marker='o', linewidth=2)
                plt.title('Average Price Trend Over 52 Weeks')
                plt.xlabel('Week')
                plt.ylabel('Average Price')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
    def medium_competitive_analysis(self):
        """2.2 Competitive Analysis"""
        print("\n🏆 2.2 COMPETITIVE ANALYSIS")
        print("-" * 40)
        
        # Price comparison analysis
        if all(col in self.df.columns for col in ['base_unit_retail_amt', 'comp_reg']):
            # Correlation between Ken's and competitor prices
            corr_coef = self.df['base_unit_retail_amt'].corr(self.df['comp_reg'])
            print(f"Price correlation (Ken's vs Competitor): {corr_coef:.3f}")
            
            # Price gap analysis
            if 'reg_price_gap' in self.df.columns:
                gap_stats = self.df['reg_price_gap'].describe()
                print("\nPrice Gap Statistics:")
                print(gap_stats.round(2))
                
                # Win/Loss analysis
                if 'item_win' in self.df.columns:
                    win_rate = (self.df['item_win'] == 'Y').mean() * 100
                    loss_rate = (self.df['item_win'] == 'N').mean() * 100
                    print(f"\nWin Rate: {win_rate:.1f}%")
                    print(f"Loss Rate: {loss_rate:.1f}%")
                    
        # Competitor comparison by banner
        if 'banner_name' in self.df.columns and 'comp_reg' in self.df.columns:
            competitor_analysis = self.df.groupby('banner_name').agg({
                'comp_reg': ['mean', 'count', 'std'],
                'reg_price_gap': 'mean'
            }).round(2)
            
            print("\nCompetitor Analysis by Banner:")
            print(competitor_analysis)
            
    def medium_segmentation_analysis(self):
        """2.3 Segmentation Analysis"""
        print("\n🎯 2.3 SEGMENTATION ANALYSIS")
        print("-" * 40)
        
        # Regional analysis
        if 'ken_region' in self.df.columns and 'base_unit_retail_amt' in self.df.columns:
            regional_analysis = self.df.groupby('ken_region').agg({
                'base_unit_retail_amt': ['mean', 'median', 'count', 'std'],
                'reg_price_gap': 'mean'
            }).round(2)
            
            print("Regional Analysis:")
            print(regional_analysis)
            
            # Regional box plot
            plt.figure(figsize=(12, 6))
            self.df.boxplot(column='base_unit_retail_amt', by='ken_region', ax=plt.gca())
            plt.title('Price Distribution by Region')
            plt.suptitle('')  # Remove default title
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        # Brand type analysis
        if 'brand_type' in self.df.columns:
            brand_analysis = self.df.groupby('brand_type').agg({
                'base_unit_retail_amt': ['mean', 'count'],
                'reg_price_gap': 'mean'
            }).round(2)
            
            print("\nBrand Type Analysis:")
            print(brand_analysis)
            
    def medium_statistical_relationships(self):
        """2.4 Statistical Relationships"""
        print("\n📊 2.4 STATISTICAL RELATIONSHIPS")
        print("-" * 40)
        
        # Correlation analysis
        if len(self.numeric_cols) > 1:
            price_numeric_cols = [col for col in self.numeric_cols 
                                if any(keyword in col.lower() for keyword in ['price', 'gap', 'amt'])]
            
            if len(price_numeric_cols) > 1:
                correlation_matrix = self.df[price_numeric_cols].corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                           square=True, linewidths=0.5)
                plt.title('Price Variables Correlation Matrix')
                plt.tight_layout()
                plt.show()
                
                print("Strong correlations (|r| > 0.5):")
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            print(f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_val:.3f}")
                            
        # T-test for brand types
        if all(col in self.df.columns for col in ['brand_type', 'base_unit_retail_amt']):
            pl_prices = self.df[self.df['brand_type'] == 'PL']['base_unit_retail_amt'].dropna()
            nb_prices = self.df[self.df['brand_type'] == 'NB']['base_unit_retail_amt'].dropna()
            
            if len(pl_prices) > 10 and len(nb_prices) > 10:
                t_stat, p_value = stats.ttest_ind(pl_prices, nb_prices)
                print(f"\nT-test (PL vs NB prices):")
                print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
                print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                
    def medium_distribution_analysis(self):
        """2.5 Distribution Analysis"""
        print("\n📈 2.5 DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        # Distribution analysis for key variables
        key_vars = ['base_unit_retail_amt', 'reg_price_gap', 'comp_reg']
        
        for var in key_vars:
            if var in self.df.columns:
                data = self.df[var].dropna()
                if len(data) > 20:
                    # Normality test
                    _, p_value = normaltest(data)
                    skewness = stats.skew(data)
                    kurtosis = stats.kurtosis(data)
                    
                    print(f"\n{var} Distribution Analysis:")
                    print(f"  Skewness: {skewness:.3f}")
                    print(f"  Kurtosis: {kurtosis:.3f}")
                    print(f"  Normality test p-value: {p_value:.3f}")
                    print(f"  Normal distribution: {'No' if p_value < 0.05 else 'Possibly'}")

    # ========================================
    # LEVEL 3: ADVANCED EDA
    # ========================================
    
    def level_3_advanced_analysis(self):
        """Complete Level 3 Advanced Analysis"""
        print("\n" + "="*60)
        print("LEVEL 3: ADVANCED EDA")
        print("="*60)
        
        self.advanced_temporal_exploration()
        self.advanced_multidimensional_analysis()
        self.advanced_pattern_recognition()
        self.advanced_statistical_analysis()
        self.advanced_visualization()
        
        print("\n✅ LEVEL 3 ADVANCED ANALYSIS COMPLETED")
        
    def advanced_temporal_exploration(self):
        """3.1 Advanced Temporal Exploration"""
        print("\n⏰ 3.1 ADVANCED TEMPORAL EXPLORATION")
        print("-" * 40)
        
        if 'ken_week' in self.df.columns and 'base_unit_retail_amt' in self.df.columns:
            # Weekly time series
            weekly_series = self.df.groupby('ken_week')['base_unit_retail_amt'].mean()
            
            if len(weekly_series) >= 52:  # Full year of data
                # Simple trend analysis
                weeks = np.arange(len(weekly_series))
                trend_coef = np.polyfit(weeks, weekly_series.values, 1)[0]
                print(f"Weekly trend coefficient: {trend_coef:.4f}")
                
                # Seasonal pattern (assuming 52 weeks = 1 year)
                weekly_series_df = weekly_series.reset_index()
                weekly_series_df['week_of_year'] = weekly_series_df['ken_week'] % 52
                seasonal_pattern = weekly_series_df.groupby('week_of_year')['base_unit_retail_amt'].mean()
                
                # Plot seasonal pattern
                plt.figure(figsize=(15, 6))
                plt.subplot(1, 2, 1)
                plt.plot(weekly_series.index, weekly_series.values)
                plt.title('Weekly Price Time Series')
                plt.xlabel('Week')
                plt.ylabel('Average Price')
                
                plt.subplot(1, 2, 2)
                plt.plot(seasonal_pattern.index, seasonal_pattern.values, marker='o')
                plt.title('Seasonal Pattern (52-week cycle)')
                plt.xlabel('Week of Year')
                plt.ylabel('Average Price')
                
                plt.tight_layout()
                plt.show()
                
                # Coefficient of variation over time
                weekly_cv = self.df.groupby('ken_week')['base_unit_retail_amt'].apply(
                    lambda x: x.std() / x.mean() if x.mean() != 0 else 0
                )
                print(f"Average weekly coefficient of variation: {weekly_cv.mean():.3f}")
                
    def advanced_multidimensional_analysis(self):
        """3.2 Multi-Dimensional Analysis"""
        print("\n🔄 3.2 MULTI-DIMENSIONAL ANALYSIS")
        print("-" * 40)
        
        # Three-way analysis: Region × Brand × Performance
        if all(col in self.df.columns for col in ['ken_region', 'brand_type', 'base_unit_retail_amt']):
            three_way = self.df.groupby(['ken_region', 'brand_type']).agg({
                'base_unit_retail_amt': ['mean', 'count', 'std'],
                'reg_price_gap': 'mean',
                'item_win': lambda x: (x == 'Y').sum() if 'item_win' in self.df.columns else 0
            }).round(2)
            
            print("Three-way Analysis (Region × Brand Type):")
            print(three_way.head(10))
            
            # Interaction effect visualization
            if 'reg_price_gap' in self.df.columns:
                plt.figure(figsize=(12, 6))
                interaction_data = self.df.pivot_table(
                    values='reg_price_gap', 
                    index='ken_region', 
                    columns='brand_type', 
                    aggfunc='mean'
                )
                sns.heatmap(interaction_data, annot=True, cmap='RdBu_r', center=0)
                plt.title('Price Gap Interaction: Region × Brand Type')
                plt.tight_layout()
                plt.show()
                
        # Panel data structure analysis
        if all(col in self.df.columns for col in ['ken_store_id', 'ken_week', 'base_unit_retail_amt']):
            # Between vs within store variation
            store_means = self.df.groupby('ken_store_id')['base_unit_retail_amt'].mean()
            week_means = self.df.groupby('ken_week')['base_unit_retail_amt'].mean()
            
            between_store_var = store_means.var()
            within_store_var = self.df.groupby('ken_store_id')['base_unit_retail_amt'].var().mean()
            
            print(f"\nPanel Data Variation Analysis:")
            print(f"Between-store variance: {between_store_var:.3f}")
            print(f"Within-store variance: {within_store_var:.3f}")
            print(f"Variance ratio (between/within): {between_store_var/within_store_var:.3f}")
            
    def advanced_pattern_recognition(self):
        """3.3 Advanced Pattern Recognition"""
        print("\n🔍 3.3 ADVANCED PATTERN RECOGNITION")
        print("-" * 40)
        
        # Price cycle analysis
        if all(col in self.df.columns for col in ['ken_week', 'base_unit_retail_amt']):
            weekly_prices = self.df.groupby('ken_week')['base_unit_retail_amt'].mean()
            
            if len(weekly_prices) >= 12:  # Need sufficient data
                # Find peaks and troughs
                from scipy.signal import find_peaks
                
                peaks, _ = find_peaks(weekly_prices.values, distance=4)  # At least 4 weeks apart
                troughs, _ = find_peaks(-weekly_prices.values, distance=4)
                
                print(f"Price peaks detected: {len(peaks)}")
                print(f"Price troughs detected: {len(troughs)}")
                
                if len(peaks) > 1:
                    avg_peak_distance = np.mean(np.diff(peaks))
                    print(f"Average cycle length: {avg_peak_distance:.1f} weeks")
                    
                # Visualize cycles
                plt.figure(figsize=(15, 6))
                plt.plot(weekly_prices.index, weekly_prices.values, linewidth=2)
                if len(peaks) > 0:
                    plt.scatter(weekly_prices.index[peaks], weekly_prices.values[peaks], 
                              color='red', s=100, label='Peaks', zorder=5)
                if len(troughs) > 0:
                    plt.scatter(weekly_prices.index[troughs], weekly_prices.values[troughs], 
                              color='blue', s=100, label='Troughs', zorder=5)
                plt.title('Price Cycles Detection')
                plt.xlabel('Week')
                plt.ylabel('Average Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
        # Competitive response pattern analysis
        if all(col in self.df.columns for col in ['ken_week', 'base_unit_retail_amt', 'comp_reg']):
            # Calculate price changes
            weekly_ken = self.df.groupby('ken_week')['base_unit_retail_amt'].mean()
            weekly_comp = self.df.groupby('ken_week')['comp_reg'].mean()
            
            ken_changes = weekly_ken.diff()
            comp_changes = weekly_comp.diff()
            
            # Cross-correlation analysis (simplified)
            if len(ken_changes.dropna()) > 10 and len(comp_changes.dropna()) > 10:
                correlation = ken_changes.corr(comp_changes)
                print(f"\nPrice change correlation (Ken's vs Competitor): {correlation:.3f}")
                
                # Lead-lag analysis (simplified)
                lag_1_corr = ken_changes[:-1].corr(comp_changes[1:])  # Ken leads by 1 week
                lag_neg1_corr = ken_changes[1:].corr(comp_changes[:-1])  # Competitor leads by 1 week
                
                print(f"Ken leads (1 week): {lag_1_corr:.3f}")
                print(f"Competitor leads (1 week): {lag_neg1_corr:.3f}")
                
    def advanced_statistical_analysis(self):
        """3.4 Advanced Statistical Analysis"""
        print("\n📊 3.4 ADVANCED STATISTICAL ANALYSIS")
        print("-" * 40)
        
        # Principal Component Analysis on price variables
        price_vars = [col for col in self.numeric_cols 
                     if any(keyword in col.lower() for keyword in ['price', 'gap', 'amt'])]
        
        if len(price_vars) >= 3:
            price_data = self.df[price_vars].dropna()
            
            if len(price_data) > 50:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(price_data)
                
                # Apply PCA
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)
                
                # Explained variance
                explained_var = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var)
                
                print("PCA Analysis on Price Variables:")
                print(f"Variables included: {len(price_vars)}")
                print("Explained variance by component:")
                for i, var in enumerate(explained_var[:5]):
                    print(f"  PC{i+1}: {var:.3f} ({cumulative_var[i]:.3f} cumulative)")
                    
                # Plot explained variance
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(explained_var[:10])+1), explained_var[:10], 'bo-')
                plt.title('Explained Variance by Component')
                plt.xlabel('Principal Component')
                plt.ylabel('Explained Variance Ratio')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(cumulative_var[:10])+1), cumulative_var[:10], 'ro-')
                plt.title('Cumulative Explained Variance')
                plt.xlabel('Principal Component')
                plt.ylabel('Cumulative Variance Ratio')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
        # ANOVA analysis
        if all(col in self.df.columns for col in ['ken_region', 'brand_type', 'base_unit_retail_amt']):
            # Two-way ANOVA
            from scipy.stats import f_oneway
            
            # Test region effect
            region_groups = [group['base_unit_retail_amt'].dropna() 
                           for name, group in self.df.groupby('ken_region')]
            
            if len(region_groups) > 1 and all(len(group) > 0 for group in region_groups):
                f_stat, p_value = f_oneway(*region_groups)
                print(f"\nOne-way ANOVA - Region Effect:")
                print(f"F-statistic: {f_stat:.3f}, P-value: {p_value:.3f}")
                print(f"Significant region effect: {'Yes' if p_value < 0.05 else 'No'}")
                
            # Test brand type effect
            brand_groups = [group['base_unit_retail_amt'].dropna() 
                          for name, group in self.df.groupby('brand_type')]
            
            if len(brand_groups) > 1 and all(len(group) > 0 for group in brand_groups):
                f_stat, p_value = f_oneway(*brand_groups)
                print(f"\nOne-way ANOVA - Brand Type Effect:")
                print(f"F-statistic: {f_stat:.3f}, P-value: {p_value:.3f}")
                print(f"Significant brand effect: {'Yes' if p_value < 0.05 else 'No'}")
                
        # Variance decomposition analysis
        if 'base_unit_retail_amt' in self.df.columns:
            total_variance = self.df['base_unit_retail_amt'].var()
            
            # Variance by different groupings
            variance_components = {}
            
            for group_col in ['ken_region', 'brand_type', 'banner_name']:
                if group_col in self.df.columns:
                    group_means = self.df.groupby(group_col)['base_unit_retail_amt'].mean()
                    between_group_var = group_means.var()
                    variance_components[f'{group_col}_between'] = between_group_var / total_variance
                    
            print(f"\nVariance Decomposition (as % of total variance):")
            for component, ratio in variance_components.items():
                print(f"{component}: {ratio:.3f} ({ratio*100:.1f}%)")
                
    def advanced_visualization(self):
        """3.5 Advanced Visualization & Exploration"""
        print("\n📈 3.5 ADVANCED VISUALIZATION")
        print("-" * 40)
        
        # Small multiples analysis
        if all(col in self.df.columns for col in ['ken_region', 'ken_week', 'base_unit_retail_amt']):
            regions = self.df['ken_region'].unique()[:6]  # Limit to 6 regions for clarity
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, region in enumerate(regions):
                if i < 6:
                    region_data = self.df[self.df['ken_region'] == region]
                    weekly_avg = region_data.groupby('ken_week')['base_unit_retail_amt'].mean()
                    
                    axes[i].plot(weekly_avg.index, weekly_avg.values, linewidth=2)
                    axes[i].set_title(f'Price Trend - {region}')
                    axes[i].set_xlabel('Week')
                    axes[i].set_ylabel('Average Price')
                    axes[i].grid(True, alpha=0.3)
                    
            plt.suptitle('Small Multiples: Price Trends by Region', fontsize=16)
            plt.tight_layout()
            plt.show()
            
        # Parallel coordinates plot
        if len(self.numeric_cols) >= 4:
            numeric_subset = [col for col in self.numeric_cols 
                            if any(keyword in col.lower() for keyword in ['price', 'gap'])][:4]
            
            if len(numeric_subset) >= 3:
                # Sample data for visualization (to avoid overcrowding)
                sample_data = self.df[numeric_subset + ['ken_region']].dropna().sample(
                    min(500, len(self.df)), random_state=42
                )
                
                # Normalize the data for parallel coordinates
                normalized_data = sample_data[numeric_subset].copy()
                for col in numeric_subset:
                    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                         (normalized_data[col].max() - normalized_data[col].min())
                
                plt.figure(figsize=(15, 8))
                
                if 'ken_region' in sample_data.columns:
                    regions = sample_data['ken_region'].unique()[:5]  # Limit colors
                    colors = plt.cm.Set1(np.linspace(0, 1, len(regions)))
                    
                    for i, region in enumerate(regions):
                        region_mask = sample_data['ken_region'] == region
                        region_normalized = normalized_data[region_mask]
                        
                        for idx, row in region_normalized.iterrows():
                            plt.plot(range(len(numeric_subset)), row.values, 
                                   color=colors[i], alpha=0.3, linewidth=0.5)
                else:
                    for idx, row in normalized_data.iterrows():
                        plt.plot(range(len(numeric_subset)), row.values, 
                               color='blue', alpha=0.3, linewidth=0.5)
                
                plt.xticks(range(len(numeric_subset)), numeric_subset, rotation=45)
                plt.title('Parallel Coordinates Plot - Price Variables')
                plt.ylabel('Normalized Values')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EDA REPORT")
        print("="*80)
        
        # Data summary
        print(f"\n📊 DATASET SUMMARY")
        print(f"   Dimensions: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"   Time Period: {self.df['ken_week'].min() if 'ken_week' in self.df.columns else 'N/A'} to {self.df['ken_week'].max() if 'ken_week' in self.df.columns else 'N/A'}")
        print(f"   Completeness: {(1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))*100:.1f}%")
        
        # Key insights
        insights = []
        
        # Price insights
        if 'base_unit_retail_amt' in self.df.columns:
            avg_price = self.df['base_unit_retail_amt'].mean()
            price_range = self.df['base_unit_retail_amt'].max() - self.df['base_unit_retail_amt'].min()
            insights.append(f"Average Ken's price: ${avg_price:.2f} (range: ${price_range:.2f})")
            
        # Competitive insights
        if 'reg_price_gap' in self.df.columns:
            avg_gap = self.df['reg_price_gap'].mean()
            positive_gaps = (self.df['reg_price_gap'] > 0).sum()
            total_gaps = self.df['reg_price_gap'].notna().sum()
            insights.append(f"Average price gap: ${avg_gap:.2f}")
            insights.append(f"Ken's higher priced: {positive_gaps/total_gaps*100:.1f}% of comparisons")
            
        # Win rate
        if 'item_win' in self.df.columns:
            win_rate = (self.df['item_win'] == 'Y').mean() * 100
            insights.append(f"Overall win rate: {win_rate:.1f}%")
            
        # Regional insights
        if 'ken_region' in self.df.columns:
            num_regions = self.df['ken_region'].nunique()
            top_region = self.df['ken_region'].mode().iloc[0]
            insights.append(f"Geographic coverage: {num_regions} regions")
            insights.append(f"Most represented region: {top_region}")
            
        # Temporal insights
        if 'ken_week' in self.df.columns:
            num_weeks = self.df['ken_week'].nunique()
            insights.append(f"Time coverage: {num_weeks} weeks of data")
            
        print(f"\n🔍 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
            
        return {
            'dataset_shape': self.df.shape,
            'completeness': (1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))*100,
            'insights': insights,
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols)
        }
        
    def run_complete_three_tier_eda(self):
        """Execute the complete three-tier EDA pipeline"""
        print("🚀 STARTING THREE-TIER EDA ANALYSIS")
        print("="*80)
        
        # Execute all three levels
        self.level_1_basic_checks()
        self.level_2_medium_analysis()
        self.level_3_advanced_analysis()
        
        # Generate final report
        report = self.generate_comprehensive_report()
        
        print("\n🎉 THREE-TIER EDA COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return report

# Usage Example and Main Execution
def main():
    """Main execution function"""
    print("Three-Tier EDA Implementation for Retail Pricing Data")
    print("="*60)
    
    # Example usage
    """
    # Initialize the analysis
    eda = ThreeTierEDA('your_retail_data.csv')
    
    # Run complete analysis
    report = eda.run_complete_three_tier_eda()
    
    # Or run individual levels
    eda.level_1_basic_checks()
    eda.level_2_medium_analysis()
    eda.level_3_advanced_analysis()
    """
    
    print("\nFramework Components:")
    print("✓ Level 1: Basic Checks (Data quality, descriptive stats)")
    print("✓ Level 2: Medium Analysis (Temporal, competitive, segmentation)")
    print("✓ Level 3: Advanced Analysis (PCA, patterns, complex viz)")
    print("\nReady for implementation with your dataset!")

if __name__ == "__main__":
    main()

# Additional utility functions for specific analyses

def weekly_seasonality_analysis(df, price_col='base_unit_retail_amt', week_col='ken_week'):
    """Detailed weekly seasonality analysis"""
    if all(col in df.columns for col in [price_col, week_col]):
        weekly_data = df.groupby(week_col)[price_col].mean()
        
        # Calculate seasonal indices
        overall_mean = weekly_data.mean()
        seasonal_indices = weekly_data / overall_mean
        
        print("Seasonal Analysis:")
        print(f"Highest price week: {seasonal_indices.idxmax()} (index: {seasonal_indices.max():.3f})")
        print(f"Lowest price week: {seasonal_indices.idxmin()} (index: {seasonal_indices.min():.3f})")
        print(f"Seasonal volatility: {seasonal_indices.std():.3f}")
        
        return seasonal_indices

def competitive_response_timing(df, ken_price_col='base_unit_retail_amt', 
                               comp_price_col='comp_reg', week_col='ken_week'):
    """Analyze competitive response timing patterns"""
    if all(col in df.columns for col in [ken_price_col, comp_price_col, week_col]):
        # Calculate price changes
        weekly_ken = df.groupby(week_col)[ken_price_col].mean()
        weekly_comp = df.groupby(week_col)[comp_price_col].mean()
        
        ken_changes = weekly_ken.pct_change()
        comp_changes = weekly_comp.pct_change()
        
        # Analyze response patterns
        significant_ken_changes = ken_changes[abs(ken_changes) > ken_changes.std()]
        
        response_analysis = []
        for week in significant_ken_changes.index:
            if week + 1 in comp_changes.index:
                ken_change = ken_changes[week]
                comp_response = comp_changes[week + 1]
                response_analysis.append({
                    'week': week,
                    'ken_change': ken_change,
                    'comp_response': comp_response,
                    'same_direction': (ken_change * comp_response) > 0
                })
        
        if response_analysis:
            same_direction_pct = sum(1 for r in response_analysis if r['same_direction']) / len(response_analysis) * 100
            print(f"Competitive Response Analysis:")
            print(f"Same direction responses: {same_direction_pct:.1f}%")
            
        return response_analysis

print("\n🎯 Three-Tier EDA Framework Complete!")
print("This implementation provides:")
print("• Level 1: Essential data quality and basic exploration")
print("• Level 2: Business-focused analysis with statistical testing")  
print("• Level 3: Advanced pattern recognition and multivariate analysis")
print("• Comprehensive reporting and visualization")
print("• Master's level analytical depth without ML complexity")