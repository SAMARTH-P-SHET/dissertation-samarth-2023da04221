# EDA Implementation for New Retail Pricing Dataset
# Store-Level Pricing Analysis Across 52 Weeks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

class RetailPricingEDA_V2:
    """
    EDA Implementation for Store-Level Pricing Dataset
    Focus: Geographic, temporal, and promotional analysis
    """
    
    def __init__(self, data_path):
        """Initialize with data loading and preparation"""
        self.df = self.load_and_prepare_data(data_path)
        self.setup_analysis_variables()
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare the pricing dataset"""
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        # Convert numeric columns
        price_columns = [
            'live_price_1_amt', 'base_unit_price_amt', 'defined_reg_price_1_amt',
            'multi_save_price_amt', 'live_price_amt', 'defined_reg_price_amt',
            'upc_live_price_amt', 'upc_reg_price_amt', 'wm_wk_nbr'
        ]
        
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived variables
        if 'live_price_amt' in df.columns and 'defined_reg_price_amt' in df.columns:
            df['price_gap'] = df['defined_reg_price_amt'] - df['live_price_amt']
            df['discount_pct'] = (df['price_gap'] / df['defined_reg_price_amt'] * 100).fillna(0)
            
        # Convert multi_save_ind to boolean
        if 'multi_save_ind' in df.columns:
            df['on_promotion'] = df['multi_save_ind'] == 1
            
        return df
    
    def setup_analysis_variables(self):
        """Setup variables for analysis"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Key analysis columns
        self.price_cols = [col for col in self.numeric_cols if 'price' in col.lower() or 'amt' in col.lower()]
        self.geo_cols = ['prov_cd', 'region_nm', 'store_nbr']
        self.time_col = 'wm_wk_nbr'

    # ========================================
    # LEVEL 1: BASIC EDA CHECKS
    # ========================================
    
    def level_1_basic_analysis(self):
        """Complete Level 1 Basic Analysis"""
        print("="*60)
        print("LEVEL 1: BASIC EDA ANALYSIS")
        print("="*60)
        
        self.basic_data_overview()
        self.basic_geographic_analysis()
        self.basic_pricing_analysis()
        self.basic_quality_checks()
        self.basic_visualizations()
        
        print("\n✅ LEVEL 1 BASIC ANALYSIS COMPLETED")
        
    def basic_data_overview(self):
        """1.1 Data Structure Overview"""
        print("\n📋 1.1 DATA STRUCTURE OVERVIEW")
        print("-" * 40)
        
        print(f"Dataset Dimensions: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Unique counts
        unique_products = self.df['item_nbr'].nunique() if 'item_nbr' in self.df.columns else 0
        unique_stores = self.df['store_nbr'].nunique() if 'store_nbr' in self.df.columns else 0
        unique_weeks = self.df['wm_wk_nbr'].nunique() if 'wm_wk_nbr' in self.df.columns else 0
        unique_provinces = self.df['prov_cd'].nunique() if 'prov_cd' in self.df.columns else 0
        
        print(f"\nKey Dimensions:")
        print(f"  Unique Products: {unique_products:,}")
        print(f"  Unique Stores: {unique_stores:,}")
        print(f"  Unique Weeks: {unique_weeks}")
        print(f"  Unique Provinces: {unique_provinces}")
        
        # Time period coverage
        if 'wm_wk_nbr' in self.df.columns:
            week_range = f"{self.df['wm_wk_nbr'].min()} to {self.df['wm_wk_nbr'].max()}"
            print(f"  Week Range: {week_range}")
            
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())
        
    def basic_geographic_analysis(self):
        """1.2 Geographic Coverage Analysis"""
        print("\n🗺️ 1.2 GEOGRAPHIC COVERAGE")
        print("-" * 40)
        
        if 'prov_cd' in self.df.columns:
            province_summary = self.df['prov_cd'].value_counts()
            print("Province Distribution:")
            print(province_summary)
            
        if 'region_nm' in self.df.columns:
            region_summary = self.df['region_nm'].value_counts()
            print("\nRegion Distribution:")
            print(region_summary)
            
        if 'store_nbr' in self.df.columns and 'prov_cd' in self.df.columns:
            stores_by_province = self.df.groupby('prov_cd')['store_nbr'].nunique()
            print("\nStores by Province:")
            print(stores_by_province)
            
    def basic_pricing_analysis(self):
        """1.3 Basic Pricing Overview"""
        print("\n💰 1.3 BASIC PRICING OVERVIEW")
        print("-" * 40)
        
        # Price distributions
        if self.price_cols:
            print("Price Variable Summary:")
            price_summary = self.df[self.price_cols].describe()
            print(price_summary.round(2))
            
        # Promotional activity
        if 'on_promotion' in self.df.columns:
            promo_rate = self.df['on_promotion'].mean() * 100
            print(f"\nPromotional Activity:")
            print(f"  Products on promotion: {promo_rate:.1f}%")
            
        # Price gap analysis
        if 'price_gap' in self.df.columns:
            gap_stats = self.df['price_gap'].describe()
            print(f"\nPrice Gap Statistics (Regular - Live):")
            print(gap_stats.round(2))
            
    def basic_quality_checks(self):
        """1.4 Data Quality Assessment"""
        print("\n🛡️ 1.4 DATA QUALITY CHECKS")
        print("-" * 40)
        
        # Missing data
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        print("Missing Data Summary:")
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df[missing_df.Missing_Count > 0].head(10))
        
        # Impossible values
        print("\nData Quality Issues:")
        
        # Negative prices
        for col in self.price_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    print(f"  Negative values in {col}: {negative_count}")
                    
        # Zero prices
        for col in self.price_cols:
            if col in self.df.columns:
                zero_count = (self.df[col] == 0).sum()
                if zero_count > 0:
                    print(f"  Zero values in {col}: {zero_count}")
                    
        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"  Duplicate rows: {duplicates}")
        
    def basic_visualizations(self):
        """1.5 Basic Visualizations"""
        print("\n📈 1.5 BASIC VISUALIZATIONS")
        print("-" * 40)
        
        # Price distributions
        if 'live_price_amt' in self.df.columns:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            live_prices = self.df['live_price_amt'].dropna()
            plt.hist(live_prices, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Live Price Distribution')
            plt.xlabel('Price ($)')
            plt.ylabel('Frequency')
            
            if 'price_gap' in self.df.columns:
                plt.subplot(1, 3, 2)
                price_gaps = self.df['price_gap'].dropna()
                plt.hist(price_gaps, bins=50, alpha=0.7, edgecolor='black')
                plt.title('Price Gap Distribution')
                plt.xlabel('Price Gap ($)')
                plt.ylabel('Frequency')
                
            if 'prov_cd' in self.df.columns:
                plt.subplot(1, 3, 3)
                province_counts = self.df['prov_cd'].value_counts()
                plt.bar(province_counts.index, province_counts.values)
                plt.title('Records by Province')
                plt.xlabel('Province')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.show()

    # ========================================
    # LEVEL 2: MEDIUM ANALYSIS
    # ========================================
    
    def level_2_medium_analysis(self):
        """Complete Level 2 Medium Analysis"""
        print("\n" + "="*60)
        print("LEVEL 2: MEDIUM-LEVEL ANALYSIS")
        print("="*60)
        
        self.medium_geographic_analysis()
        self.medium_temporal_analysis()
        self.medium_promotional_analysis()
        self.medium_store_performance()
        self.medium_statistical_analysis()
        
        print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")
        
    def medium_geographic_analysis(self):
        """2.1 Geographic Price Analysis"""
        print("\n🗺️ 2.1 GEOGRAPHIC PRICE ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in ['prov_cd', 'live_price_amt']):
            # Provincial analysis
            provincial_stats = self.df.groupby('prov_cd').agg({
                'live_price_amt': ['mean', 'median', 'std', 'count'],
                'price_gap': 'mean' if 'price_gap' in self.df.columns else lambda x: None,
                'on_promotion': 'mean' if 'on_promotion' in self.df.columns else lambda x: None
            }).round(2)
            
            print("Provincial Price Analysis:")
            print(provincial_stats)
            
            # Statistical test for provincial differences
            provinces = self.df['prov_cd'].unique()
            if len(provinces) > 1:
                province_groups = [self.df[self.df['prov_cd'] == prov]['live_price_amt'].dropna() 
                                 for prov in provinces]
                
                if all(len(group) > 0 for group in province_groups):
                    f_stat, p_value = stats.f_oneway(*province_groups)
                    print(f"\nANOVA Test for Provincial Price Differences:")
                    print(f"  F-statistic: {f_stat:.3f}")
                    print(f"  P-value: {p_value:.3f}")
                    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
                    
            # Box plot by province
            plt.figure(figsize=(12, 6))
            self.df.boxplot(column='live_price_amt', by='prov_cd', ax=plt.gca())
            plt.title('Price Distribution by Province')
            plt.suptitle('')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
    def medium_temporal_analysis(self):
        """2.2 Temporal Price Analysis"""
        print("\n⏰ 2.2 TEMPORAL PRICE ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in ['wm_wk_nbr', 'live_price_amt']):
            # Weekly aggregation
            weekly_stats = self.df.groupby('wm_wk_nbr').agg({
                'live_price_amt': ['mean', 'std', 'count'],
                'price_gap': 'mean' if 'price_gap' in self.df.columns else lambda x: None,
                'on_promotion': 'mean' if 'on_promotion' in self.df.columns else lambda x: None
            }).round(2)
            
            print("Weekly Trends (first 10 weeks):")
            print(weekly_stats.head(10))
            
            # Time series visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Average price over time
            weekly_price = self.df.groupby('wm_wk_nbr')['live_price_amt'].mean()
            axes[0, 0].plot(weekly_price.index, weekly_price.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Average Price Over Time')
            axes[0, 0].set_xlabel('Week')
            axes[0, 0].set_ylabel('Average Price ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Price volatility over time
            weekly_std = self.df.groupby('wm_wk_nbr')['live_price_amt'].std()
            axes[0, 1].plot(weekly_std.index, weekly_std.values, marker='o', linewidth=2, color='orange')
            axes[0, 1].set_title('Price Volatility Over Time')
            axes[0, 1].set_xlabel('Week')
            axes[0, 1].set_ylabel('Price Standard Deviation')
            axes[0, 1].grid(True, alpha=0.3)
            
            if 'on_promotion' in self.df.columns:
                # Promotional activity over time
                weekly_promo = self.df.groupby('wm_wk_nbr')['on_promotion'].mean() * 100
                axes[1, 0].plot(weekly_promo.index, weekly_promo.values, marker='o', linewidth=2, color='green')
                axes[1, 0].set_title('Promotional Activity Over Time')
                axes[1, 0].set_xlabel('Week')
                axes[1, 0].set_ylabel('% Products on Promotion')
                axes[1, 0].grid(True, alpha=0.3)
                
            if 'price_gap' in self.df.columns:
                # Average discount over time
                weekly_gap = self.df.groupby('wm_wk_nbr')['price_gap'].mean()
                axes[1, 1].plot(weekly_gap.index, weekly_gap.values, marker='o', linewidth=2, color='red')
                axes[1, 1].set_title('Average Discount Over Time')
                axes[1, 1].set_xlabel('Week')
                axes[1, 1].set_ylabel('Average Discount ($)')
                axes[1, 1].grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.show()
            
    def medium_promotional_analysis(self):
        """2.3 Promotional Activity Analysis"""
        print("\n🎯 2.3 PROMOTIONAL ACTIVITY ANALYSIS")
        print("-" * 40)
        
        if 'on_promotion' in self.df.columns:
            # Overall promotional statistics
            total_promo_rate = self.df['on_promotion'].mean() * 100
            print(f"Overall promotion rate: {total_promo_rate:.1f}%")
            
            # Promotional activity by province
            if 'prov_cd' in self.df.columns:
                promo_by_province = self.df.groupby('prov_cd')['on_promotion'].mean() * 100
                print("\nPromotion Rate by Province:")
                print(promo_by_province.round(1))
                
            # Promotional activity over time
            if 'wm_wk_nbr' in self.df.columns:
                weekly_promo = self.df.groupby('wm_wk_nbr')['on_promotion'].mean() * 100
                
                print(f"\nPromotional Activity Trends:")
                print(f"  Highest promotion week: {weekly_promo.idxmax()} ({weekly_promo.max():.1f}%)")
                print(f"  Lowest promotion week: {weekly_promo.idxmin()} ({weekly_promo.min():.1f}%)")
                print(f"  Average weekly variation: {weekly_promo.std():.1f}%")
                
        # Discount analysis
        if 'price_gap' in self.df.columns and 'on_promotion' in self.df.columns:
            promo_discounts = self.df[self.df['on_promotion']]['price_gap']
            non_promo_gaps = self.df[~self.df['on_promotion']]['price_gap']
            
            print(f"\nDiscount Analysis:")
            print(f"  Average discount when on promotion: ${promo_discounts.mean():.2f}")
            print(f"  Average gap when not on promotion: ${non_promo_gaps.mean():.2f}")
            
            # Statistical test
            if len(promo_discounts) > 10 and len(non_promo_gaps) > 10:
                t_stat, p_value = stats.ttest_ind(promo_discounts.dropna(), non_promo_gaps.dropna())
                print(f"  Statistical difference: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.3f})")
                
    def medium_store_performance(self):
        """2.4 Store Performance Analysis"""
        print("\n🏪 2.4 STORE PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if 'store_nbr' in self.df.columns and 'live_price_amt' in self.df.columns:
            # Store-level analysis
            store_stats = self.df.groupby('store_nbr').agg({
                'live_price_amt': ['mean', 'std', 'count'],
                'price_gap': 'mean' if 'price_gap' in self.df.columns else lambda x: None,
                'on_promotion': 'mean' if 'on_promotion' in self.df.columns else lambda x: None
            }).round(2)
            
            # Flatten column names
            store_stats.columns = ['_'.join(col).strip() for col in store_stats.columns]
            
            print("Store Performance Summary (top 10 by record count):")
            top_stores = store_stats.nlargest(10, store_stats.columns[2])  # Sort by count
            print(top_stores)
            
            # Store price ranking
            avg_prices = self.df.groupby('store_nbr')['live_price_amt'].mean().sort_values()
            
            print(f"\nStore Price Ranking:")
            print(f"  Lowest priced stores (top 5):")
            print(avg_prices.head().round(2))
            print(f"  Highest priced stores (top 5):")
            print(avg_prices.tail().round(2))
            
            # Store consistency analysis
            store_volatility = self.df.groupby('store_nbr')['live_price_amt'].std().sort_values()
            print(f"\nStore Price Consistency (lowest volatility = most consistent):")
            print(store_volatility.head().round(2))
            
    def medium_statistical_analysis(self):
        """2.5 Statistical Relationship Analysis"""
        print("\n📊 2.5 STATISTICAL RELATIONSHIPS")
        print("-" * 40)
        
        # Correlation analysis
        if len(self.price_cols) > 1:
            price_data = self.df[self.price_cols].dropna()
            
            if len(price_data) > 50:
                correlation_matrix = price_data.corr()
                
                plt.figure(figsize=(10, 8))
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
                            
        # Regional price differences
        if all(col in self.df.columns for col in ['region_nm', 'live_price_amt']):
            regions = self.df['region_nm'].unique()
            if len(regions) > 1:
                region_groups = [self.df[self.df['region_nm'] == region]['live_price_amt'].dropna() 
                               for region in regions]
                
                if all(len(group) > 0 for group in region_groups):
                    f_stat, p_value = stats.f_oneway(*region_groups)
                    print(f"\nRegional Price Differences (ANOVA):")
                    print(f"  F-statistic: {f_stat:.3f}")
                    print(f"  P-value: {p_value:.3f}")
                    print(f"  Significant regional differences: {'Yes' if p_value < 0.05 else 'No'}")

    # ========================================
    # LEVEL 3: ADVANCED ANALYSIS
    # ========================================
    
    def level_3_advanced_analysis(self):
        """Complete Level 3 Advanced Analysis"""
        print("\n" + "="*60)
        print("LEVEL 3: ADVANCED ANALYSIS")
        print("="*60)
        
        self.advanced_store_clustering()
        self.advanced_temporal_patterns()
        self.advanced_geographic_analysis()
        self.advanced_pricing_strategy()
        self.advanced_network_analysis()
        
        print("\n✅ LEVEL 3 ADVANCED ANALYSIS COMPLETED")
        
    def advanced_store_clustering(self):
        """3.1 Advanced Store Clustering Analysis"""
        print("\n🎯 3.1 STORE CLUSTERING ANALYSIS")
        print("-" * 40)
        
        if 'store_nbr' in self.df.columns and len(self.price_cols) > 2:
            # Create store-level features
            store_features = self.df.groupby('store_nbr').agg({
                'live_price_amt': ['mean', 'std'],
                'price_gap': 'mean' if 'price_gap' in self.df.columns else lambda x: None,
                'on_promotion': 'mean' if 'on_promotion' in self.df.columns else lambda x: None,
                'wm_wk_nbr': 'count'  # Number of records
            }).round(3)
            
            # Flatten column names
            store_features.columns = ['_'.join(col).strip() for col in store_features.columns]
            store_features = store_features.dropna()
            
            if len(store_features) > 10:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                
                # Standardize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(store_features)
                
                # K-means clustering
                n_clusters = min(5, len(store_features) // 3)  # Reasonable number of clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_features)
                
                # Add clusters back to store features
                store_features['cluster'] = clusters
                
                print(f"Store Clustering Results ({n_clusters} clusters):")
                cluster_summary = store_features.groupby('cluster').agg({
                    store_features.columns[0]: 'mean',  # Average price
                    store_features.columns[1]: 'mean',  # Price volatility
                    'cluster': 'count'  # Cluster size
                }).round(2)
                cluster_summary.columns = ['Avg_Price', 'Price_Volatility', 'Num_Stores']
                print(cluster_summary)
                
                # PCA for visualization
                if len(store_features.columns) > 2:
                    pca = PCA(n_components=2)
                    pca_features = pca.fit_transform(scaled_features)
                    
                    plt.figure(figsize=(12, 8))
                    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                                        c=clusters, cmap='viridis', alpha=0.6, s=50)
                    plt.colorbar(scatter, label='Cluster')
                    plt.title('Store Clustering Visualization (PCA Space)')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"PCA Variance Explained: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
                    
    def advanced_temporal_patterns(self):
        """3.2 Advanced Temporal Pattern Analysis"""
        print("\n⏰ 3.2 ADVANCED TEMPORAL PATTERNS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in ['wm_wk_nbr', 'live_price_amt']):
            weekly_data = self.df.groupby('wm_wk_nbr')['live_price_amt'].mean()
            
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
                volatility_periods = []
                
                # Find high volatility periods (changes > 2 std deviations)
                threshold = price_changes.std() * 2
                high_vol_weeks = price_changes[abs(price_changes) > threshold]
                
                if len(high_vol_weeks) > 0:
                    print(f"\nHigh Volatility Periods:")
                    print(f"  Number of volatile weeks: {len(high_vol_weeks)}")
                    print(f"  Average change in volatile weeks: {high_vol_weeks.mean()*100:.1f}%")
                    print(f"  Most volatile week: {high_vol_weeks.abs().idxmax()} ({high_vol_weeks[high_vol_weeks.abs().idxmax()]*100:.1f}% change)")
                    
    def advanced_geographic_analysis(self):
        """3.3 Advanced Geographic Analysis"""
        print("\n🗺️ 3.3 ADVANCED GEOGRAPHIC ANALYSIS")
        print("-" * 40)
        
        # Multi-level geographic analysis
        if all(col in self.df.columns for col in ['prov_cd', 'store_nbr', 'live_price_amt']):
            # Price coordination within provinces
            coordination_analysis = {}
            
            for province in self.df['prov_cd'].unique():
                prov_data = self.df[self.df['prov_cd'] == province]
                
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
                for prov, corr in coordination_analysis.items():
                    print(f"  {prov}: {corr:.3f}")
                    
            # Geographic price gradient analysis
            if 'region_nm' in self.df.columns:
                regional_prices = self.df.groupby('region_nm')['live_price_amt'].mean().sort_values()
                price_range = regional_prices.max() - regional_prices.min()
                
                print(f"\nRegional Price Analysis:")
                print(f"  Price range across regions: ${price_range:.2f}")
                print(f"  Lowest price region: {regional_prices.index[0]} (${regional_prices.iloc[0]:.2f})")
                print(f"  Highest price region: {regional_prices.index[-1]} (${regional_prices.iloc[-1]:.2f})")
                
    def advanced_pricing_strategy(self):
        """3.4 Advanced Pricing Strategy Analysis"""
        print("\n💡 3.4 PRICING STRATEGY ANALYSIS")
        print("-" * 40)
        
        # Pricing strategy classification
        if all(col in self.df.columns for col in ['store_nbr', 'live_price_amt', 'on_promotion']):
            store_strategies = self.df.groupby('store_nbr').agg({
                'live_price_amt': 'mean',
                'on_promotion': 'mean',
                'price_gap': 'mean' if 'price_gap' in self.df.columns else lambda x: None
            }).round(3)
            
            # Classify stores by strategy
            avg_price = store_strategies['live_price_amt'].median()
            avg_promo = store_strategies['on_promotion'].median()
            
            strategies = []
            for store in store_strategies.index:
                price_level = 'High' if store_strategies.loc[store, 'live_price_amt'] > avg_price else 'Low'
                promo_level = 'High' if store_strategies.loc[store, 'on_promotion'] > avg_promo else 'Low'
                
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
            if 'price_gap' in self.df.columns:
                strategy_performance = store_strategies.groupby('strategy').agg({
                    'live_price_amt': 'mean',
                    'on_promotion': 'mean',
                    'price_gap': 'mean'
                }).round(2)
                
                print("\nStrategy Performance Analysis:")
                print(strategy_performance)
                
    def advanced_network_analysis(self):
        """3.5 Store Network Analysis"""
        print("\n🔗 3.5 STORE NETWORK ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in ['store_nbr', 'wm_wk_nbr', 'live_price_amt']):
            # Create store-week price matrix
            price_matrix = self.df.pivot_table(
                index='wm_wk_nbr',
                columns='store_nbr', 
                values='live_price_amt'
            )
            
            if price_matrix.shape[1] > 2:  # Need at least 3 stores
                # Calculate store similarity (correlation)
                store_correlations = price_matrix.corr()
                
                # Network statistics
                avg_correlation = store_correlations.values[np.triu_indices_from(store_correlations.values, k=1)].mean()
                max_correlation = store_correlations.values[np.triu_indices_from(store_correlations.values, k=1)].max()
                min_correlation = store_correlations.values[np.triu_indices_from(store_correlations.values, k=1)].min()
                
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
                
                # Visualize correlation network (simplified)
                if len(store_correlations) <= 20:  # Only for manageable number of stores
                    plt.figure(figsize=(12, 10))
                    
                    # Mask for upper triangle
                    mask = np.triu(np.ones_like(store_correlations, dtype=bool))
                    
                    sns.heatmap(store_correlations, mask=mask, annot=False, cmap='RdBu_r', 
                               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                    plt.title('Store Price Correlation Network')
                    plt.tight_layout()
                    plt.show()
                    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # Dataset summary
        print(f"\n📊 DATASET SUMMARY")
        print(f"   Records: {self.df.shape[0]:,} product-store-week combinations")
        print(f"   Products: {self.df['item_nbr'].nunique():,} unique items")
        print(f"   Stores: {self.df['store_nbr'].nunique()} locations")
        print(f"   Time Period: {self.df['wm_wk_nbr'].nunique()} weeks")
        print(f"   Geographic Coverage: {self.df['prov_cd'].nunique()} provinces")
        
        # Key insights
        insights = []
        
        # Price insights
        if 'live_price_amt' in self.df.columns:
            avg_price = self.df['live_price_amt'].mean()
            price_range = self.df['live_price_amt'].max() - self.df['live_price_amt'].min()
            insights.append(f"Average price across network: ${avg_price:.2f}")
            insights.append(f"Price range: ${price_range:.2f}")
            
        # Promotional insights
        if 'on_promotion' in self.df.columns:
            promo_rate = self.df['on_promotion'].mean() * 100
            insights.append(f"Promotional activity rate: {promo_rate:.1f}%")
            
        # Geographic insights
        if all(col in self.df.columns for col in ['prov_cd', 'live_price_amt']):
            provincial_prices = self.df.groupby('prov_cd')['live_price_amt'].mean()
            highest_price_prov = provincial_prices.idxmax()
            lowest_price_prov = provincial_prices.idxmin()
            insights.append(f"Highest price province: {highest_price_prov} (${provincial_prices[highest_price_prov]:.2f})")
            insights.append(f"Lowest price province: {lowest_price_prov} (${provincial_prices[lowest_price_prov]:.2f})")
            
        # Temporal insights
        if 'wm_wk_nbr' in self.df.columns:
            week_range = f"Week {self.df['wm_wk_nbr'].min()} to {self.df['wm_wk_nbr'].max()}"
            insights.append(f"Analysis period: {week_range}")
            
        print(f"\n🔍 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
            
        return {
            'dataset_shape': self.df.shape,
            'unique_products': self.df['item_nbr'].nunique(),
            'unique_stores': self.df['store_nbr'].nunique(),
            'unique_weeks': self.df['wm_wk_nbr'].nunique(),
            'insights': insights
        }
        
    def run_complete_analysis(self):
        """Execute complete three-tier analysis"""
        print("🚀 STARTING COMPREHENSIVE RETAIL PRICING ANALYSIS")
        print("="*80)
        
        # Run all three levels
        self.level_1_basic_analysis()
        self.level_2_medium_analysis()
        self.level_3_advanced_analysis()
        
        # Generate final report
        report = self.generate_comprehensive_report()
        
        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*80)
        
        return report

# Usage example and utility functions
def main():
    """Main execution function"""
    print("Retail Pricing EDA Framework - Store Level Analysis")
    print("="*60)
    
    print("Features included:")
    print("✓ Geographic pricing analysis across provinces/regions")
    print("✓ Temporal pricing trends over 52 weeks")
    print("✓ Promotional activity analysis")
    print("✓ Store performance benchmarking")
    print("✓ Advanced store clustering")
    print("✓ Price coordination analysis")
    print("✓ Pricing strategy classification")
    print("✓ Store network analysis")
    
    print("\nUsage:")
    print("eda = RetailPricingEDA_V2('your_data.csv')")
    print("report = eda.run_complete_analysis()")

def analyze_pricing_consistency(df, store_col='store_nbr', price_col='live_price_amt', item_col='item_nbr'):
    """Analyze pricing consistency across stores for same products"""
    if all(col in df.columns for col in [store_col, price_col, item_col]):
        # Calculate coefficient of variation for each product across stores
        product_consistency = df.groupby(item_col)[price_col].agg(['mean', 'std']).reset_index()
        product_consistency['cv'] = product_consistency['std'] / product_consistency['mean']
        product_consistency = product_consistency.dropna()
        
        print("Product Pricing Consistency Analysis:")
        print(f"Average coefficient of variation: {product_consistency['cv'].mean():.3f}")
        print(f"Most consistent products (lowest CV):")
        print(product_consistency.nsmallest(5, 'cv')[['item_nbr', 'cv']])
        print(f"Least consistent products (highest CV):")
        print(product_consistency.nlargest(5, 'cv')[['item_nbr', 'cv']])
        
        return product_consistency

def seasonal_promotion_analysis(df, week_col='wm_wk_nbr', promo_col='on_promotion'):
    """Analyze seasonal patterns in promotional activity"""
    if all(col in df.columns for col in [week_col, promo_col]):
        # Assume 52-week cycle for seasonality
        df_temp = df.copy()
        df_temp['season_week'] = df_temp[week_col] % 52
        
        seasonal_promo = df_temp.groupby('season_week')[promo_col].mean() * 100
        
        print("Seasonal Promotion Analysis:")
        print(f"Peak promotion weeks:")
        print(seasonal_promo.nlargest(5))
        print(f"Low promotion weeks:")
        print(seasonal_promo.nsmallest(5))
        
        return seasonal_promo

if __name__ == "__main__":
    main()

print("\n🎯 EDA Framework Ready for New Dataset!")
print("This implementation provides:")
print("• Store-level pricing analysis")
print("• Geographic pricing patterns")
print("• Promotional activity insights")
print("• Temporal trend analysis")
print("• Advanced store clustering")
print("• Network relationship analysis")
print("• Comprehensive business reporting")
