# Product Catalog EDA Implementation
# Comprehensive Analysis for Product Portfolio & Vendor Management

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

class ProductCatalogEDA:
    """
    Comprehensive EDA for Product Catalog/Master Data
    Focus: Product portfolio, vendor relationships, brand analysis, category management
    """
    
    def __init__(self, data_path):
        """Initialize with data loading and preparation"""
        self.df = self.load_and_prepare_data(data_path)
        self.setup_analysis_variables()
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare the product catalog dataset"""
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        # Convert numeric columns
        numeric_columns = [
            'item_nbr', 'old_nbr', 'sbu_nbr', 'division_nbr', 'dept_nbr',
            'dept_category_group_nbr', 'dept_category_nbr', 'dept_subcatg_nbr',
            'fineline_nbr', 'brand_id', 'vendor_id', 'buyer_id', 'director_id',
            'cost', 'base_unit_retail_amt', 'sell_qty', 'brand_family_id'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived metrics
        self.create_derived_metrics(df)
        
        return df
    
    def create_derived_metrics(self, df):
        """Create business-relevant derived metrics"""
        # Pricing metrics
        if all(col in df.columns for col in ['cost', 'base_unit_retail_amt']):
            df['margin_dollars'] = df['base_unit_retail_amt'] - df['cost']
            df['margin_percent'] = (df['margin_dollars'] / df['base_unit_retail_amt'] * 100).fillna(0)
            
        # Price segments
        if 'base_unit_retail_amt' in df.columns:
            df['price_segment'] = pd.cut(
                df['base_unit_retail_amt'].fillna(0),
                bins=[0, 5, 15, 50, float('inf')],
                labels=['Value', 'Mid-tier', 'Premium', 'Luxury']
            )
            
        # Product status indicators
        if 'item_status_code' in df.columns:
            df['is_active'] = df['item_status_code'] == 'A'
            
        # Brand type indicators
        if 'brand_type' in df.columns:
            df['is_national_brand'] = df['brand_type'] == 'NB'
            df['is_private_label'] = df['brand_type'] == 'PL'
            
        # Category hierarchy levels
        hierarchy_cols = ['sbu_desc', 'division_desc', 'dept_desc', 
                         'dept_category_desc', 'dept_subcatg_desc']
        for col in hierarchy_cols:
            if col in df.columns:
                df[f'{col}_clean'] = df[col].fillna('Unknown').astype(str)
                
    def setup_analysis_variables(self):
        """Setup key variables for analysis"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Key business dimensions
        self.hierarchy_cols = ['sbu_desc', 'division_desc', 'dept_desc', 
                              'dept_category_desc', 'dept_subcatg_desc', 'fineline_desc']
        self.vendor_cols = ['vendor_name', 'vendor_id']
        self.brand_cols = ['brand_name', 'brand_family_name', 'brand_type']
        self.pricing_cols = ['cost', 'base_unit_retail_amt', 'margin_dollars', 'margin_percent']
        self.operational_cols = ['buyer_name', 'item_status_code', 'country_name']

    # ========================================
    # LEVEL 1: BASIC EDA CHECKS
    # ========================================
    
    def level_1_basic_analysis(self):
        """Complete Level 1: Basic Product Catalog Analysis"""
        print("="*60)
        print("LEVEL 1: BASIC PRODUCT CATALOG ANALYSIS")
        print("="*60)
        
        self.basic_portfolio_overview()
        self.basic_category_structure()
        self.basic_vendor_brand_landscape()
        self.basic_pricing_overview()
        self.basic_quality_assessment()
        
        print("\n✅ LEVEL 1 BASIC ANALYSIS COMPLETED")
        
    def basic_portfolio_overview(self):
        """1.1 Product Portfolio Overview"""
        print("\n📦 1.1 PRODUCT PORTFOLIO OVERVIEW")
        print("-" * 40)
        
        # Dataset dimensions
        total_products = len(self.df)
        unique_upcs = self.df['upc'].nunique() if 'upc' in self.df.columns else 0
        
        print(f"Portfolio Size:")
        print(f"  Total Products (SKUs): {total_products:,}")
        print(f"  Unique UPCs: {unique_upcs:,}")
        
        # Product status distribution
        if 'item_status_code' in self.df.columns:
            status_dist = self.df['item_status_code'].value_counts()
            print(f"\nProduct Status Distribution:")
            for status, count in status_dist.items():
                percentage = count / total_products * 100
                print(f"  {status}: {count:,} ({percentage:.1f}%)")
                
        # Active vs inactive
        if 'is_active' in self.df.columns:
            active_count = self.df['is_active'].sum()
            inactive_count = total_products - active_count
            print(f"\nActive vs Inactive:")
            print(f"  Active Products: {active_count:,} ({active_count/total_products*100:.1f}%)")
            print(f"  Inactive Products: {inactive_count:,} ({inactive_count/total_products*100:.1f}%)")
            
        # Geographic sourcing
        if 'country_name' in self.df.columns:
            country_dist = self.df['country_name'].value_counts().head()
            print(f"\nTop Source Countries:")
            print(country_dist)
            
    def basic_category_structure(self):
        """1.2 Category Structure Analysis"""
        print("\n🏗️ 1.2 CATEGORY STRUCTURE ANALYSIS")
        print("-" * 40)
        
        # Hierarchy depth analysis
        for i, col in enumerate(self.hierarchy_cols):
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                level_name = col.replace('_desc', '').replace('_', ' ').title()
                print(f"  Level {i+1} - {level_name}: {unique_count} categories")
                
        # Department distribution
        if 'dept_desc' in self.df.columns:
            dept_dist = self.df['dept_desc'].value_counts().head(10)
            print(f"\nTop 10 Departments by Product Count:")
            print(dept_dist)
            
        # Category concentration
        if 'dept_category_desc' in self.df.columns:
            category_dist = self.df['dept_category_desc'].value_counts()
            top_5_categories = category_dist.head(5)
            top_5_share = top_5_categories.sum() / total_products * 100
            
            print(f"\nCategory Concentration:")
            print(f"  Top 5 categories contain {top_5_share:.1f}% of products")
            print(top_5_categories)
            
    def basic_vendor_brand_landscape(self):
        """1.3 Vendor & Brand Landscape"""
        print("\n🏭 1.3 VENDOR & BRAND LANDSCAPE")
        print("-" * 40)
        
        # Vendor analysis
        if 'vendor_name' in self.df.columns:
            unique_vendors = self.df['vendor_name'].nunique()
            top_vendors = self.df['vendor_name'].value_counts().head(10)
            
            print(f"Vendor Portfolio:")
            print(f"  Total Unique Vendors: {unique_vendors:,}")
            print(f"\nTop 10 Vendors by Product Count:")
            print(top_vendors)
            
        # Brand analysis
        if 'brand_name' in self.df.columns:
            unique_brands = self.df['brand_name'].nunique()
            top_brands = self.df['brand_name'].value_counts().head(10)
            
            print(f"\nBrand Portfolio:")
            print(f"  Total Unique Brands: {unique_brands:,}")
            print(f"\nTop 10 Brands by Product Count:")
            print(top_brands)
            
        # Brand type distribution
        if 'brand_type' in self.df.columns:
            brand_type_dist = self.df['brand_type'].value_counts()
            print(f"\nBrand Type Distribution:")
            for brand_type, count in brand_type_dist.items():
                percentage = count / len(self.df) * 100
                print(f"  {brand_type}: {count:,} ({percentage:.1f}%)")
                
    def basic_pricing_overview(self):
        """1.4 Basic Pricing Overview"""
        print("\n💰 1.4 PRICING OVERVIEW")
        print("-" * 40)
        
        # Price distribution
        if 'base_unit_retail_amt' in self.df.columns:
            price_summary = self.df['base_unit_retail_amt'].describe()
            print("Retail Price Distribution:")
            print(price_summary.round(2))
            
        # Cost distribution
        if 'cost' in self.df.columns:
            cost_summary = self.df['cost'].describe()
            print(f"\nCost Distribution:")
            print(cost_summary.round(2))
            
        # Margin analysis
        if 'margin_percent' in self.df.columns:
            margin_summary = self.df['margin_percent'].describe()
            print(f"\nMargin Percentage Distribution:")
            print(margin_summary.round(2))
            
        # Price segments
        if 'price_segment' in self.df.columns:
            price_seg_dist = self.df['price_segment'].value_counts()
            print(f"\nPrice Segment Distribution:")
            print(price_seg_dist)
            
    def basic_quality_assessment(self):
        """1.5 Data Quality Assessment"""
        print("\n🛡️ 1.5 DATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        # Missing data analysis
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        print("Missing Data Summary (Top 15):")
        missing_summary = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_summary.head(15))
        
        # Critical field completeness
        critical_fields = ['item_description', 'dept_desc', 'vendor_name', 
                          'brand_name', 'base_unit_retail_amt', 'cost']
        
        print(f"\nCritical Field Completeness:")
        for field in critical_fields:
            if field in self.df.columns:
                completeness = (1 - self.df[field].isnull().mean()) * 100
                print(f"  {field}: {completeness:.1f}% complete")
                
        # Duplicates analysis
        if 'item_nbr' in self.df.columns:
            duplicates = self.df['item_nbr'].duplicated().sum()
            print(f"\nDuplicate Analysis:")
            print(f"  Duplicate item numbers: {duplicates}")
            
        # Data consistency checks
        print(f"\nData Consistency Checks:")
        
        # Negative prices/costs
        if 'base_unit_retail_amt' in self.df.columns:
            negative_prices = (self.df['base_unit_retail_amt'] < 0).sum()
            print(f"  Negative retail prices: {negative_prices}")
            
        if 'cost' in self.df.columns:
            negative_costs = (self.df['cost'] < 0).sum()
            print(f"  Negative costs: {negative_costs}")
            
        # Invalid margins
        if 'margin_percent' in self.df.columns:
            extreme_margins = ((self.df['margin_percent'] < -100) | (self.df['margin_percent'] > 100)).sum()
            print(f"  Extreme margins (>100% or <-100%): {extreme_margins}")

    # ========================================
    # LEVEL 2: MEDIUM ANALYSIS
    # ========================================
    
    def level_2_medium_analysis(self):
        """Complete Level 2: Medium Analysis"""
        print("\n" + "="*60)
        print("LEVEL 2: MEDIUM-LEVEL ANALYSIS")
        print("="*60)
        
        self.medium_category_management()
        self.medium_vendor_analysis()
        self.medium_brand_analysis()
        self.medium_pricing_analysis()
        self.medium_operational_analysis()
        
        print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")
        
		
	def medium_category_management(self):
		"""2.1 Category Management Analysis"""
		print("\n📊 2.1 CATEGORY MANAGEMENT ANALYSIS")
		print("-" * 40)
		
		if 'dept_desc' in self.df.columns:
			# Department analysis
			dept_analysis = self.df.groupby('dept_desc').agg({
				'item_nbr': 'count',
				'base_unit_retail_amt': ['mean', 'median', 'std'] if 'base_unit_retail_amt' in self.df.columns else lambda x: None,
				'margin_percent': 'mean' if 'margin_percent' in self.df.columns else lambda x: None,
				'is_active': 'sum' if 'is_active' in self.df.columns else lambda x: None
			}).round(2)
			
			dept_analysis.columns = ['_'.join(col).strip() for col in dept_analysis.columns]
			dept_analysis = dept_analysis.sort_values(dept_analysis.columns[0], ascending=False)
			
			print("Department Analysis (Top 10):")
			print(dept_analysis.head(10))
			
			# Category breadth analysis
			if all(col in self.df.columns for col in ['dept_desc', 'dept_category_desc']):
				category_breadth = self.df.groupby('dept_desc')['dept_category_desc'].nunique().sort_values(ascending=False)
				
				print(f"\nCategory Breadth Analysis (Categories per Department):")
				print(category_breadth.head(10))
				
				# Identify departments with high/low complexity
				high_complexity = category_breadth[category_breadth > category_breadth.quantile(0.75)]
				low_complexity = category_breadth[category_breadth < category_breadth.quantile(0.25)]
				
				print(f"\nHigh Complexity Departments (>{category_breadth.quantile(0.75):.0f} categories):")
				print(high_complexity)
				
				print(f"\nLow Complexity Departments (<{category_breadth.quantile(0.25):.0f} categories):")
				print(low_complexity)
				
			# Visualize department distribution
			if len(dept_analysis) > 0:
				plt.figure(figsize=(15, 8))
				
				plt.subplot(2, 2, 1)
				top_depts = dept_analysis.head(15)
				plt.bar(range(len(top_depts)), top_depts.iloc[:, 0])
				plt.title('Product Count by Department (Top 15)')
				plt.xlabel('Department Rank')
				plt.ylabel('Product Count')
				plt.xticks(range(len(top_depts)), [
					# THIS IS WHERE YOUR CODE STOPS - HERE'S THE REST:
					
					dept[:15] + '...' if len(dept) > 15 else dept 
					for dept in top_depts.index
				], rotation=45, ha='right')
				
				# Average price by department
				if 'base_unit_retail_amt_mean' in dept_analysis.columns:
					plt.subplot(2, 2, 2)
					price_data = dept_analysis['base_unit_retail_amt_mean'].dropna().head(15)
					plt.bar(range(len(price_data)), price_data.values)
					plt.title('Average Price by Department (Top 15)')
					plt.xlabel('Department Rank')
					plt.ylabel('Average Price ($)')
					plt.xticks(range(len(price_data)), [
						dept[:15] + '...' if len(dept) > 15 else dept 
						for dept in price_data.index
					], rotation=45, ha='right')
					
				# Margin analysis by department
				if 'margin_percent_mean' in dept_analysis.columns:
					plt.subplot(2, 2, 3)
					margin_data = dept_analysis['margin_percent_mean'].dropna().head(15)
					plt.bar(range(len(margin_data)), margin_data.values)
					plt.title('Average Margin % by Department (Top 15)')
					plt.xlabel('Department Rank')
					plt.ylabel('Average Margin (%)')
					plt.xticks(range(len(margin_data)), [
						dept[:15] + '...' if len(dept) > 15 else dept 
						for dept in margin_data.index
					], rotation=45, ha='right')
					
				# Category complexity visualization
				if 'dept_category_desc' in self.df.columns:
					plt.subplot(2, 2, 4)
					complexity_data = category_breadth.head(15)
					plt.bar(range(len(complexity_data)), complexity_data.values)
					plt.title('Category Count by Department (Top 15)')
					plt.xlabel('Department Rank')
					plt.ylabel('Number of Categories')
					plt.xticks(range(len(complexity_data)), [
						dept[:15] + '...' if len(dept) > 15 else dept 
						for dept in complexity_data.index
					], rotation=45, ha='right')
				
				plt.tight_layout()
				plt.show()
				
			# Portfolio concentration analysis
			total_products = len(self.df)
			dept_counts = self.df['dept_desc'].value_counts()
			
			# Calculate cumulative percentage
			cumulative_products = dept_counts.cumsum()
			cumulative_percentage = cumulative_products / total_products * 100
			
			# Find departments contributing to 80% of products
			top_depts_80pct = cumulative_percentage <= 80
			num_top_depts = top_depts_80pct.sum()
			
			print(f"\nPortfolio Concentration Analysis:")
			print(f"  Top {num_top_depts} departments contain 80% of products ({num_top_depts/len(dept_counts)*100:.1f}% of departments)")
			print(f"  Bottom {len(dept_counts) - num_top_depts} departments contain 20% of products ({(len(dept_counts) - num_top_depts)/len(dept_counts)*100:.1f}% of departments)")
			
			# SKU density analysis
			if all(col in self.df.columns for col in ['dept_desc', 'dept_category_desc', 'dept_subcatg_desc']):
				sku_density = self.df.groupby('dept_desc').agg({
					'dept_category_desc': 'nunique',
					'dept_subcatg_desc': 'nunique',
					'item_nbr': 'count'
				})
				sku_density.columns = ['Categories', 'Subcategories', 'SKUs']
				sku_density['SKUs_per_Category'] = sku_density['SKUs'] / sku_density['Categories']
				sku_density['SKUs_per_Subcategory'] = sku_density['SKUs'] / sku_density['Subcategories']
				
				print(f"\nSKU Density Analysis (Top 10 by SKU count):")
				sku_density_sorted = sku_density.sort_values('SKUs', ascending=False)
				print(sku_density_sorted.head(10).round(1))
				
				# Identify departments with high SKU proliferation
				high_sku_density = sku_density[sku_density['SKUs_per_Category'] > sku_density['SKUs_per_Category'].quantile(0.75)]
				print(f"\nHigh SKU Proliferation Departments (SKU rationalization candidates):")
				print(high_sku_density.sort_values('SKUs_per_Category', ascending=False))
				
			# Category performance matrix
			if all(col in self.df.columns for col in ['dept_desc', 'base_unit_retail_amt', 'margin_percent']):
				category_matrix = self.df.groupby('dept_desc').agg({
					'item_nbr': 'count',
					'base_unit_retail_amt': 'mean',
					'margin_percent': 'mean'
				}).round(2)
				
				category_matrix.columns = ['Product_Count', 'Avg_Price', 'Avg_Margin']
				
				# Classify departments by volume and margin
				high_volume_threshold = category_matrix['Product_Count'].median()
				high_margin_threshold = category_matrix['Avg_Margin'].median()
				
				def classify_department(row):
					if row['Product_Count'] > high_volume_threshold and row['Avg_Margin'] > high_margin_threshold:
						return 'High Volume, High Margin'
					elif row['Product_Count'] > high_volume_threshold:
						return 'High Volume, Low Margin'
					elif row['Avg_Margin'] > high_margin_threshold:
						return 'Low Volume, High Margin'
					else:
						return 'Low Volume, Low Margin'
				
				category_matrix['Strategy_Classification'] = category_matrix.apply(classify_department, axis=1)
				
				print(f"\nDepartment Strategic Classification:")
				strategy_summary = category_matrix['Strategy_Classification'].value_counts()
				print(strategy_summary)
				
				# Show examples of each classification
				for strategy in strategy_summary.index:
					examples = category_matrix[category_matrix['Strategy_Classification'] == strategy].head(3)
					print(f"\n{strategy} Examples:")
					print(examples[['Product_Count', 'Avg_Price', 'Avg_Margin']])
					
		else:
			print("Required column 'dept_desc' not found for category management analysis")

# What your incomplete function was missing:
"""
MISSING COMPONENTS:
1. Complete visualization code (xticks labels, additional subplots)
2. Portfolio concentration analysis (80/20 rule)
3. SKU density analysis
4. High SKU proliferation identification
5. Category performance matrix
6. Strategic classification of departments
7. Example departments for each classification
8. Error handling for missing columns

YOUR CODE STOPPED AT: plt.xticks(range(len(top_depts)), [
MISSING: Everything after that line - about 70% of the function!

The complete function provides:
✅ Department rankings and analysis
✅ Category complexity assessment
✅ Complete visualizations (4 subplots)
✅ Portfolio concentration insights
✅ SKU rationalization opportunities
✅ Strategic department classification
✅ Actionable business recommendations
"""