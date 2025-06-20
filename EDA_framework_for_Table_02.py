# Store performance ranking
            total_revenue = store_revenue.iloc[:, 0]  # First column should be sum of sales
            top_stores = total_revenue.nlargest(10)
            bottom_stores = total_revenue.nsmallest(5)
            
            print("Top 10 Revenue Generating Stores:")
            print(top_stores)
            
            print("\nBottom 5 Revenue Generating Stores:")
            print(bottom_stores)
            
            # Store consistency analysis
            avg_revenue = store_revenue.iloc[:, 1]  # Mean sales
            revenue_std = store_revenue.iloc[:, 2]   # Std sales
            
            # Calculate coefficient of variation for consistency
            cv = revenue_std / avg_revenue
            cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
            
            most_consistent = cv.nsmallest(5)
            least_consistent = cv.nlargest(5)
            
            print(f"\nMost Consistent Stores (low variability):")
            print(most_consistent.round(3))
            
            print(f"\nLeast Consistent Stores (high variability):")
            print(least_consistent.round(3))
            
            # Store performance visualization
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            plt.bar(range(len(top_stores)), top_stores.values)
            plt.title('Top 10 Stores by Revenue')
            plt.xlabel('Store Rank')
            plt.ylabel('Total Revenue ($)')
            plt.xticks(range(len(top_stores)), [f'Store {s}' for s in top_stores.index], rotation=45)
            
            plt.subplot(1, 2, 2)
            plt.scatter(avg_revenue, revenue_std, alpha=0.6)
            plt.title('Store Performance: Average vs Variability')
            plt.xlabel('Average Weekly Sales ($)')
            plt.ylabel('Sales Variability (Std Dev)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Revenue concentration analysis (80/20 rule for stores)
            store_revenue_sorted = total_revenue.sort_values(ascending=False)
            cumulative_revenue = store_revenue_sorted.cumsum()
            total_network_revenue = store_revenue_sorted.sum()
            cumulative_percentage = cumulative_revenue / total_network_revenue * 100
            
            # Find stores contributing to 80% of revenue
            top_stores_80pct = cumulative_percentage <= 80
            num_top_stores = top_stores_80pct.sum()
            
            print(f"\nRevenue Concentration Analysis (80/20 Rule):")
            print(f"  Top {num_top_stores} stores generate 80% of revenue ({num_top_stores/len(total_revenue)*100:.1f}% of stores)")
            print(f"  Bottom {len(total_revenue) - num_top_stores} stores generate 20% of revenue ({(len(total_revenue) - num_top_stores)/len(total_revenue)*100:.1f}% of stores)")
            
    def medium_product_analysis(self):
        """2.2 Product Performance Analysis"""
        print("\n📦 2.2 PRODUCT PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in [self.product_col, 'wkly_sales_amt']):
            # Product revenue analysis
            product_performance = self.df.groupby(self.product_col).agg({
                'wkly_sales_amt': ['sum', 'mean', 'count'],
                'wkly_qty': ['sum', 'mean'] if 'wkly_qty' in self.df.columns else lambda x: None,
                'onhand_invt_qty': 'mean' if 'onhand_invt_qty' in self.df.columns else lambda x: None
            }).round(2)
            
            product_performance.columns = ['_'.join(col).strip() for col in product_performance.columns]
            
            # Top/bottom products
            total_product_revenue = product_performance.iloc[:, 0]
            top_products = total_product_revenue.nlargest(10)
            bottom_products = total_product_revenue[total_product_revenue > 0].nsmallest(10)
            
            print("Top 10 Revenue Generating Products:")
            print(top_products)
            
            print("\nBottom 10 Revenue Generating Products (excluding zeros):")
            print(bottom_products)
            
            # ABC Analysis (80/20 rule)
            sorted_products = total_product_revenue.sort_values(ascending=False)
            cumulative_revenue = sorted_products.cumsum()
            total_revenue_all = sorted_products.sum()
            cumulative_percentage = cumulative_revenue / total_revenue_all * 100
            
            # Classify products
            a_products = cumulative_percentage <= 80
            b_products = (cumulative_percentage > 80) & (cumulative_percentage <= 95)
            c_products = cumulative_percentage > 95
            
            print(f"\nABC Analysis:")
            print(f"  A-Products (80% of revenue): {a_products.sum()} products ({a_products.sum()/len(sorted_products)*100:.1f}%)")
            print(f"  B-Products (15% of revenue): {b_products.sum()} products ({b_products.sum()/len(sorted_products)*100:.1f}%)")
            print(f"  C-Products (5% of revenue): {c_products.sum()} products ({c_products.sum()/len(sorted_products)*100:.1f}%)")
            
            # Product velocity analysis
            if 'wkly_qty' in self.df.columns:
                product_velocity = self.df.groupby(self.product_col)['wkly_qty'].sum().sort_values(ascending=False)
                
                print(f"\nTop 5 Products by Quantity Sold:")
                print(product_velocity.head())
                
                # Fast vs slow moving products
                median_velocity = product_velocity.median()
                fast_moving = product_velocity > median_velocity
                slow_moving = product_velocity <= median_velocity
                
                print(f"\nProduct Velocity Classification:")
                print(f"  Fast-moving products: {fast_moving.sum()} ({fast_moving.sum()/len(product_velocity)*100:.1f}%)")
                print(f"  Slow-moving products: {slow_moving.sum()} ({slow_moving.sum()/len(product_velocity)*100:.1f}%)")
                
    def medium_temporal_analysis(self):
        """2.3 Temporal Analysis"""
        print("\n⏰ 2.3 TEMPORAL ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in [self.time_col, 'wkly_sales_amt']):
            # Weekly sales trends
            weekly_performance = self.df.groupby(self.time_col).agg({
                'wkly_sales_amt': ['sum', 'mean', 'count'],
                'wkly_qty': 'sum' if 'wkly_qty' in self.df.columns else lambda x: None,
                'onhand_invt_qty': 'mean' if 'onhand_invt_qty' in self.df.columns else lambda x: None
            }).round(2)
            
            weekly_performance.columns = ['_'.join(col).strip() for col in weekly_performance.columns]
            
            # Trend analysis
            weekly_revenue = weekly_performance.iloc[:, 0].sort_index()  # Total weekly revenue
            
            print("Weekly Performance Summary (first 10 weeks):")
            print(weekly_performance.head(10))
            
            # Growth analysis
            weekly_growth = weekly_revenue.pct_change() * 100
            avg_growth = weekly_growth.mean()
            growth_volatility = weekly_growth.std()
            
            print(f"\nGrowth Analysis:")
            print(f"  Average weekly growth: {avg_growth:.2f}%")
            print(f"  Growth volatility: {growth_volatility:.2f}%")
            print(f"  Best week: {weekly_revenue.idxmax()} (${weekly_revenue.max():,.2f})")
            print(f"  Worst week: {weekly_revenue.idxmin()} (${weekly_revenue.min():,.2f})")
            
            # Seasonal patterns (if full year available)
            if len(weekly_revenue) >= 50:  # Close to full year
                # Extract week numbers for seasonality
                if 'week' in self.df.columns:
                    seasonal_pattern = self.df.groupby('week')['wkly_sales_amt'].sum()
                    
                    peak_week = seasonal_pattern.idxmax()
                    low_week = seasonal_pattern.idxmin()
                    
                    print(f"\nSeasonal Patterns:")
                    print(f"  Peak sales week: Week {peak_week}")
                    print(f"  Lowest sales week: Week {low_week}")
                    print(f"  Seasonal variation: {(seasonal_pattern.max()/seasonal_pattern.min()-1)*100:.1f}%")
                    
            # Time series visualization
            plt.figure(figsize=(15, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(weekly_revenue.index, weekly_revenue.values, marker='o', linewidth=2)
            plt.title('Weekly Revenue Trend')
            plt.xlabel('Week')
            plt.ylabel('Total Revenue ($)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(weekly_growth.index, weekly_growth.values, marker='o', linewidth=2, color='orange')
            plt.title('Weekly Growth Rate')
            plt.xlabel('Week')
            plt.ylabel('Growth Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            if 'onhand_invt_qty' in self.df.columns:
                weekly_inventory = weekly_performance.iloc[:, -1].sort_index()  # Average weekly inventory
                plt.subplot(2, 2, 3)
                plt.plot(weekly_inventory.index, weekly_inventory.values, marker='o', linewidth=2, color='green')
                plt.title('Average Weekly Inventory')
                plt.xlabel('Week')
                plt.ylabel('Average Inventory')
                plt.grid(True, alpha=0.3)
                
            if 'week' in self.df.columns and len(weekly_revenue) >= 50:
                plt.subplot(2, 2, 4)
                plt.bar(seasonal_pattern.index, seasonal_pattern.values)
                plt.title('Seasonal Pattern (by Week of Year)')
                plt.xlabel('Week of Year')
                plt.ylabel('Total Sales ($)')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
    def medium_inventory_management(self):
        """2.4 Inventory Management Analysis"""
        print("\n📊 2.4 INVENTORY MANAGEMENT ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in ['onhand_invt_qty', 'wkly_sales_amt']):
            # Inventory turnover analysis
            store_inventory_analysis = self.df.groupby(self.store_col).agg({
                'onhand_invt_qty': ['mean', 'sum'],
                'wkly_sales_amt': ['sum', 'mean'],
                'sales_to_inventory_ratio': 'mean' if 'sales_to_inventory_ratio' in self.df.columns else lambda x: None
            }).round(3)
            
            store_inventory_analysis.columns = ['_'.join(col).strip() for col in store_inventory_analysis.columns]
            
            # Calculate inventory metrics
            avg_inventory = store_inventory_analysis.iloc[:, 0]  # Average inventory
            total_sales = store_inventory_analysis.iloc[:, 2]     # Total sales
            
            # Inventory efficiency (sales per inventory dollar - proxy)
            inventory_efficiency = total_sales / avg_inventory
            inventory_efficiency = inventory_efficiency.replace([np.inf, -np.inf], np.nan).dropna()
            
            print("Inventory Management Performance:")
            print("\nMost Efficient Stores (highest sales per inventory unit):")
            print(inventory_efficiency.nlargest(5))
            
            print("\nLeast Efficient Stores (lowest sales per inventory unit):")
            print(inventory_efficiency.nsmallest(5))
            
            # Stockout analysis
            stockout_analysis = self.df.groupby(self.store_col).agg({
                'onhand_invt_qty': lambda x: (x == 0).sum(),  # Count of stockouts
                self.store_col: 'count'  # Total records
            })
            stockout_analysis.columns = ['stockout_count', 'total_records']
            stockout_analysis['stockout_rate'] = stockout_analysis['stockout_count'] / stockout_analysis['total_records'] * 100
            
            print(f"\nStockout Analysis:")
            print(f"Average stockout rate: {stockout_analysis['stockout_rate'].mean():.2f}%")
            
            worst_stockout_stores = stockout_analysis['stockout_rate'].nlargest(5)
            print(f"Stores with highest stockout rates:")
            print(worst_stockout_stores.round(2))
            
            # Overstock analysis
            if len(self.df) > 0:
                inventory_75th = self.df['onhand_invt_qty'].quantile(0.75)
                overstock_analysis = self.df.groupby(self.store_col).agg({
                    'onhand_invt_qty': lambda x: (x > inventory_75th).sum(),
                    self.store_col: 'count'
                })
                overstock_analysis.columns = ['overstock_count', 'total_records']
                overstock_analysis['overstock_rate'] = overstock_analysis['overstock_count'] / overstock_analysis['total_records'] * 100
                
                print(f"\nOverstock Analysis (inventory > 75th percentile):")
                print(f"Average overstock rate: {overstock_analysis['overstock_rate'].mean():.2f}%")
                
                highest_overstock = overstock_analysis['overstock_rate'].nlargest(5)
                print(f"Stores with highest overstock rates:")
                print(highest_overstock.round(2))
                
    def medium_efficiency_analysis(self):
        """2.5 Efficiency Analysis"""
        print("\n⚡ 2.5 EFFICIENCY ANALYSIS")
        print("-" * 40)
        
        # Sales efficiency metrics
        if all(col in self.df.columns for col in ['wkly_sales_amt', 'onhand_invt_qty']):
            # Create efficiency segments
            efficiency_data = self.df[
                (self.df['wkly_sales_amt'] > 0) & (self.df['onhand_invt_qty'] > 0)
            ].copy()
            
            if len(efficiency_data) > 0:
                # Calculate efficiency quartiles
                efficiency_data['efficiency_quartile'] = pd.qcut(
                    efficiency_data['sales_to_inventory_ratio'], 
                    q=4, labels=['Low Efficiency', 'Medium-Low', 'Medium-High', 'High Efficiency']
                )
                
                efficiency_summary = efficiency_data.groupby('efficiency_quartile').agg({
                    'wkly_sales_amt': ['mean', 'count'],
                    'onhand_invt_qty': 'mean',
                    'sales_to_inventory_ratio': 'mean'
                }).round(3)
                
                print("Efficiency Analysis by Quartile:")
                print(efficiency_summary)
                
                # Store efficiency ranking
                store_efficiency = efficiency_data.groupby(self.store_col)['sales_to_inventory_ratio'].mean().sort_values(ascending=False)
                
                print(f"\nTop 5 Most Efficient Stores:")
                print(store_efficiency.head())
                
                print(f"\nBottom 5 Least Efficient Stores:")
                print(store_efficiency.tail())

    # ========================================
    # LEVEL 3: ADVANCED ANALYSIS
    # ========================================
    
    def level_3_advanced_analysis(self):
        """Complete Level 3: Advanced Analysis"""
        print("\n" + "="*60)
        print("LEVEL 3: ADVANCED ANALYSIS")
        print("="*60)
        
        self.advanced_store_clustering()
        self.advanced_abc_xyz_analysis()
        self.advanced_inventory_optimization()
        self.advanced_forecasting_insights()
        self.advanced_network_analysis()
        
        print("\n✅ LEVEL 3 ADVANCED ANALYSIS COMPLETED")
        
    def advanced_store_clustering(self):
        """3.1 Advanced Store Clustering"""
        print("\n🎯 3.1 STORE CLUSTERING ANALYSIS")
        print("-" * 40)
        
        if self.store_col in self.df.columns:
            # Create store-level features for clustering
            store_features = self.df.groupby(self.store_col).agg({
                'wkly_sales_amt': ['mean', 'std', 'sum'],
                'wkly_qty': ['mean', 'sum'] if 'wkly_qty' in self.df.columns else lambda x: None,
                'onhand_invt_qty': ['mean', 'std'],
                'sales_to_inventory_ratio': 'mean' if 'sales_to_inventory_ratio' in self.df.columns else lambda x: None
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
                
                # Determine optimal number of clusters
                n_clusters = min(5, len(store_features) // 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_features)
                
                store_features['cluster'] = clusters
                
                # Analyze clusters
                cluster_summary = store_features.groupby('cluster').agg({
                    store_features.columns[0]: 'mean',  # Average sales
                    store_features.columns[2]: 'mean',  # Total sales
                    store_features.columns[3]: 'mean',  # Average inventory
                    'cluster': 'count'
                }).round(2)
                cluster_summary.columns = ['Avg_Weekly_Sales', 'Total_Sales', 'Avg_Inventory', 'Store_Count']
                
                print(f"Store Clustering Results ({n_clusters} clusters):")
                print(cluster_summary)
                
                # Cluster interpretation
                for cluster_id in range(n_clusters):
                    cluster_stores = store_features[store_features['cluster'] == cluster_id]
                    avg_sales = cluster_stores.iloc[:, 0].mean()
                    avg_inventory = cluster_stores.iloc[:, 3].mean()
                    
                    if avg_sales > store_features.iloc[:, 0].median() and avg_inventory < store_features.iloc[:, 3].median():
                        cluster_type = "High Performance (High Sales, Low Inventory)"
                    elif avg_sales > store_features.iloc[:, 0].median():
                        cluster_type = "High Volume"
                    elif avg_inventory < store_features.iloc[:, 3].median():
                        cluster_type = "Lean Operations"
                    else:
                        cluster_type = "Standard Operations"
                        
                    print(f"  Cluster {cluster_id}: {cluster_type}")
                    
                # PCA visualization
                if len(store_features.columns) > 3:
                    pca = PCA(n_components=2)
                    pca_features = pca.fit_transform(scaled_features)
                    
                    plt.figure(figsize=(12, 8))
                    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                                        c=clusters, cmap='viridis', alpha=0.7, s=60)
                    plt.colorbar(scatter, label='Cluster')
                    plt.title('Store Clustering Visualization (PCA Space)')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    
                    # Add store labels for reference
                    for i, store in enumerate(store_features.index):
                        if i % 3 == 0:  # Label every 3rd store to avoid overcrowding
                            plt.annotate(f'S{store}', (pca_features[i, 0], pca_features[i, 1]), 
                                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
                    
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                    
    def advanced_abc_xyz_analysis(self):
        """3.2 Advanced ABC/XYZ Analysis"""
        print("\n📈 3.2 ABC/XYZ ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in [self.product_col, 'wkly_sales_amt']):
            # ABC Analysis (Revenue-based)
            product_revenue = self.df.groupby(self.product_col)['wkly_sales_amt'].sum().sort_values(ascending=False)
            total_revenue = product_revenue.sum()
            cumulative_revenue = product_revenue.cumsum()
            cumulative_percentage = cumulative_revenue / total_revenue * 100
            
            # ABC Classification
            abc_classification = []
            for pct in cumulative_percentage:
                if pct <= 80:
                    abc_classification.append('A')
                elif pct <= 95:
                    abc_classification.append('B')
                else:
                    abc_classification.append('C')
                    
            # XYZ Analysis (Demand Variability)
            product_variability = self.df.groupby(self.product_col)['wkly_sales_amt'].agg(['mean', 'std']).reset_index()
            product_variability['cv'] = product_variability['std'] / product_variability['mean']
            product_variability['cv'] = product_variability['cv'].fillna(0)
            
            # XYZ Classification based on coefficient of variation
            cv_33rd = product_variability['cv'].quantile(0.33)
            cv_67th = product_variability['cv'].quantile(0.67)
            
            xyz_classification = []
            for cv in product_variability['cv']:
                if cv <= cv_33rd:
                    xyz_classification.append('X')  # Predictable
                elif cv <= cv_67th:
                    xyz_classification.append('Y')  # Variable
                else:
                    xyz_classification.append('Z')  # Unpredictable
                    
            # Combine classifications
            abc_df = pd.DataFrame({
                'product': product_revenue.index,
                'revenue': product_revenue.values,
                'abc_class': abc_classification
            })
            
            xyz_df = pd.DataFrame({
                'product': product_variability[self.product_col],
                'cv': product_variability['cv'],
                'xyz_class': xyz_classification
            })
            
            combined_analysis = abc_df.merge(xyz_df, on='product')
            combined_analysis['abc_xyz'] = combined_analysis['abc_class'] + combined_analysis['xyz_class']
            
            # Summary by classification
            classification_summary = combined_analysis.groupby('abc_xyz').agg({
                'product': 'count',
                'revenue': 'sum'
            }).round(2)
            classification_summary['revenue_pct'] = classification_summary['revenue'] / total_revenue * 100
            
            print("ABC/XYZ Classification Summary:")
            print(classification_summary)
            
            # Strategic recommendations
            print(f"\nStrategic Recommendations by Category:")
            
            for category in ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']:
                if category in classification_summary.index:
                    count = classification_summary.loc[category, 'product']
                    revenue_pct = classification_summary.loc[category, 'revenue_pct']
                    
                    if category == 'AX':
                        strategy = "Key products - maintain high service levels"
                    elif category == 'AY':
                        strategy = "Important but variable - improve forecasting"
                    elif category == 'AZ':
                        strategy = "High value, unpredictable - safety stock strategy"
                    elif category[0] == 'B':
                        strategy = "Medium importance - standard management"
                    else:  # C category
                        strategy = "Low priority - consider discontinuation"
                        
                    print(f"  {category}: {count} products ({revenue_pct:.1f}% revenue) - {strategy}")
                    
    def advanced_inventory_optimization(self):
        """3.3 Advanced Inventory Optimization"""
        print("\n📦 3.3 INVENTORY OPTIMIZATION ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in ['wkly_sales_amt', 'onhand_invt_qty', 'wkly_qty']):
            # Calculate key inventory metrics
            inventory_metrics = self.df.groupby([self.store_col, self.product_col]).agg({
                'wkly_sales_amt': ['sum', 'mean', 'std'],
                'wkly_qty': ['sum', 'mean', 'std'],
                'onhand_invt_qty': ['mean', 'std']
            }).round(3)
            
            inventory_metrics.columns = ['_'.join(col).strip() for col in inventory_metrics.columns]
            
            # Calculate inventory turnover (simplified)
            total_sold = inventory_metrics.iloc[:, 4]  # Total quantity sold
            avg_inventory = inventory_metrics.iloc[:, 7]  # Average inventory
            
            # Avoid division by zero
            inventory_turnover = total_sold / (avg_inventory + 0.1)
            inventory_turnover = inventory_turnover.replace([np.inf, -np.inf], np.nan).dropna()
            
            print("Inventory Turnover Analysis:")
            print(f"  Average turnover ratio: {inventory_turnover.mean():.2f}")
            print(f"  Median turnover ratio: {inventory_turnover.median():.2f}")
            
            # High and low turnover items
            high_turnover = inventory_turnover.nlargest(10)
            low_turnover = inventory_turnover.nsmallest(10)
            
            print(f"\nHighest Turnover (Store-Product combinations):")
            for (store, product), turnover in high_turnover.items():
                print(f"  Store {store}, Product {product}: {turnover:.2f}")
                
            print(f"\nLowest Turnover (Store-Product combinations):")
            for (store, product), turnover in low_turnover.items():
                print(f"  Store {store}, Product {product}: {turnover:.2f}")
                
            # Service level analysis (proxy using stockout frequency)
            stockout_analysis = self.df.groupby([self.store_col, self.product_col]).agg({
                'onhand_invt_qty': [lambda x: (x == 0).sum(), 'count']
            })
            stockout_analysis.columns = ['stockouts', 'total_periods']
            stockout_analysis['service_level'] = (1 - stockout_analysis['stockouts'] / stockout_analysis['total_periods']) * 100
            
            avg_service_level = stockout_analysis['service_level'].mean()
            print(f"\nService Level Analysis:")
            print(f"  Average service level: {avg_service_level:.1f}%")
            
            low_service_items = stockout_analysis[stockout_analysis['service_level'] < 90]
            print(f"  Items with service level < 90%: {len(low_service_items)}")
            
    def advanced_forecasting_insights(self):
        """3.4 Advanced Forecasting Insights"""
        print("\n🔮 3.4 FORECASTING INSIGHTS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in [self.time_col, 'wkly_sales_amt']):
            # Demand pattern analysis
            weekly_demand = self.df.groupby(self.time_col)['wkly_sales_amt'].sum().sort_index()
            
            if len(weekly_demand) >= 12:  # Need sufficient data
                # Calculate demand characteristics
                demand_mean = weekly_demand.mean()
                demand_std = weekly_demand.std()
                demand_cv = demand_std / demand_mean
                
                print("Demand Characteristics:")
                print(f"  Average weekly demand: ${demand_mean:,.2f}")
                print(f"  Demand variability (CV): {demand_cv:.3f}")
                
                # Trend analysis
                from scipy import stats as scipy_stats
                weeks = np.arange(len(weekly_demand))
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(weeks, weekly_demand.values)
                
                print(f"\nTrend Analysis:")
                print(f"  Weekly trend: ${slope:.2f} per week")
                print(f"  Trend strength (R²): {r_value**2:.3f}")
                print(f"  Trend significance: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.3f})")
                
                # Seasonality detection (simplified)
                if 'week' in self.df.columns:
                    seasonal_demand = self.df.groupby('week')['wkly_sales_amt'].sum()
                    seasonal_cv = seasonal_demand.std() / seasonal_demand.mean()
                    
                    print(f"\nSeasonality Analysis:")
                    print(f"  Seasonal variation (CV): {seasonal_cv:.3f}")
                    print(f"  Peak week: {seasonal_demand.idxmax()}")
                    print(f"  Low week: {seasonal_demand.idxmin()}")
                    
                # Forecast accuracy proxy (using recent vs historical average)
                if len(weekly_demand) >= 20:
                    recent_weeks = weekly_demand.tail(8).mean()  # Last 8 weeks
                    historical_avg = weekly_demand.head(12).mean()  # First 12 weeks
                    
                    accuracy_error = abs(recent_weeks - historical_avg) / historical_avg * 100
                    print(f"\nForecast Accuracy Proxy:")
                    print(f"  Recent vs Historical deviation: {accuracy_error:.1f}%")
                    
    def advanced_network_analysis(self):
        """3.5 Network Analysis"""
        print("\n🔗 3.5 STORE NETWORK ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in [self.store_col, self.time_col, 'wkly_sales_amt']):
            # Create store-week sales matrix for correlation analysis
            sales_matrix = self.df.pivot_table(
                index=self.time_col,
                columns=self.store_col,
                values='wkly_sales_amt',
                fill_value=0
            )
            
            if sales_matrix.shape[1] > 2:  # Need at least 3 stores
                # Calculate store correlations
                store_correlations = sales_matrix.corr()
                
                # Network statistics
                correlation_values = store_correlations.values[np.triu_indices_from(store_correlations.values, k=1)]
                avg_correlation = correlation_values.mean()
                max_correlation = correlation_values.max()
                min_correlation = correlation_values.min()
                
                print(f"Store Network Analysis:")
                print(f"  Average store correlation: {avg_correlation:.3f}")
                print(f"  Highest correlation: {max_correlation:.3f}")
                print(f"  Lowest correlation: {min_correlation:.3f}")
                
                # Find most/least connected stores
                store_avg_corr = store_correlations.mean()
                most_connected = store_avg_corr.idxmax()
                least_connected = store_avg_corr.idxmin()
                
                print(f"  Most connected store: {most_connected} (avg corr: {store_avg_corr[most_connected]:.3f})")
                print(f"  Least connected store: {least_connected} (avg corr: {store_avg_corr[least_connected]:.3f})")
                
                # Store similarity analysis
                high_similarity_threshold = 0.7
                similar_store_pairs = []
                
                for i in range(len(store_correlations.columns)):
                    for j in range(i+1, len(store_correlations.columns)):
                        corr = store_correlations.iloc[i, j]
                        if corr > high_similarity_threshold:
                            store1 = store_correlations.columns[i]
                            store2 = store_correlations.columns[j]
                            similar_store_pairs.append((store1, store2, corr))
                
                if similar_store_pairs:
                    print(f"\nHighly Similar Store Pairs (correlation > {high_similarity_threshold}):")
                    for store1, store2, corr in similar_store_pairs[:10]:  # Show top 10
                        print(f"  Store {store1} & {store2}: {corr:.3f}")
                else:
                    print(f"\nNo store pairs with correlation > {high_similarity_threshold}")
                
                # Store influence analysis
                # Calculate lead-lag relationships (simplified)
                influence_scores = {}
                for store in sales_matrix.columns[:10]:  # Analyze first 10 stores to save time
                    store_sales = sales_matrix[store]
                    influence_count = 0
                    
                    for other_store in sales_matrix.columns:
                        if store != other_store:
                            other_sales = sales_matrix[other_store]
                            
                            # Check if store's sales in week t correlate with other store's sales in week t+1
                            if len(store_sales) > 1:
                                correlation = store_sales[:-1].corr(other_sales[1:])
                                if correlation > 0.3:  # Threshold for influence
                                    influence_count += 1
                    
                    influence_scores[store] = influence_count
                
                if influence_scores:
                    most_influential = max(influence_scores, key=influence_scores.get)
                    print(f"\nStore Influence Analysis:")
                    print(f"  Most influential store: {most_influential} (influences {influence_scores[most_influential]} stores)")
                
    def generate_comprehensive_report(self):
        """Generate comprehensive business report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE SALES & INVENTORY ANALYSIS REPORT")
        print("="*80)
        
        # Business summary
        total_revenue = self.df['wkly_sales_amt'].sum() if 'wkly_sales_amt' in self.df.columns else 0
        total_units = self.df['wkly_qty'].sum() if 'wkly_qty' in self.df.columns else 0
        unique_stores = self.df[self.store_col].nunique() if self.store_col in self.df.columns else 0
        unique_products = self.df[self.product_col].nunique() if self.product_col in self.df.columns else 0
        unique_weeks = self.df[self.time_col].nunique() if self.time_col in self.df.columns else 0
        
        print(f"\n📊 BUSINESS PERFORMANCE SUMMARY")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Total Units Sold: {total_units:,.0f}")
        print(f"   Store Network: {unique_stores} locations")
        print(f"   Product Portfolio: {unique_products:,} SKUs")
        print(f"   Analysis Period: {unique_weeks} weeks")
        
        # Key insights
        insights = []
        
        # Revenue insights
        if 'wkly_sales_amt' in self.df.columns:
            avg_weekly_revenue = self.df.groupby(self.time_col)['wkly_sales_amt'].sum().mean()
            insights.append(f"Average weekly network revenue: ${avg_weekly_revenue:,.2f}")
            
            # Top performer
            if unique_stores > 0:
                top_store = self.df.groupby(self.store_col)['wkly_sales_amt'].sum().idxmax()
                top_store_revenue = self.df.groupby(self.store_col)['wkly_sales_amt'].sum().max()
                insights.append(f"Top performing store: #{top_store} (${top_store_revenue:,.2f} total revenue)")
        
        # Inventory insights
        if 'onhand_invt_qty' in self.df.columns:
            avg_inventory = self.df['onhand_invt_qty'].mean()
            stockout_rate = (self.df['onhand_invt_qty'] == 0).mean() * 100
            insights.append(f"Average inventory level: {avg_inventory:.0f} units")
            insights.append(f"Stockout rate: {stockout_rate:.1f}%")
        
        # Product insights
        if unique_products > 0:
            # ABC analysis summary
            product_revenue = self.df.groupby(self.product_col)['wkly_sales_amt'].sum().sort_values(ascending=False)
            total_revenue_products = product_revenue.sum()
            top_20_pct_products = int(len(product_revenue) * 0.2)
            top_20_pct_revenue = product_revenue.head(top_20_pct_products).sum()
            top_20_pct_share = top_20_pct_revenue / total_revenue_products * 100 if total_revenue_products > 0 else 0
            
            insights.append(f"Top 20% of products generate {top_20_pct_share:.1f}% of revenue")
        
        # Efficiency insights
        if 'sales_to_inventory_ratio' in self.df.columns:
            avg_efficiency = self.df['sales_to_inventory_ratio'].mean()
            insights.append(f"Average sales-to-inventory ratio: {avg_efficiency:.3f}")
        
        print(f"\n🔍 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        # Performance recommendations
        recommendations = []
        
        if 'onhand_invt_qty' in self.df.columns:
            stockout_rate = (self.df['onhand_invt_qty'] == 0).mean() * 100
            if stockout_rate > 10:
                recommendations.append("High stockout rate detected - review inventory planning")
            elif stockout_rate < 2:
                recommendations.append("Very low stockout rate - consider inventory reduction")
        
        if unique_stores > 5:
            recommendations.append("Multi-store network - implement store benchmarking program")
        
        if unique_products > 1000:
            recommendations.append("Large product portfolio - consider ABC analysis for optimization")
        
        print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return {
            'total_revenue': total_revenue,
            'unique_stores': unique_stores,
            'unique_products': unique_products,
            'unique_weeks': unique_weeks,
            'insights': insights,
            'recommendations': recommendations
        }
    
    def run_complete_analysis(self):
        """Execute complete three-tier analysis"""
        print("🚀 STARTING COMPREHENSIVE SALES & INVENTORY ANALYSIS")
        print("="*80)
        
        # Run all analysis levels
        self.level_1_basic_analysis()
        self.level_2_medium_analysis()
        self.level_3_advanced_analysis()
        
        # Generate final report
        report = self.generate_comprehensive_report()
        
        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*80)
        
        return report

# Utility functions for specific analyses
def calculate_inventory_kpis(df, store_col='store_nbr', sales_col='wkly_sales_amt', inventory_col='onhand_invt_qty'):
    """Calculate key inventory performance indicators"""
    if all(col in df.columns for col in [store_col, sales_col, inventory_col]):
        store_kpis = df.groupby(store_col).agg({
            sales_col: ['sum', 'mean'],
            inventory_col: ['mean', 'std'],
            inventory_col: lambda x: (x == 0).sum()  # Stockouts
        })
        
        # Calculate derived metrics
        store_kpis['inventory_turnover'] = store_kpis[(sales_col, 'sum')] / store_kpis[(inventory_col, 'mean')]
        store_kpis['service_level'] = 1 - (store_kpis[(inventory_col, '<lambda>')] / len(df.groupby(store_col).size()))
        
        return store_kpis

def product_lifecycle_analysis(df, product_col='item_nbr', time_col='wm_yr_wk_nbr', sales_col='wkly_sales_amt'):
    """Analyze product lifecycle stages"""
    if all(col in df.columns for col in [product_col, time_col, sales_col]):
        product_lifecycle = []
        
        for product in df[product_col].unique()[:100]:  # Analyze first 100 products
            product_data = df[df[product_col] == product].sort_values(time_col)
            
            if len(product_data) >= 8:  # Need sufficient data
                sales_trend = product_data[sales_col].values
                
                # Simple lifecycle detection
                first_half = sales_trend[:len(sales_trend)//2].mean()
                second_half = sales_trend[len(sales_trend)//2:].mean()
                
                if second_half > first_half * 1.2:
                    stage = 'Growth'
                elif second_half < first_half * 0.8:
                    stage = 'Decline'
                else:
                    stage = 'Mature'
                
                product_lifecycle.append({
                    'product': product,
                    'stage': stage,
                    'total_sales': sales_trend.sum(),
                    'trend_ratio': second_half / first_half if first_half > 0 else 1
                })
        
        lifecycle_df = pd.DataFrame(product_lifecycle)
        
        print("Product Lifecycle Analysis:")
        if len(lifecycle_df) > 0:
            stage_summary = lifecycle_df['stage'].value_counts()
            print(stage_summary)
        
        return lifecycle_df

def demand_variability_analysis(df, product_col='item_nbr', sales_col='wkly_sales_amt'):
    """Analyze demand variability patterns"""
    if all(col in df.columns for col in [product_col, sales_col]):
        variability_analysis = df.groupby(product_col)[sales_col].agg(['mean', 'std', 'count']).reset_index()
        variability_analysis['cv'] = variability_analysis['std'] / variability_analysis['mean']
        variability_analysis['cv'] = variability_analysis['cv'].fillna(0)
        
        # Classify by variability
        cv_low = variability_analysis['cv'].quantile(0.33)
        cv_high = variability_analysis['cv'].quantile(0.67)
        
        def classify_variability(cv):
            if cv <= cv_low:
                return 'Predictable'
            elif cv <= cv_high:
                return 'Variable'
            else:
                return 'Highly Variable'
        
        variability_analysis['variability_class'] = variability_analysis['cv'].apply(classify_variability)
        
        print("Demand Variability Analysis:")
        variability_summary = variability_analysis['variability_class'].value_counts()
        print(variability_summary)
        
        return variability_analysis

# Main execution
def main():
    """Main execution function"""
    print("Sales & Inventory EDA Framework")
    print("="*40)
    
    print("Analysis Components:")
    print("✓ Sales performance analysis")
    print("✓ Inventory management evaluation") 
    print("✓ Store efficiency benchmarking")
    print("✓ Product portfolio optimization")
    print("✓ ABC/XYZ classification")
    print("✓ Network relationship analysis")
    print("✓ Forecasting insights")
    print("✓ Inventory optimization")
    
    print("\nUsage:")
    print("eda = SalesInventoryEDA('your_sales_data.csv')")
    print("report = eda.run_complete_analysis()")

if __name__ == "__main__":
    main()

print("\n🎯 Sales & Inventory EDA Framework Ready!")
print("This implementation provides:")
print("• Comprehensive sales performance analysis")
print("• Advanced inventory management insights")
print("• Store clustering and benchmarking")
print("• Product portfolio optimization")
print("• ABC/XYZ analysis for strategic planning")
print("• Network analysis for operational coordination")
print("• Forecasting insights for demand planning")
print("• Business-ready recommendations and reporting")# Sales & Inventory EDA Implementation
# Comprehensive Analysis for Store Performance & Inventory Management

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

class SalesInventoryEDA:
    """
    Comprehensive EDA for Sales and Inventory Management Data
    Focus: Store performance, inventory optimization, and business insights
    """
    
    def __init__(self, data_path):
        """Initialize with data loading and preparation"""
        self.df = self.load_and_prepare_data(data_path)
        self.setup_analysis_variables()
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare the sales/inventory dataset"""
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        # Convert numeric columns
        numeric_columns = [
            'wkly_sales_amt', 'wkly_qty', 'onhand_invt_qty', 
            'in_trnst_invt_qty', 'wm_yr_wk_nbr'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived metrics
        self.create_derived_metrics(df)
        
        return df
    
    def create_derived_metrics(self, df):
        """Create business-relevant derived metrics"""
        # Sales efficiency metrics
        if all(col in df.columns for col in ['wkly_sales_amt', 'onhand_invt_qty']):
            # Sales to inventory ratio (inventory turnover indicator)
            df['sales_to_inventory_ratio'] = df['wkly_sales_amt'] / (df['onhand_invt_qty'] + 0.01)  # Avoid division by zero
            
        # Inventory metrics
        if 'onhand_invt_qty' in df.columns and 'in_trnst_invt_qty' in df.columns:
            df['total_inventory'] = df['onhand_invt_qty'] + df['in_trnst_invt_qty']
            
        # Sales performance indicators
        if 'wkly_sales_amt' in df.columns:
            df['has_sales'] = df['wkly_sales_amt'] > 0
            df['sales_category'] = pd.cut(df['wkly_sales_amt'], 
                                        bins=[0, 1, 10, 100, float('inf')], 
                                        labels=['No Sales', 'Low Sales', 'Medium Sales', 'High Sales'])
            
        # Inventory status
        if 'onhand_invt_qty' in df.columns:
            df['inventory_status'] = pd.cut(df['onhand_invt_qty'],
                                          bins=[0, 1, 50, 500, float('inf')],
                                          labels=['Out of Stock', 'Low Stock', 'Normal Stock', 'High Stock'])
            
        # Week/year extraction
        if 'wm_yr_wk_nbr' in df.columns:
            df['year'] = df['wm_yr_wk_nbr'] // 100
            df['week'] = df['wm_yr_wk_nbr'] % 100
            
    def setup_analysis_variables(self):
        """Setup key variables for analysis"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Key business metrics
        self.sales_cols = ['wkly_sales_amt', 'wkly_qty']
        self.inventory_cols = ['onhand_invt_qty', 'in_trnst_invt_qty', 'total_inventory']
        self.performance_cols = ['sales_to_inventory_ratio']
        
        # Identifiers
        self.store_col = 'store_nbr'
        self.product_col = 'item_nbr'
        self.time_col = 'wm_yr_wk_nbr'

    # ========================================
    # LEVEL 1: BASIC EDA CHECKS
    # ========================================
    
    def level_1_basic_analysis(self):
        """Complete Level 1: Basic Sales & Inventory Analysis"""
        print("="*60)
        print("LEVEL 1: BASIC SALES & INVENTORY ANALYSIS")
        print("="*60)
        
        self.basic_business_overview()
        self.basic_sales_analysis()
        self.basic_inventory_analysis()
        self.basic_quality_assessment()
        self.basic_visualizations()
        
        print("\n✅ LEVEL 1 BASIC ANALYSIS COMPLETED")
        
    def basic_business_overview(self):
        """1.1 Business Overview"""
        print("\n📊 1.1 BUSINESS OVERVIEW")
        print("-" * 40)
        
        # Dataset dimensions
        print(f"Dataset Size: {self.df.shape[0]:,} records × {self.df.shape[1]} columns")
        
        # Business coverage
        unique_stores = self.df[self.store_col].nunique() if self.store_col in self.df.columns else 0
        unique_products = self.df[self.product_col].nunique() if self.product_col in self.df.columns else 0
        unique_weeks = self.df[self.time_col].nunique() if self.time_col in self.df.columns else 0
        
        print(f"\nBusiness Coverage:")
        print(f"  Stores: {unique_stores:,}")
        print(f"  Products: {unique_products:,}")
        print(f"  Time Periods: {unique_weeks}")
        
        # Time period analysis
        if self.time_col in self.df.columns:
            time_range = f"{self.df[self.time_col].min()} to {self.df[self.time_col].max()}"
            print(f"  Period Range: {time_range}")
            
        # Revenue overview
        if 'wkly_sales_amt' in self.df.columns:
            total_revenue = self.df['wkly_sales_amt'].sum()
            avg_weekly_revenue = self.df.groupby(self.time_col)['wkly_sales_amt'].sum().mean()
            print(f"\nRevenue Overview:")
            print(f"  Total Revenue: ${total_revenue:,.2f}")
            print(f"  Average Weekly Revenue: ${avg_weekly_revenue:,.2f}")
            
    def basic_sales_analysis(self):
        """1.2 Basic Sales Performance Analysis"""
        print("\n💰 1.2 SALES PERFORMANCE OVERVIEW")
        print("-" * 40)
        
        if 'wkly_sales_amt' in self.df.columns:
            sales_summary = self.df['wkly_sales_amt'].describe()
            print("Weekly Sales Amount Summary:")
            print(sales_summary.round(2))
            
            # Sales distribution
            zero_sales = (self.df['wkly_sales_amt'] == 0).sum()
            positive_sales = (self.df['wkly_sales_amt'] > 0).sum()
            total_records = len(self.df)
            
            print(f"\nSales Distribution:")
            print(f"  Records with sales: {positive_sales:,} ({positive_sales/total_records*100:.1f}%)")
            print(f"  Records with no sales: {zero_sales:,} ({zero_sales/total_records*100:.1f}%)")
            
        if 'wkly_qty' in self.df.columns:
            qty_summary = self.df['wkly_qty'].describe()
            print(f"\nWeekly Quantity Summary:")
            print(qty_summary.round(2))
            
        # Store performance overview
        if all(col in self.df.columns for col in [self.store_col, 'wkly_sales_amt']):
            store_performance = self.df.groupby(self.store_col)['wkly_sales_amt'].sum().sort_values(ascending=False)
            print(f"\nTop 5 Revenue Generating Stores:")
            print(store_performance.head().round(2))
            
    def basic_inventory_analysis(self):
        """1.3 Basic Inventory Analysis"""
        print("\n📦 1.3 INVENTORY OVERVIEW")
        print("-" * 40)
        
        if 'onhand_invt_qty' in self.df.columns:
            inventory_summary = self.df['onhand_invt_qty'].describe()
            print("On-Hand Inventory Summary:")
            print(inventory_summary.round(2))
            
            # Inventory status
            zero_inventory = (self.df['onhand_invt_qty'] == 0).sum()
            positive_inventory = (self.df['onhand_invt_qty'] > 0).sum()
            
            print(f"\nInventory Status:")
            print(f"  Items with inventory: {positive_inventory:,} ({positive_inventory/len(self.df)*100:.1f}%)")
            print(f"  Items out of stock: {zero_inventory:,} ({zero_inventory/len(self.df)*100:.1f}%)")
            
        if 'in_trnst_invt_qty' in self.df.columns:
            transit_summary = self.df['in_trnst_invt_qty'].describe()
            print(f"\nIn-Transit Inventory Summary:")
            print(transit_summary.round(2))
            
        # Store inventory levels
        if all(col in self.df.columns for col in [self.store_col, 'onhand_invt_qty']):
            store_inventory = self.df.groupby(self.store_col)['onhand_invt_qty'].sum().sort_values(ascending=False)
            print(f"\nTop 5 Inventory Holding Stores:")
            print(store_inventory.head().round(0))
            
    def basic_quality_assessment(self):
        """1.4 Data Quality Assessment"""
        print("\n🛡️ 1.4 DATA QUALITY ASSESSMENT")
        print("-" * 40)
        
        # Missing data analysis
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        print("Missing Data Summary:")
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_summary[missing_summary.Missing_Count > 0].head(10))
        
        # Business logic validation
        print("\nBusiness Logic Validation:")
        
        # Negative sales check
        if 'wkly_sales_amt' in self.df.columns:
            negative_sales = (self.df['wkly_sales_amt'] < 0).sum()
            print(f"  Negative sales records: {negative_sales}")
            
        # Negative inventory check
        if 'onhand_invt_qty' in self.df.columns:
            negative_inventory = (self.df['onhand_invt_qty'] < 0).sum()
            print(f"  Negative inventory records: {negative_inventory}")
            
        # Sales without inventory
        if all(col in self.df.columns for col in ['wkly_sales_amt', 'onhand_invt_qty']):
            sales_no_inventory = ((self.df['wkly_sales_amt'] > 0) & (self.df['onhand_invt_qty'] == 0)).sum()
            print(f"  Sales with zero inventory: {sales_no_inventory}")
            
        # Duplicates
        if all(col in self.df.columns for col in [self.store_col, self.product_col, self.time_col]):
            duplicates = self.df.duplicated(subset=[self.store_col, self.product_col, self.time_col]).sum()
            print(f"  Duplicate store-product-week records: {duplicates}")
            
    def basic_visualizations(self):
        """1.5 Basic Visualizations"""
        print("\n📈 1.5 BASIC VISUALIZATIONS")
        print("-" * 40)
        
        # Create visualization grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sales & Inventory Overview', fontsize=16, fontweight='bold')
        
        # Sales distribution
        if 'wkly_sales_amt' in self.df.columns:
            sales_data = self.df[self.df['wkly_sales_amt'] > 0]['wkly_sales_amt']  # Exclude zeros for better visualization
            axes[0, 0].hist(sales_data, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Weekly Sales Distribution (Excluding Zeros)')
            axes[0, 0].set_xlabel('Sales Amount ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_yscale('log')  # Log scale for better visualization
            
        # Inventory distribution
        if 'onhand_invt_qty' in self.df.columns:
            inventory_data = self.df[self.df['onhand_invt_qty'] > 0]['onhand_invt_qty']
            axes[0, 1].hist(inventory_data, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Inventory Distribution (Excluding Zeros)')
            axes[0, 1].set_xlabel('Inventory Quantity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_yscale('log')
            
        # Sales vs Inventory scatter
        if all(col in self.df.columns for col in ['wkly_sales_amt', 'onhand_invt_qty']):
            sample_data = self.df[(self.df['wkly_sales_amt'] > 0) & (self.df['onhand_invt_qty'] > 0)].sample(
                min(5000, len(self.df)), random_state=42
            )
            axes[0, 2].scatter(sample_data['onhand_invt_qty'], sample_data['wkly_sales_amt'], 
                             alpha=0.5, s=10)
            axes[0, 2].set_title('Sales vs Inventory (Sample)')
            axes[0, 2].set_xlabel('Inventory Quantity')
            axes[0, 2].set_ylabel('Weekly Sales ($)')
            axes[0, 2].set_xscale('log')
            axes[0, 2].set_yscale('log')
            
        # Store count
        if self.store_col in self.df.columns:
            store_counts = self.df[self.store_col].value_counts().head(20)
            axes[1, 0].bar(range(len(store_counts)), store_counts.values)
            axes[1, 0].set_title('Records per Store (Top 20)')
            axes[1, 0].set_xlabel('Store Rank')
            axes[1, 0].set_ylabel('Number of Records')
            
        # Product count
        if self.product_col in self.df.columns:
            product_counts = self.df[self.product_col].value_counts().head(20)
            axes[1, 1].bar(range(len(product_counts)), product_counts.values)
            axes[1, 1].set_title('Records per Product (Top 20)')
            axes[1, 1].set_xlabel('Product Rank')
            axes[1, 1].set_ylabel('Number of Records')
            
        # Time series
        if all(col in self.df.columns for col in [self.time_col, 'wkly_sales_amt']):
            weekly_sales = self.df.groupby(self.time_col)['wkly_sales_amt'].sum().sort_index()
            axes[1, 2].plot(weekly_sales.index, weekly_sales.values, marker='o', linewidth=2)
            axes[1, 2].set_title('Total Weekly Sales Over Time')
            axes[1, 2].set_xlabel('Week Number')
            axes[1, 2].set_ylabel('Total Sales ($)')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()

    # ========================================
    # LEVEL 2: MEDIUM ANALYSIS
    # ========================================
    
    def level_2_medium_analysis(self):
        """Complete Level 2: Medium Analysis"""
        print("\n" + "="*60)
        print("LEVEL 2: MEDIUM-LEVEL ANALYSIS")
        print("="*60)
        
        self.medium_store_performance()
        self.medium_product_analysis()
        self.medium_temporal_analysis()
        self.medium_inventory_management()
        self.medium_efficiency_analysis()
        
        print("\n✅ LEVEL 2 MEDIUM ANALYSIS COMPLETED")
        
    def medium_store_performance(self):
        """2.1 Store Performance Analysis"""
        print("\n🏪 2.1 STORE PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if all(col in self.df.columns for col in [self.store_col, 'wkly_sales_amt']):
            # Store revenue analysis
            store_revenue = self.df.groupby(self.store_col).agg({
                'wkly_sales_amt': ['sum', 'mean', 'std', 'count'],
                'wkly_qty': ['sum', 'mean'] if 'wkly_qty' in self.df.columns else lambda x: None,
                'onhand_invt_qty': 'mean' if 'onhand_invt_qty' in self.df.columns else lambda x: None
            }).round(2)
            
            # Flatten column names
            store_revenue.columns = ['_'.join(col).strip() for col in store_revenue.columns]
            
            # Store performance ranking
            total_revenue = store_revenue.iloc[:, 0]  # First column should be sum of sales
			

	def medium_store_performance(self):
		"""2.1 Store Performance Analysis"""
		print("\n🏪 2.1 STORE PERFORMANCE ANALYSIS")
		print("-" * 40)
		
		if all(col in self.df.columns for col in [self.store_col, 'wkly_sales_amt']):
			# Store revenue analysis
			store_revenue = self.df.groupby(self.store_col).agg({
				'wkly_sales_amt': ['sum', 'mean', 'std', 'count'],
				'wkly_qty': ['sum', 'mean'] if 'wkly_qty' in self.df.columns else lambda x: None,
				'onhand_invt_qty': 'mean' if 'onhand_invt_qty' in self.df.columns else lambda x: None
			}).round(2)
			
			# Flatten column names
			store_revenue.columns = ['_'.join(col).strip() for col in store_revenue.columns]
			
			# Store performance ranking
			total_revenue = store_revenue.iloc[:, 0]  # First column should be sum of sales
			
			# THIS IS WHERE YOUR CODE STOPS - HERE'S THE REST:
			
			top_stores = total_revenue.nlargest(10)
			bottom_stores = total_revenue.nsmallest(5)
			
			print("Top 10 Revenue Generating Stores:")
			print(top_stores)
			
			print("\nBottom 5 Revenue Generating Stores:")
			print(bottom_stores)
			
			# Store consistency analysis
			avg_revenue = store_revenue.iloc[:, 1]  # Mean sales
			revenue_std = store_revenue.iloc[:, 2]   # Std sales
			
			# Calculate coefficient of variation for consistency
			cv = revenue_std / avg_revenue
			cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
			
			most_consistent = cv.nsmallest(5)
			least_consistent = cv.nlargest(5)
			
			print(f"\nMost Consistent Stores (low variability):")
			print(most_consistent.round(3))
			
			print(f"\nLeast Consistent Stores (high variability):")
			print(least_consistent.round(3))
			
			# Store performance visualization
			plt.figure(figsize=(15, 6))
			
			plt.subplot(1, 2, 1)
			plt.bar(range(len(top_stores)), top_stores.values)
			plt.title('Top 10 Stores by Revenue')
			plt.xlabel('Store Rank')
			plt.ylabel('Total Revenue ($)')
			plt.xticks(range(len(top_stores)), [f'Store {s}' for s in top_stores.index], rotation=45)
			
			plt.subplot(1, 2, 2)
			plt.scatter(avg_revenue, revenue_std, alpha=0.6)
			plt.title('Store Performance: Average vs Variability')
			plt.xlabel('Average Weekly Sales ($)')
			plt.ylabel('Sales Variability (Std Dev)')
			plt.grid(True, alpha=0.3)
			
			plt.tight_layout()
			plt.show()
			
			# Revenue concentration analysis (80/20 rule for stores)
			store_revenue_sorted = total_revenue.sort_values(ascending=False)
			cumulative_revenue = store_revenue_sorted.cumsum()
			total_network_revenue = store_revenue_sorted.sum()
			cumulative_percentage = cumulative_revenue / total_network_revenue * 100
			
			# Find stores contributing to 80% of revenue
			top_stores_80pct = cumulative_percentage <= 80
			num_top_stores = top_stores_80pct.sum()
			
			print(f"\nRevenue Concentration Analysis (80/20 Rule):")
			print(f"  Top {num_top_stores} stores generate 80% of revenue ({num_top_stores/len(total_revenue)*100:.1f}% of stores)")
			print(f"  Bottom {len(total_revenue) - num_top_stores} stores generate 20% of revenue ({(len(total_revenue) - num_top_stores)/len(total_revenue)*100:.1f}% of stores)")
			
			# Store efficiency analysis (if inventory data available)
			if 'onhand_invt_qty' in store_revenue.columns:
				avg_inventory = store_revenue['onhand_invt_qty_mean']
				inventory_efficiency = total_revenue / avg_inventory
				inventory_efficiency = inventory_efficiency.replace([np.inf, -np.inf], np.nan).dropna()
				
				print(f"\nInventory Efficiency Analysis:")
				print(f"Most Efficient Stores (highest sales per inventory unit):")
				print(inventory_efficiency.nlargest(5).round(3))
				
				print(f"\nLeast Efficient Stores (lowest sales per inventory unit):")
				print(inventory_efficiency.nsmallest(5).round(3))
				
			# Performance summary statistics
			print(f"\nStore Performance Summary:")
			print(f"  Number of stores analyzed: {len(total_revenue)}")
			print(f"  Average store revenue: ${total_revenue.mean():,.2f}")
			print(f"  Revenue range: ${total_revenue.min():,.2f} to ${total_revenue.max():,.2f}")
			print(f"  Revenue standard deviation: ${total_revenue.std():,.2f}")
			print(f"  Performance spread (max/min): {total_revenue.max()/total_revenue.min():.1f}x")
			
		else:
			print("Required columns not found for store performance analysis")
			print(f"Looking for: {self.store_col}, 'wkly_sales_amt'")
			print(f"Available columns: {list(self.df.columns)}")

# What your incomplete function was missing:
"""
MISSING COMPONENTS:
1. Top/bottom store identification and printing
2. Consistency analysis (coefficient of variation)
3. Store performance visualizations
4. 80/20 rule revenue concentration analysis
5. Inventory efficiency analysis
6. Performance summary statistics
7. Error handling for missing columns

YOUR CODE STOPPED AT: total_revenue = store_revenue.iloc[:, 0]
MISSING: Everything after that line!
"""