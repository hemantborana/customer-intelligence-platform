import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class EDAAnalysis:
    def __init__(self, db_path='database/customer_intelligence.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.results = {}
        
    def load_data(self):
        """load main datasets for analysis"""
        print(" Loading data from database...\n")
        
        # customer data with segment info
        self.df_customers = pd.read_sql('''
            SELECT c.*, cs.segment_name 
            FROM customers c
            LEFT JOIN customer_segments cs ON c.segment_id = cs.segment_id
        ''', self.conn)
        
        # transaction data
        self.df_transactions = pd.read_sql('''
            SELECT t.*, p.product_name, p.category
            FROM transactions t
            LEFT JOIN products p ON t.product_id = p.product_id
        ''', self.conn)
        
        # reviews
        self.df_reviews = pd.read_sql('SELECT * FROM customer_reviews', self.conn)
        
        # convert dates
        self.df_customers['signup_date'] = pd.to_datetime(self.df_customers['signup_date'])
        self.df_transactions['transaction_date'] = pd.to_datetime(self.df_transactions['transaction_date'])
        
        print(f" Loaded {len(self.df_customers)} customers")
        print(f" Loaded {len(self.df_transactions)} transactions")
        print(f" Loaded {len(self.df_reviews)} reviews\n")
    
    def missing_data_analysis(self):
        """analyze missing data patterns"""
        print("=" * 60)
        print("1. MISSING DATA ANALYSIS")
        print("=" * 60)
        
        missing_stats = {}
        
        for name, df in [('customers', self.df_customers), 
                        ('transactions', self.df_transactions),
                        ('reviews', self.df_reviews)]:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            
            if missing.sum() > 0:
                print(f"\n{name.upper()}:")
                for col in missing[missing > 0].index:
                    print(f"  - {col}: {missing[col]} ({missing_pct[col]}%)")
                    
                missing_stats[name] = {
                    'total_missing': missing.sum(),
                    'columns_affected': len(missing[missing > 0])
                }
            else:
                print(f"\n{name.upper()}: No missing values ")
                missing_stats[name] = {'total_missing': 0, 'columns_affected': 0}
        
        self.results['missing_data'] = missing_stats
        return missing_stats
    
    def outlier_detection(self):
        """detect outliers using multiple methods"""
        print("\n" + "=" * 60)
        print("2. OUTLIER DETECTION")
        print("=" * 60)
        
        # focus on transaction amounts
        amounts = self.df_transactions['amount'].values
        
        # method 1: IQR method
        Q1 = np.percentile(amounts, 25)
        Q3 = np.percentile(amounts, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((amounts < lower_bound) | (amounts > upper_bound)).sum()
        
        # method 2: Z-score method
        z_scores = np.abs(zscore(amounts))
        zscore_outliers = (z_scores > 3).sum()
        
        # method 3: Percentile method
        p1 = np.percentile(amounts, 1)
        p99 = np.percentile(amounts, 99)
        percentile_outliers = ((amounts < p1) | (amounts > p99)).sum()
        
        print(f"\nTransaction Amount Outliers:")
        print(f"  - IQR Method: {iqr_outliers} outliers ({iqr_outliers/len(amounts)*100:.2f}%)")
        print(f"  - Z-Score Method (>3σ): {zscore_outliers} outliers ({zscore_outliers/len(amounts)*100:.2f}%)")
        print(f"  - Percentile Method (1-99%): {percentile_outliers} outliers ({percentile_outliers/len(amounts)*100:.2f}%)")
        print(f"\nAmount Statistics:")
        print(f"  - Mean: ${amounts.mean():.2f}")
        print(f"  - Median: ${np.median(amounts):.2f}")
        print(f"  - Std Dev: ${amounts.std():.2f}")
        print(f"  - Range: ${amounts.min():.2f} - ${amounts.max():.2f}")
        
        self.results['outliers'] = {
            'iqr_outliers': int(iqr_outliers),
            'zscore_outliers': int(zscore_outliers),
            'percentile_outliers': int(percentile_outliers)
        }
    
    def trend_seasonality_analysis(self):
        """analyze trends and seasonality"""
        print("\n" + "=" * 60)
        print("3. TREND & SEASONALITY ANALYSIS")
        print("=" * 60)
        
        # aggregate transactions by month
        self.df_transactions['year_month'] = self.df_transactions['transaction_date'].dt.to_period('M')
        monthly_sales = self.df_transactions.groupby('year_month').agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        monthly_sales.columns = ['year_month', 'total_sales', 'transaction_count']
        monthly_sales['year_month'] = monthly_sales['year_month'].astype(str)
        
        # calculate month over month growth
        monthly_sales['sales_growth'] = monthly_sales['total_sales'].pct_change() * 100
        
        print(f"\nMonthly Sales Trends:")
        print(f"  - Total Months: {len(monthly_sales)}")
        print(f"  - Average Monthly Sales: ${monthly_sales['total_sales'].mean():.2f}")
        print(f"  - Average MoM Growth: {monthly_sales['sales_growth'].mean():.2f}%")
        print(f"  - Best Month: {monthly_sales.loc[monthly_sales['total_sales'].idxmax(), 'year_month']} (${monthly_sales['total_sales'].max():.2f})")
        print(f"  - Worst Month: {monthly_sales.loc[monthly_sales['total_sales'].idxmin(), 'year_month']} (${monthly_sales['total_sales'].min():.2f})")
        
        # day of week analysis
        self.df_transactions['day_of_week'] = self.df_transactions['transaction_date'].dt.day_name()
        dow_sales = self.df_transactions.groupby('day_of_week')['amount'].agg(['sum', 'mean', 'count'])
        
        print(f"\nDay of Week Patterns:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            if day in dow_sales.index:
                print(f"  - {day}: {dow_sales.loc[day, 'count']} transactions, Avg ${dow_sales.loc[day, 'mean']:.2f}")
        
        self.results['trends'] = {
            'monthly_sales': monthly_sales.to_dict('records'),
            'avg_monthly_sales': float(monthly_sales['total_sales'].mean()),
            'avg_growth': float(monthly_sales['sales_growth'].mean())
        }
    
    def correlation_analysis(self):
        """correlation analysis with spurious detection"""
        print("\n" + "=" * 60)
        print("4. CORRELATION ANALYSIS")
        print("=" * 60)
        
        # customer lifetime value calculation
        customer_metrics = self.df_transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        customer_metrics.columns = ['customer_id', 'total_spent', 'avg_transaction', 
                                    'purchase_count', 'first_purchase', 'last_purchase']
        
        # merge with customer data
        analysis_df = self.df_customers.merge(customer_metrics, on='customer_id', how='left')
        analysis_df['total_spent'] = analysis_df['total_spent'].fillna(0)
        analysis_df['purchase_count'] = analysis_df['purchase_count'].fillna(0)
        
        # select numeric columns
        numeric_cols = ['age', 'total_spent', 'avg_transaction', 'purchase_count']
        corr_matrix = analysis_df[numeric_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # identify strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    print(f"  - {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        self.results['correlations'] = corr_matrix.to_dict()
    
    def pattern_discovery(self):
        """association rule mining and pattern discovery"""
        print("\n" + "=" * 60)
        print("5. PATTERN DISCOVERY")
        print("=" * 60)
        
        # product category patterns
        category_analysis = self.df_transactions.groupby(['customer_id', 'category']).size().reset_index(name='count')
        top_categories = self.df_transactions['category'].value_counts()
        
        print("\nProduct Category Distribution:")
        for cat, count in top_categories.items():
            pct = count / len(self.df_transactions) * 100
            print(f"  - {cat}: {count} transactions ({pct:.1f}%)")
        
        # channel preference analysis
        channel_pref = self.df_transactions.groupby('channel').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean']
        }).round(2)
        
        print("\nChannel Performance:")
        for channel in channel_pref.index:
            count = channel_pref.loc[channel, ('transaction_id', 'count')]
            total = channel_pref.loc[channel, ('amount', 'sum')]
            avg = channel_pref.loc[channel, ('amount', 'mean')]
            print(f"  - {channel}: {count} trans, ${total:.2f} total, ${avg:.2f} avg")
        
        # payment method patterns
        payment_dist = self.df_transactions['payment_method'].value_counts()
        print("\nPayment Method Preferences:")
        for method, count in payment_dist.items():
            pct = count / len(self.df_transactions) * 100
            print(f"  - {method}: {count} ({pct:.1f}%)")
        
        self.results['patterns'] = {
            'top_categories': top_categories.to_dict(),
            'channel_performance': channel_pref.to_dict(),
            'payment_methods': payment_dist.to_dict()
        }
    
    def winsorization_analysis(self):
        """winsorization and percentile analysis"""
        print("\n" + "=" * 60)
        print("6. WINSORIZATION & PERCENTILE ANALYSIS")
        print("=" * 60)
        
        amounts = self.df_transactions['amount'].values
        
        # calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("\nTransaction Amount Percentiles:")
        for p in percentiles:
            val = np.percentile(amounts, p)
            print(f"  - {p}th percentile: ${val:.2f}")
        
        # winsorize at 5% and 95%
        p5 = np.percentile(amounts, 5)
        p95 = np.percentile(amounts, 95)
        winsorized = np.clip(amounts, p5, p95)
        
        print(f"\nWinsorization (5%-95%):")
        print(f"  - Original Mean: ${amounts.mean():.2f}")
        print(f"  - Winsorized Mean: ${winsorized.mean():.2f}")
        print(f"  - Original Std: ${amounts.std():.2f}")
        print(f"  - Winsorized Std: ${winsorized.std():.2f}")
        print(f"  - Values clipped: {((amounts < p5) | (amounts > p95)).sum()}")
        
        self.results['winsorization'] = {
            'original_mean': float(amounts.mean()),
            'winsorized_mean': float(winsorized.mean()),
            'values_clipped': int(((amounts < p5) | (amounts > p95)).sum())
        }
    
    def sentiment_analysis(self):
        """analyze review sentiment patterns"""
        print("\n" + "=" * 60)
        print("7. SENTIMENT ANALYSIS (Reviews)")
        print("=" * 60)
        
        sentiment_dist = self.df_reviews['sentiment'].value_counts()
        sentiment_by_rating = self.df_reviews.groupby(['rating', 'sentiment']).size().reset_index(name='count')
        
        print("\nOverall Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            pct = count / len(self.df_reviews) * 100
            print(f"  - {sentiment.capitalize()}: {count} ({pct:.1f}%)")
        
        print("\nAverage Rating by Sentiment:")
        avg_rating = self.df_reviews.groupby('sentiment')['rating'].mean()
        for sentiment, rating in avg_rating.items():
            print(f"  - {sentiment.capitalize()}: {rating:.2f} stars")
        
        self.results['sentiment'] = {
            'distribution': sentiment_dist.to_dict(),
            'avg_rating_by_sentiment': avg_rating.to_dict()
        }
    
    def generate_eda_report(self):
        """run all eda analyses and generate report"""
        print("\n" + "🔬" * 30)
        print("COMPREHENSIVE EDA ANALYSIS")
        print("🔬" * 30 + "\n")
        
        self.load_data()
        self.missing_data_analysis()
        self.outlier_detection()
        self.trend_seasonality_analysis()
        self.correlation_analysis()
        self.pattern_discovery()
        self.winsorization_analysis()
        self.sentiment_analysis()
        
        print("\n" + "=" * 60)
        print(" EDA ANALYSIS COMPLETED!")
        print("=" * 60)
        
        # save results
        import json
        
        def make_serializable(obj):
            """recursively convert all dict keys to strings"""
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        with open('analytics/eda_results.json', 'w') as f:
            results_clean = make_serializable(self.results)
            json.dump(results_clean, f, indent=2, default=str)
        
        print("\n Results saved to: analytics/eda_results.json")
        
        self.conn.close()
        return self.results

if __name__ == "__main__":
    eda = EDAAnalysis()
    results = eda.generate_eda_report()