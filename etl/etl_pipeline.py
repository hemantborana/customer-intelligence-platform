import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import sqlite3
from datetime import datetime
import os

class ETLPipeline:
    def __init__(self, db_path='database/customer_intelligence.db'):
        self.db_path = db_path
        self.conn = None
        self.errors = []
        
    def connect_db(self):
        """establish db connection"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def log_etl_metadata(self, source_name, processed, loaded, failed, status, error_msg=None):
        """track etl runs for monitoring"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO etl_metadata (source_name, records_processed, records_loaded, 
                                     records_failed, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (source_name, processed, loaded, failed, status, error_msg))
        self.conn.commit()
    
    def check_data_quality(self, df, table_name):
        """perform quality checks"""
        total_records = len(df)
        null_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        
        # calculate quality score (simple metric)
        quality_score = 100 - ((null_count + duplicate_count) / (total_records * len(df.columns)) * 100)
        quality_score = max(0, min(100, quality_score))
        
        issues = []
        if null_count > 0:
            issues.append(f"{null_count} null values found")
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicates found")
            
        # log quality metrics
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO data_quality_log (table_name, total_records, null_count, 
                                         duplicate_count, quality_score, issues)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (table_name, total_records, null_count, duplicate_count, 
              quality_score, '; '.join(issues) if issues else 'No issues'))
        self.conn.commit()
        
        return quality_score, issues
    
    def extract_csv(self, filepath):
        """extract data from csv files"""
        try:
            df = pd.read_csv(filepath)
            print(f" Extracted {len(df)} records from {os.path.basename(filepath)}")
            return df
        except Exception as e:
            self.errors.append(f"CSV extraction error: {str(e)}")
            return None
    
    def extract_json(self, filepath):
        """extract semi-structured json data"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f" Extracted {len(df)} records from {os.path.basename(filepath)}")
            return df
        except Exception as e:
            self.errors.append(f"JSON extraction error: {str(e)}")
            return None
    
    def extract_xml(self, filepath):
        """extract semi-structured xml data"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            records = []
            for campaign in root.findall('Campaign'):
                record = {}
                for child in campaign:
                    record[child.tag] = child.text
                records.append(record)
            
            df = pd.DataFrame(records)
            print(f" Extracted {len(df)} records from {os.path.basename(filepath)}")
            return df
        except Exception as e:
            self.errors.append(f"XML extraction error: {str(e)}")
            return None
    
    def transform_customers(self, df):
        """transform and clean customer data"""
        # remove duplicates
        df = df.drop_duplicates(subset=['email'])
        
        # handle missing values
        df['phone'] = df['phone'].fillna('Unknown')
        df['age'] = df['age'].fillna(df['age'].median())
        
        # map segment to segment_id
        segment_map = {'Premium': 1, 'Regular': 2, 'Occasional': 3}
        df['segment_id'] = df['customer_segment'].map(segment_map)
        df = df.drop('customer_segment', axis=1)
        
        # ensure proper data types
        df['is_active'] = df['is_active'].astype(int)
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        
        return df
    
    def transform_transactions(self, df):
        """transform transaction data"""
        # remove duplicates
        df = df.drop_duplicates(subset=['transaction_id'])
        
        # handle outliers using IQR method
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['amount'] >= Q1 - 1.5*IQR) & (df['amount'] <= Q3 + 1.5*IQR)]
        
        # ensure proper types
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        return df
    
    def load_to_db(self, df, table_name, if_exists='append'):
        """load transformed data to database"""
        try:
            # for customers, need to handle foreign key separately
            if table_name == 'customers':
                df_load = df[['customer_id', 'name', 'email', 'phone', 'age', 
                             'gender', 'city', 'state', 'signup_date', 'segment_id', 'is_active']]
            else:
                df_load = df
            
            df_load.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            print(f" Loaded {len(df_load)} records to {table_name}")
            return len(df_load)
        except Exception as e:
            self.errors.append(f"Load error for {table_name}: {str(e)}")
            return 0
    
    def run_full_pipeline(self):
        """execute complete etl pipeline"""
        print(" Starting ETL Pipeline...\n")
        self.connect_db()
        
        # 1. CUSTOMERS (structured csv)
        print("=" * 50)
        print("Processing CUSTOMERS...")
        df_customers = self.extract_csv('data/raw/customers.csv')
        if df_customers is not None:
            df_customers = self.transform_customers(df_customers)
            quality_score, issues = self.check_data_quality(df_customers, 'customers')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_customers, 'customers', if_exists='replace')
            self.log_etl_metadata('customers', len(df_customers), loaded, 
                                 len(df_customers)-loaded, 'Success')
        
        # 2. PRODUCTS (extract from transactions)
        print("\n" + "=" * 50)
        print("Processing PRODUCTS...")
        df_trans_raw = self.extract_csv('data/raw/transactions.csv')
        if df_trans_raw is not None:
            df_products = df_trans_raw[['product_name', 'category']].drop_duplicates()
            df_products.insert(0, 'product_id', range(1, len(df_products) + 1))
            loaded = self.load_to_db(df_products, 'products', if_exists='replace')
            print(f" Created {loaded} unique products")
        
        # 3. TRANSACTIONS (structured csv with foreign keys)
        print("\n" + "=" * 50)
        print("Processing TRANSACTIONS...")
        if df_trans_raw is not None:
            df_transactions = self.transform_transactions(df_trans_raw)
            
            # map product names to product_id
            product_map = dict(zip(df_products['product_name'], df_products['product_id']))
            df_transactions['product_id'] = df_transactions['product_name'].map(product_map)
            df_transactions = df_transactions.drop(['product_name', 'category'], axis=1)
            
            quality_score, issues = self.check_data_quality(df_transactions, 'transactions')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_transactions, 'transactions', if_exists='replace')
            self.log_etl_metadata('transactions', len(df_transactions), loaded,
                                 len(df_transactions)-loaded, 'Success')
        
        # 4. WEB ANALYTICS (semi-structured json)
        print("\n" + "=" * 50)
        print("Processing WEB ANALYTICS...")
        df_web = self.extract_json('data/raw/web_analytics.json')
        if df_web is not None:
            df_web['timestamp'] = pd.to_datetime(df_web['timestamp'])
            quality_score, issues = self.check_data_quality(df_web, 'web_analytics')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_web, 'web_analytics', if_exists='replace')
            self.log_etl_metadata('web_analytics', len(df_web), loaded,
                                 len(df_web)-loaded, 'Success')
        
        # 5. APP LOGS (semi-structured csv)
        print("\n" + "=" * 50)
        print("Processing APP LOGS...")
        df_app = self.extract_csv('data/raw/app_logs.csv')
        if df_app is not None:
            df_app['timestamp'] = pd.to_datetime(df_app['timestamp'])
            quality_score, issues = self.check_data_quality(df_app, 'app_logs')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_app, 'app_logs', if_exists='replace')
            self.log_etl_metadata('app_logs', len(df_app), loaded,
                                 len(df_app)-loaded, 'Success')
        
        # 6. EMAIL CAMPAIGNS (semi-structured xml)
        print("\n" + "=" * 50)
        print("Processing EMAIL CAMPAIGNS...")
        df_email = self.extract_xml('data/raw/email_campaigns.xml')
        if df_email is not None:
            # convert string booleans to int
            df_email['opened'] = df_email['opened'].map({'True': 1, 'False': 0})
            df_email['clicked'] = df_email['clicked'].map({'True': 1, 'False': 0})
            df_email['sent_date'] = pd.to_datetime(df_email['sent_date'])
            df_email['campaign_id'] = df_email['campaign_id'].astype(int)
            df_email['customer_id'] = df_email['customer_id'].astype(int)
            
            quality_score, issues = self.check_data_quality(df_email, 'email_campaigns')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_email, 'email_campaigns', if_exists='replace')
            self.log_etl_metadata('email_campaigns', len(df_email), loaded,
                                 len(df_email)-loaded, 'Success')
        
        # 7. CUSTOMER REVIEWS (unstructured text)
        print("\n" + "=" * 50)
        print("Processing CUSTOMER REVIEWS...")
        df_reviews = self.extract_csv('data/raw/customer_reviews.csv')
        if df_reviews is not None:
            df_reviews['review_date'] = pd.to_datetime(df_reviews['review_date'])
            quality_score, issues = self.check_data_quality(df_reviews, 'customer_reviews')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_reviews, 'customer_reviews', if_exists='replace')
            self.log_etl_metadata('customer_reviews', len(df_reviews), loaded,
                                 len(df_reviews)-loaded, 'Success')
        
        # 8. SUPPORT TICKETS (unstructured text)
        print("\n" + "=" * 50)
        print("Processing SUPPORT TICKETS...")
        df_tickets = self.extract_csv('data/raw/support_tickets.csv')
        if df_tickets is not None:
            df_tickets['created_date'] = pd.to_datetime(df_tickets['created_date'])
            df_tickets['resolved_date'] = pd.to_datetime(df_tickets['resolved_date'], errors='coerce')
            quality_score, issues = self.check_data_quality(df_tickets, 'support_tickets')
            print(f" Data Quality Score: {quality_score:.2f}%")
            loaded = self.load_to_db(df_tickets, 'support_tickets', if_exists='replace')
            self.log_etl_metadata('support_tickets', len(df_tickets), loaded,
                                 len(df_tickets)-loaded, 'Success')
        
        print("\n" + "=" * 50)
        print(" ETL Pipeline Completed!")
        
        if self.errors:
            print(f"\n Encountered {len(self.errors)} errors:")
            for error in self.errors:
                print(f"  - {error}")
        
        self.conn.close()

# run the pipeline
if __name__ == "__main__":
    etl = ETLPipeline()
    etl.run_full_pipeline()