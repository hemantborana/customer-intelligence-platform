import sqlite3
import pandas as pd
import os

# create database connection
db_path = 'database/customer_intelligence.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print(" Setting up database schema...")

# drop tables if exist (clean slate)
cursor.execute("DROP TABLE IF EXISTS customers")
cursor.execute("DROP TABLE IF EXISTS transactions")
cursor.execute("DROP TABLE IF EXISTS products")
cursor.execute("DROP TABLE IF EXISTS customer_segments")

# create normalized tables (3NF)

# dimension table - customer segments
cursor.execute('''
CREATE TABLE customer_segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_name TEXT UNIQUE NOT NULL,
    description TEXT
)
''')

# insert segment data
segments = [
    ('Premium', 'High-value customers with frequent purchases'),
    ('Regular', 'Average customers with moderate activity'),
    ('Occasional', 'Low-frequency customers')
]
cursor.executemany('INSERT INTO customer_segments (segment_name, description) VALUES (?, ?)', segments)

# main customers table
cursor.execute('''
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    age INTEGER,
    gender TEXT CHECK(gender IN ('M', 'F', 'Other')),
    city TEXT,
    state TEXT,
    signup_date DATE NOT NULL,
    segment_id INTEGER,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (segment_id) REFERENCES customer_segments(segment_id)
)
''')

# create indexes for performance
cursor.execute('CREATE INDEX idx_customer_email ON customers(email)')
cursor.execute('CREATE INDEX idx_customer_segment ON customers(segment_id)')
cursor.execute('CREATE INDEX idx_customer_signup ON customers(signup_date)')

# product catalog table
cursor.execute('''
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# transactions fact table
cursor.execute('''
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER,
    transaction_date DATE NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    quantity INTEGER DEFAULT 1,
    payment_method TEXT,
    channel TEXT CHECK(channel IN ('Online', 'Store', 'Mobile App')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
''')

# indexes for transaction queries
cursor.execute('CREATE INDEX idx_trans_customer ON transactions(customer_id)')
cursor.execute('CREATE INDEX idx_trans_date ON transactions(transaction_date)')
cursor.execute('CREATE INDEX idx_trans_product ON transactions(product_id)')

# web analytics table
cursor.execute('''
CREATE TABLE web_analytics (
    session_id TEXT PRIMARY KEY,
    customer_id INTEGER,
    timestamp DATETIME NOT NULL,
    page_views INTEGER,
    time_on_site INTEGER,
    bounce_rate DECIMAL(3,2),
    device TEXT,
    browser TEXT,
    referrer TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
''')

cursor.execute('CREATE INDEX idx_web_customer ON web_analytics(customer_id)')
cursor.execute('CREATE INDEX idx_web_timestamp ON web_analytics(timestamp)')

# app logs table
cursor.execute('''
CREATE TABLE app_logs (
    log_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    timestamp DATETIME NOT NULL,
    event_type TEXT NOT NULL,
    screen_name TEXT,
    duration_seconds INTEGER,
    app_version TEXT,
    os TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
''')

cursor.execute('CREATE INDEX idx_app_customer ON app_logs(customer_id)')
cursor.execute('CREATE INDEX idx_app_event ON app_logs(event_type)')

# email campaigns table
cursor.execute('''
CREATE TABLE email_campaigns (
    campaign_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    sent_date DATE NOT NULL,
    subject TEXT,
    opened BOOLEAN DEFAULT 0,
    clicked BOOLEAN DEFAULT 0,
    campaign_type TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
''')

cursor.execute('CREATE INDEX idx_email_customer ON email_campaigns(customer_id)')
cursor.execute('CREATE INDEX idx_email_date ON email_campaigns(sent_date)')

# reviews table
cursor.execute('''
CREATE TABLE customer_reviews (
    review_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_name TEXT NOT NULL,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    review_text TEXT,
    review_date DATE NOT NULL,
    sentiment TEXT CHECK(sentiment IN ('positive', 'negative', 'neutral')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
''')

cursor.execute('CREATE INDEX idx_review_customer ON customer_reviews(customer_id)')
cursor.execute('CREATE INDEX idx_review_sentiment ON customer_reviews(sentiment)')

# support tickets table
cursor.execute('''
CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    issue_type TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
    priority TEXT CHECK(priority IN ('Low', 'Medium', 'High', 'Critical')),
    created_date DATETIME NOT NULL,
    resolved_date DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
''')

cursor.execute('CREATE INDEX idx_ticket_customer ON support_tickets(customer_id)')
cursor.execute('CREATE INDEX idx_ticket_status ON support_tickets(status)')

# data quality tracking table
cursor.execute('''
CREATE TABLE data_quality_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_records INTEGER,
    null_count INTEGER,
    duplicate_count INTEGER,
    quality_score DECIMAL(5,2),
    issues TEXT
)
''')

# ETL metadata table
cursor.execute('''
CREATE TABLE etl_metadata (
    etl_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    load_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    records_processed INTEGER,
    records_loaded INTEGER,
    records_failed INTEGER,
    status TEXT CHECK(status IN ('Success', 'Failed', 'Partial')),
    error_message TEXT
)
''')

conn.commit()
print(" Database schema created successfully!")

# show all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\n Created {len(tables)} tables:")
for table in tables:
    print(f"  - {table[0]}")

conn.close()
print(f"\n Database saved at: {db_path}")