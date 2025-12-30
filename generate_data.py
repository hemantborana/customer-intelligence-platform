import pandas as pd
import numpy as np
from faker import Faker
import json
import random
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

fake = Faker()
np.random.seed(42)
random.seed(42)

# how many records
n_customers = 5000
n_transactions = 25000

print(" Generating customer data...")

# 1. STRUCTURED DATA - Customer Database (SQL)
customers = []
for i in range(n_customers):
    signup_date = fake.date_between(start_date='-3y', end_date='today')
    customers.append({
        'customer_id': i + 1,
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'age': random.randint(18, 75),
        'gender': random.choice(['M', 'F', 'Other']),
        'city': fake.city(),
        'state': fake.state(),
        'signup_date': signup_date,
        'customer_segment': random.choice(['Premium', 'Regular', 'Occasional']),
        'is_active': random.choice([True, True, True, False])  # 75% active
    })

df_customers = pd.DataFrame(customers)
df_customers.to_csv('data/raw/customers.csv', index=False)
print(f" Created {n_customers} customers")

# 2. STRUCTURED DATA - Transaction History
transactions = []
products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch', 'Camera', 
            'TV', 'Speaker', 'Keyboard', 'Mouse', 'Monitor', 'Charger']
categories = ['Electronics', 'Accessories', 'Computing', 'Audio', 'Video']

for i in range(n_transactions):
    cust_id = random.randint(1, n_customers)
    trans_date = fake.date_between(start_date='-2y', end_date='today')
    amount = round(random.uniform(10, 2000), 2)
    
    transactions.append({
        'transaction_id': i + 1,
        'customer_id': cust_id,
        'transaction_date': trans_date,
        'product_name': random.choice(products),
        'category': random.choice(categories),
        'amount': amount,
        'quantity': random.randint(1, 5),
        'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash']),
        'channel': random.choice(['Online', 'Store', 'Mobile App'])
    })

df_transactions = pd.DataFrame(transactions)
df_transactions.to_csv('data/raw/transactions.csv', index=False)
print(f" Created {n_transactions} transactions")

# 3. SEMI-STRUCTURED - Web Analytics (JSON)
web_analytics = []
for i in range(3000):
    web_analytics.append({
        'session_id': fake.uuid4(),
        'customer_id': random.randint(1, n_customers),
        'timestamp': fake.date_time_between(start_date='-1y', end_date='now').isoformat(),
        'page_views': random.randint(1, 20),
        'time_on_site': random.randint(30, 3600),
        'bounce_rate': round(random.uniform(0, 1), 2),
        'device': random.choice(['Desktop', 'Mobile', 'Tablet']),
        'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
        'referrer': random.choice(['Google', 'Facebook', 'Direct', 'Email', 'Instagram'])
    })

with open('data/raw/web_analytics.json', 'w') as f:
    json.dump(web_analytics, f, indent=2)
print(" Created web analytics JSON")

# 4. SEMI-STRUCTURED - Mobile App Logs (CSV with complex structure)
app_logs = []
for i in range(2000):
    app_logs.append({
        'log_id': i + 1,
        'customer_id': random.randint(1, n_customers),
        'timestamp': fake.date_time_between(start_date='-6M', end_date='now'),
        'event_type': random.choice(['login', 'purchase', 'browse', 'search', 'logout', 'cart_add']),
        'screen_name': random.choice(['Home', 'Product', 'Cart', 'Checkout', 'Profile']),
        'duration_seconds': random.randint(5, 600),
        'app_version': random.choice(['1.0', '1.1', '1.2', '2.0']),
        'os': random.choice(['iOS', 'Android'])
    })

df_app_logs = pd.DataFrame(app_logs)
df_app_logs.to_csv('data/raw/app_logs.csv', index=False)
print(" Created app logs CSV")

# 5. SEMI-STRUCTURED - Email Campaigns (XML)
root = ET.Element('EmailCampaigns')
for i in range(500):
    campaign = ET.SubElement(root, 'Campaign')
    ET.SubElement(campaign, 'campaign_id').text = str(i + 1)
    ET.SubElement(campaign, 'customer_id').text = str(random.randint(1, n_customers))
    ET.SubElement(campaign, 'sent_date').text = str(fake.date_between(start_date='-1y', end_date='today'))
    ET.SubElement(campaign, 'subject').text = fake.sentence(nb_words=6)
    ET.SubElement(campaign, 'opened').text = str(random.choice([True, False]))
    ET.SubElement(campaign, 'clicked').text = str(random.choice([True, False, False]))
    ET.SubElement(campaign, 'campaign_type').text = random.choice(['Promotional', 'Newsletter', 'Transactional'])

tree = ET.ElementTree(root)
tree.write('data/raw/email_campaigns.xml', encoding='utf-8', xml_declaration=True)
print(" Created email campaigns XML")

# 6. UNSTRUCTURED - Customer Reviews (TEXT)
reviews = []
sentiments = ['positive', 'negative', 'neutral']
review_texts = {
    'positive': ['Great product!', 'Love it!', 'Excellent quality', 'Highly recommend', 'Amazing service'],
    'negative': ['Poor quality', 'Not worth it', 'Disappointed', 'Bad experience', 'Would not buy again'],
    'neutral': ['It is okay', 'Average product', 'As expected', 'Nothing special', 'Decent']
}

for i in range(1500):
    sentiment = random.choice(sentiments)
    reviews.append({
        'review_id': i + 1,
        'customer_id': random.randint(1, n_customers),
        'product_name': random.choice(products),
        'rating': random.randint(1, 5),
        'review_text': random.choice(review_texts[sentiment]) + ' ' + fake.text(max_nb_chars=200),
        'review_date': fake.date_between(start_date='-1y', end_date='today'),
        'sentiment': sentiment
    })

df_reviews = pd.DataFrame(reviews)
df_reviews.to_csv('data/raw/customer_reviews.csv', index=False)
print(" Created customer reviews")

# 7. UNSTRUCTURED - Social Media Mentions (TEXT)
social_media = []
for i in range(800):
    social_media.append({
        'mention_id': i + 1,
        'customer_id': random.randint(1, n_customers) if random.random() > 0.3 else None,
        'platform': random.choice(['Twitter', 'Facebook', 'Instagram', 'LinkedIn']),
        'post_text': fake.text(max_nb_chars=280),
        'likes': random.randint(0, 1000),
        'shares': random.randint(0, 100),
        'timestamp': fake.date_time_between(start_date='-6M', end_date='now'),
        'hashtags': ','.join([f'#{fake.word()}' for _ in range(random.randint(1, 4))])
    })

df_social = pd.DataFrame(social_media)
df_social.to_csv('data/raw/social_media.csv', index=False)
print(" Created social media mentions")

# 8. UNSTRUCTURED - Support Tickets (TEXT)
tickets = []
issues = ['Payment issue', 'Delivery delay', 'Product defect', 'Account access', 'Return request']
for i in range(600):
    tickets.append({
        'ticket_id': i + 1,
        'customer_id': random.randint(1, n_customers),
        'issue_type': random.choice(issues),
        'description': fake.paragraph(nb_sentences=3),
        'status': random.choice(['Open', 'In Progress', 'Resolved', 'Closed']),
        'priority': random.choice(['Low', 'Medium', 'High', 'Critical']),
        'created_date': fake.date_time_between(start_date='-1y', end_date='now'),
        'resolved_date': fake.date_time_between(start_date='-6M', end_date='now') if random.random() > 0.3 else None
    })

df_tickets = pd.DataFrame(tickets)
df_tickets.to_csv('data/raw/support_tickets.csv', index=False)
print(" Created support tickets")

print("\n ALL DATA GENERATED SUCCESSFULLY!")
print("\nData Summary:")
print(f"- Customers: {len(df_customers)}")
print(f"- Transactions: {len(df_transactions)}")
print(f"- Web Analytics: {len(web_analytics)} sessions")
print(f"- App Logs: {len(df_app_logs)} events")
print(f"- Email Campaigns: 500 campaigns")
print(f"- Reviews: {len(df_reviews)}")
print(f"- Social Media: {len(df_social)} mentions")
print(f"- Support Tickets: {len(df_tickets)}")