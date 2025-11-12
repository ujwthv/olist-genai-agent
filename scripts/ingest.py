import duckdb
import os

os.makedirs("data", exist_ok=True)
con = duckdb.connect("data/olist.duckdb")

base = "data/raw"

files = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv"
}

for name, fname in files.items():
    path = os.path.join(base, fname)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{path}')")

con.execute("""
CREATE OR REPLACE VIEW order_items_enriched AS
SELECT
    oi.*,
    p.product_category_name,
    ct.product_category_name_english
FROM order_items oi
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
""")

con.execute("""
CREATE OR REPLACE VIEW orders_enriched AS
SELECT
    o.*,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
""")
