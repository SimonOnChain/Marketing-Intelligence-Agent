"""Documented schema for the Olist dataset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Table:
    file: str
    key: str
    columns: tuple[str, ...]


TABLES: dict[str, Table] = {
    "customers": Table(
        file="olist_customers_dataset.csv",
        key="customer_id",
        columns=(
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ),
    ),
    "orders": Table(
        file="olist_orders_dataset.csv",
        key="order_id",
        columns=(
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ),
    ),
    "order_items": Table(
        file="olist_order_items_dataset.csv",
        key="order_item_id",
        columns=(
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        ),
    ),
    "order_payments": Table(
        file="olist_order_payments_dataset.csv",
        key="order_id",
        columns=(
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_installments",
            "payment_value",
        ),
    ),
    "order_reviews": Table(
        file="olist_order_reviews_dataset.csv",
        key="review_id",
        columns=(
            "review_id",
            "order_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        ),
    ),
    "products": Table(
        file="olist_products_dataset.csv",
        key="product_id",
        columns=(
            "product_id",
            "product_category_name",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ),
    ),
    "sellers": Table(
        file="olist_sellers_dataset.csv",
        key="seller_id",
        columns=(
            "seller_id",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ),
    ),
    "geolocation": Table(
        file="olist_geolocation_dataset.csv",
        key="geolocation_zip_code_prefix",
        columns=(
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ),
    ),
    "category_translation": Table(
        file="product_category_name_translation.csv",
        key="product_category_name",
        columns=("product_category_name", "product_category_name_english"),
    ),
}

