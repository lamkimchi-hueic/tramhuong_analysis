"""
AI Analysis Service — Main FastAPI Application
Provides data analysis endpoints for the Tram Huong e-commerce admin panel.

Endpoints:
  GET /analysis/overview          — KPI tổng quan (tổng doanh thu, đơn hàng, sản phẩm, khách hàng)
  GET /analysis/revenue-trend     — Doanh thu theo tháng (12 tháng gần nhất)
  GET /analysis/top-products      — Top sản phẩm bán chạy nhất
  GET /analysis/category-share    — Tỷ trọng doanh thu theo danh mục
  GET /analysis/order-status      — Phân bổ trạng thái đơn hàng
  GET /analysis/customer-insights — Phân tích khách hàng (top spenders, mới vs cũ)
  GET /analysis/forecast          — Dự báo doanh thu tháng tới (Linear Regression)
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from sklearn.linear_model import LinearRegression

from database import query

load_dotenv()

app = FastAPI(
    title="Trầm Hương AI Analysis",
    description="Dịch vụ phân tích dữ liệu cho hệ thống quản trị Trầm Hương",
    version="1.0.0",
)

# ==================== CORS ====================
# Cho phép frontend (Vite dev server + Vercel) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        "https://tramhuong-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== HELPER ====================
def safe_float(val):
    """Convert Decimal / None to float safely."""
    if val is None:
        return 0.0
    return float(val)


# ==================== ENDPOINTS ====================


@app.get("/")
def root():
    return {"status": "ok", "service": "Trầm Hương AI Analysis", "version": "1.0.0"}


# ---------- 1. KPI Overview ----------
@app.get("/analysis/overview")
def overview():
    """Trả về các chỉ số KPI tổng quan."""

    revenue_row = query("""
        SELECT COALESCE(SUM(total_amount), 0) as total_revenue,
               COUNT(*) as completed_orders
        FROM orders
        WHERE status = 'Completed' AND "deletedAt" IS NULL
    """)

    total_orders = query("""
        SELECT COUNT(*) as count FROM orders WHERE "deletedAt" IS NULL
    """)

    total_products = query("""
        SELECT COUNT(*) as count FROM "Products" WHERE "deletedAt" IS NULL
    """)

    total_customers = query("""
        SELECT COUNT(*) as count FROM "Users" WHERE role = 'Customer' AND "deletedAt" IS NULL
    """)

    # Đơn hàng tháng này vs tháng trước
    current_month_orders = query("""
        SELECT COUNT(*) as count, COALESCE(SUM(total_amount), 0) as revenue
        FROM orders
        WHERE "deletedAt" IS NULL
          AND status = 'Completed'
          AND "createdAt" >= date_trunc('month', CURRENT_DATE)
    """)

    last_month_orders = query("""
        SELECT COUNT(*) as count, COALESCE(SUM(total_amount), 0) as revenue
        FROM orders
        WHERE "deletedAt" IS NULL
          AND status = 'Completed'
          AND "createdAt" >= date_trunc('month', CURRENT_DATE) - interval '1 month'
          AND "createdAt" < date_trunc('month', CURRENT_DATE)
    """)

    curr_rev = safe_float(current_month_orders[0]["revenue"])
    last_rev = safe_float(last_month_orders[0]["revenue"])
    growth = ((curr_rev - last_rev) / last_rev * 100) if last_rev > 0 else 0

    return {
        "total_revenue": safe_float(revenue_row[0]["total_revenue"]),
        "completed_orders": revenue_row[0]["completed_orders"],
        "total_orders": total_orders[0]["count"],
        "total_products": total_products[0]["count"],
        "total_customers": total_customers[0]["count"],
        "current_month_revenue": curr_rev,
        "last_month_revenue": last_rev,
        "revenue_growth_percent": round(growth, 1),
    }


# ---------- 2. Revenue Trend (12 months) ----------
@app.get("/analysis/revenue-trend")
def revenue_trend():
    """Doanh thu theo tháng (12 tháng gần nhất)."""
    rows = query("""
        SELECT
            TO_CHAR(date_trunc('month', "createdAt"), 'YYYY-MM') as month,
            COALESCE(SUM(total_amount), 0) as revenue,
            COUNT(*) as orders
        FROM orders
        WHERE "deletedAt" IS NULL
          AND status = 'Completed'
          AND "createdAt" >= CURRENT_DATE - interval '12 months'
        GROUP BY date_trunc('month', "createdAt")
        ORDER BY date_trunc('month', "createdAt")
    """)

    return {
        "labels": [r["month"] for r in rows],
        "revenue": [safe_float(r["revenue"]) for r in rows],
        "orders": [r["orders"] for r in rows],
    }


# ---------- 3. Top Products ----------
@app.get("/analysis/top-products")
def top_products(limit: int = 10):
    """Top sản phẩm bán chạy nhất (theo số lượng và doanh thu)."""
    rows = query("""
        SELECT
            p.id_product,
            p.product_name,
            COALESCE(SUM(po.product_quantity), 0) as total_sold,
            COALESCE(SUM(po.product_quantity * po.product_price), 0) as total_revenue
        FROM products_orders po
        JOIN "Products" p ON p.id_product = po.id_product
        JOIN orders o ON o.id_order = po.id_order
        WHERE o."deletedAt" IS NULL AND o.status = 'Completed'
        GROUP BY p.id_product, p.product_name
        ORDER BY total_sold DESC
        LIMIT %s
    """, (limit,))

    return [
        {
            "id_product": r["id_product"],
            "product_name": r["product_name"],
            "total_sold": r["total_sold"],
            "total_revenue": safe_float(r["total_revenue"]),
        }
        for r in rows
    ]


# ---------- 4. Category Revenue Share ----------
@app.get("/analysis/category-share")
def category_share():
    """Tỷ trọng doanh thu theo danh mục sản phẩm."""
    rows = query("""
        SELECT
            c.category_name,
            COALESCE(SUM(po.product_quantity * po.product_price), 0) as revenue
        FROM products_orders po
        JOIN "Products" p ON p.id_product = po.id_product
        JOIN categories c ON c.id_category = p.id_category
        JOIN orders o ON o.id_order = po.id_order
        WHERE o."deletedAt" IS NULL AND o.status = 'Completed'
          AND c."deletedAt" IS NULL
        GROUP BY c.category_name
        ORDER BY revenue DESC
    """)

    total = sum(safe_float(r["revenue"]) for r in rows) or 1

    return [
        {
            "category_name": r["category_name"],
            "revenue": safe_float(r["revenue"]),
            "percent": round(safe_float(r["revenue"]) / total * 100, 1),
        }
        for r in rows
    ]


# ---------- 5. Order Status Distribution ----------
@app.get("/analysis/order-status")
def order_status():
    """Phân bổ trạng thái đơn hàng."""
    rows = query("""
        SELECT status, COUNT(*) as count
        FROM orders
        WHERE "deletedAt" IS NULL
        GROUP BY status
        ORDER BY count DESC
    """)

    status_labels = {
        "Pending": "Chờ xử lý",
        "Confirmed": "Đã xác nhận",
        "Completed": "Hoàn thành",
        "Cancelled": "Đã hủy",
    }

    return [
        {
            "status": r["status"],
            "label": status_labels.get(r["status"], r["status"]),
            "count": r["count"],
        }
        for r in rows
    ]


# ---------- 6. Customer Insights ----------
@app.get("/analysis/customer-insights")
def customer_insights():
    """Phân tích khách hàng: top spenders, khách mới vs khách cũ."""

    top_spenders = query("""
        SELECT
            u.id_user,
            u.username,
            u.phone,
            COUNT(o.id_order) as order_count,
            COALESCE(SUM(o.total_amount), 0) as total_spent
        FROM "Users" u
        JOIN orders o ON o.id_user = u.id_user
        WHERE o."deletedAt" IS NULL AND o.status = 'Completed'
          AND u."deletedAt" IS NULL
        GROUP BY u.id_user, u.username, u.phone
        ORDER BY total_spent DESC
        LIMIT 10
    """)

    # Khách mới trong 30 ngày
    new_customers = query("""
        SELECT COUNT(*) as count
        FROM "Users"
        WHERE role = 'Customer'
          AND "deletedAt" IS NULL
          AND "createdAt" >= CURRENT_DATE - interval '30 days'
    """)

    # Khách có >= 2 đơn hoàn thành = khách quay lại
    returning = query("""
        SELECT COUNT(*) as count FROM (
            SELECT id_user
            FROM orders
            WHERE "deletedAt" IS NULL AND status = 'Completed' AND id_user IS NOT NULL
            GROUP BY id_user
            HAVING COUNT(*) >= 2
        ) sub
    """)

    return {
        "top_spenders": [
            {
                "id_user": r["id_user"],
                "username": r["username"],
                "phone": r["phone"],
                "order_count": r["order_count"],
                "total_spent": safe_float(r["total_spent"]),
            }
            for r in top_spenders
        ],
        "new_customers_30d": new_customers[0]["count"],
        "returning_customers": returning[0]["count"],
    }


# ---------- 7. Revenue Forecast (Linear Regression) ----------
@app.get("/analysis/forecast")
def forecast():
    """
    Dự báo doanh thu tháng tới bằng Linear Regression.
    Sử dụng dữ liệu 6 tháng gần nhất để huấn luyện model.
    """
    rows = query("""
        SELECT
            EXTRACT(EPOCH FROM date_trunc('month', "createdAt")) as month_epoch,
            TO_CHAR(date_trunc('month', "createdAt"), 'YYYY-MM') as month_label,
            COALESCE(SUM(total_amount), 0) as revenue
        FROM orders
        WHERE "deletedAt" IS NULL
          AND status = 'Completed'
          AND "createdAt" >= CURRENT_DATE - interval '6 months'
        GROUP BY date_trunc('month', "createdAt")
        ORDER BY date_trunc('month', "createdAt")
    """)

    if len(rows) < 2:
        return {
            "message": "Chưa đủ dữ liệu để dự báo (cần ít nhất 2 tháng có đơn hoàn thành)",
            "historical": [],
            "forecast_month": None,
            "forecast_revenue": None,
            "confidence": None,
        }

    X = np.array([safe_float(r["month_epoch"]) for r in rows]).reshape(-1, 1)
    y = np.array([safe_float(r["revenue"]) for r in rows])

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    # Dự báo tháng tới
    last_epoch = safe_float(rows[-1]["month_epoch"])
    next_epoch = last_epoch + 30 * 24 * 3600  # +30 ngày
    predicted = max(0, float(model.predict([[next_epoch]])[0]))

    # Label tháng tới
    last_month_dt = datetime.fromtimestamp(last_epoch)
    next_month_dt = last_month_dt + timedelta(days=32)
    next_label = next_month_dt.strftime("%Y-%m")

    return {
        "historical": [
            {"month": r["month_label"], "revenue": safe_float(r["revenue"])}
            for r in rows
        ],
        "forecast_month": next_label,
        "forecast_revenue": round(predicted),
        "confidence_r2": round(r2, 3),
        "model": "Linear Regression",
    }


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AI_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
