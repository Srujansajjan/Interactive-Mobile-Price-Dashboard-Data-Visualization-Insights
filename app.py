import os
import io
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, Blueprint, render_template, request, redirect, url_for, flash, 
    current_app, send_file, jsonify
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user, UserMixin
)
from flask_migrate import Migrate
from sqlalchemy import (
    create_engine, Integer, String, ForeignKey, Float, Text, func, case
)
from sqlalchemy.orm import (
    scoped_session, sessionmaker, declarative_base, relationship, Mapped, mapped_column
)
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# CONFIGURATION
# ==========================================

def get_config() -> dict:
    database_uri = os.getenv(
        "DATABASE_URI",
        # Default to SQLite file for easy local run
        f"sqlite:///{os.path.abspath('app.db')}",
    )
    return {
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key-change"),
        "DATABASE_URI": database_uri,
        # Power BI: supply these if using secure embed with token
        "PBI_EMBED_URL": os.getenv("PBI_EMBED_URL", ""),
        "PBI_REPORT_URL": os.getenv("PBI_REPORT_URL", ""),
    }

# ==========================================
# MODELS
# ==========================================

Base = declarative_base()

class User(Base, UserMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="user")  # 'admin' or 'user'

class Brand(Base):
    __tablename__ = "brands"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    models = relationship("PhoneModel", back_populates="brand")

class PhoneModel(Base):
    __tablename__ = "phone_models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    ram: Mapped[str] = mapped_column(String(32))
    storage: Mapped[str] = mapped_column(String(32))
    camera: Mapped[str] = mapped_column(String(64))
    battery: Mapped[str] = mapped_column(String(64))
    processor: Mapped[str] = mapped_column(String(128))
    os: Mapped[str] = mapped_column(String(64))
    display_size: Mapped[str] = mapped_column(String(32))
    launch_year: Mapped[int] = mapped_column(Integer)

    brand = relationship("Brand", back_populates="models")
    sales = relationship("Sale", back_populates="phone_model")

class Sale(Base):
    __tablename__ = "sales"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("phone_models.id"), nullable=False)
    units_sold: Mapped[int] = mapped_column(Integer, default=0)
    total_revenue: Mapped[float] = mapped_column(Float, default=0.0)
    average_price: Mapped[float] = mapped_column(Float, default=0.0)
    region: Mapped[str] = mapped_column(String(100))
    channel: Mapped[str] = mapped_column(String(50))  # Online/Retail/Wholesale
    year: Mapped[int] = mapped_column(Integer)

    phone_model = relationship("PhoneModel", back_populates="sales")

# ==========================================
# BLUEPRINTS
# ==========================================

# --- Auth Blueprint ---
auth_bp = Blueprint("auth", __name__, template_folder="templates")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        SessionLocal = current_app.session_factory
        with SessionLocal() as db:
            user = db.query(User).filter(User.email == email).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                return redirect(url_for("dashboard.index"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        SessionLocal = current_app.session_factory
        with SessionLocal() as db:
            exists = db.query(User).filter(User.email == email).first()
            if exists:
                flash("Email already registered", "warning")
            else:
                user = User(email=email, password_hash=generate_password_hash(password), role="user")
                db.add(user)
                db.commit()
                flash("Account created. Please log in.", "success")
                return redirect(url_for("auth.login"))
    return render_template("signup.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))

# --- Dashboard Blueprint ---
dashboard_bp = Blueprint("dashboard", __name__, template_folder="templates")

@dashboard_bp.route("/")
@login_required
def index():
    # Get Power BI URL from Flask config (loaded from .env or environment)
    # Only use it if it's a valid URL (not placeholder)
    pbi_url = current_app.config.get("PBI_REPORT_URL", "").strip()
    if pbi_url and ("YOUR_REPORT_ID" in pbi_url or len(pbi_url) < 20):
        pbi_url = ""  # Ignore placeholder/invalid URLs
    return render_template("dashboard.html", pbi_url=pbi_url)

@dashboard_bp.route("/api/data")
@login_required
def api_data():
    """API endpoint to get dashboard data"""
    SessionLocal = current_app.session_factory
    with SessionLocal() as db:
        # Get filter parameters
        brand_filter = request.args.get("brand", "")
        model_filter = request.args.get("model", "")
        channel_filter = request.args.get("channel", "")
        region_filter = request.args.get("region", "")
        year_filter = request.args.get("year", "")
        price_filter = request.args.get("price", "")  # Format: "min-max"

        # Base query
        query = (
            db.query(
                Brand.name.label("brand"),
                PhoneModel.model_name.label("model"),
                PhoneModel.ram.label("ram"),
                PhoneModel.storage.label("storage"),
                Sale.units_sold,
                Sale.total_revenue,
                Sale.average_price,
                Sale.region,
                Sale.channel,
                Sale.year,
            )
            .join(PhoneModel, PhoneModel.id == Sale.model_id)
            .join(Brand, Brand.id == PhoneModel.brand_id)
        )

        # Apply filters
        if brand_filter:
            query = query.filter(Brand.name == brand_filter)
        if model_filter:
            query = query.filter(PhoneModel.model_name == model_filter)
        if channel_filter:
            query = query.filter(Sale.channel == channel_filter)
        if region_filter:
            query = query.filter(Sale.region == region_filter)
        if year_filter:
            try:
                query = query.filter(Sale.year == int(year_filter))
            except ValueError:
                pass  # Invalid year filter, ignore it

        # Apply price range filter
        if price_filter:
            try:
                if "-" in price_filter:
                    min_price, max_price = price_filter.split("-")
                    min_price = float(min_price)
                    max_price = float(max_price)
                    query = query.filter(Sale.average_price >= min_price, Sale.average_price <= max_price)
            except (ValueError, AttributeError):
                pass  # Invalid price filter, ignore it

        results = query.all()

        # Calculate KPIs
        total_units = sum(r.units_sold for r in results)
        total_revenue = sum(r.total_revenue for r in results)
        unique_models = len(set((r.brand, r.model) for r in results))
        unique_customers = len(set(r.region for r in results))  # Approximate

        # Brand-wise sales
        brand_sales = {}
        brand_revenue = {}
        for r in results:
            brand_sales[r.brand] = brand_sales.get(r.brand, 0) + r.units_sold
            brand_revenue[r.brand] = brand_revenue.get(r.brand, 0) + r.total_revenue

        # Channel distribution
        channel_sales = {}
        for r in results:
            channel_sales[r.channel] = channel_sales.get(r.channel, 0) + r.units_sold

        # Regional sales
        region_sales = {}
        for r in results:
            region_sales[r.region] = region_sales.get(r.region, 0) + r.units_sold

        # Yearly trends
        yearly_data = {}
        for r in results:
            if r.year not in yearly_data:
                yearly_data[r.year] = {"units": 0, "revenue": 0}
            yearly_data[r.year]["units"] += r.units_sold
            yearly_data[r.year]["revenue"] += r.total_revenue

        # Heatmap data: Sales by region and year
        heatmap_data = {}
        for r in results:
            key = f"{r.region}_{r.year}"
            if key not in heatmap_data:
                heatmap_data[key] = {"region": r.region, "year": r.year, "sales": 0}
            heatmap_data[key]["sales"] += r.units_sold

        # Treemap data: Brand/Model hierarchy
        treemap_data = {}
        for r in results:
            if r.brand not in treemap_data:
                treemap_data[r.brand] = {"name": r.brand, "value": 0, "children": {}}
            if r.model not in treemap_data[r.brand]["children"]:
                treemap_data[r.brand]["children"][r.model] = {"name": r.model, "value": 0}
            treemap_data[r.brand]["value"] += r.units_sold
            treemap_data[r.brand]["children"][r.model]["value"] += r.units_sold

        # Top performing models data
        model_performance = {}
        for r in results:
            key = f"{r.brand} - {r.model}"
            if key not in model_performance:
                model_performance[key] = {
                    "brand": r.brand,
                    "model": r.model,
                    "units_sold": 0,
                    "total_revenue": 0,
                    "avg_price": 0,
                    "regions": set(),
                    "channels": set()
                }
            model_performance[key]["units_sold"] += r.units_sold
            model_performance[key]["total_revenue"] += r.total_revenue
            model_performance[key]["regions"].add(r.region)
            model_performance[key]["channels"].add(r.channel)
        
        # Calculate average price and convert sets to counts
        for key, data in model_performance.items():
            # Calculate weighted average price
            if data["units_sold"] > 0:
                data["avg_price"] = data["total_revenue"] / data["units_sold"]
            data["region_count"] = len(data["regions"])
            data["channel_count"] = len(data["channels"])
            del data["regions"]
            del data["channels"]
        
        top_models_data = list(model_performance.values())

        # Scatter plot data: Price vs Units Sold
        scatter_data = []
        for r in results:
            scatter_data.append({
                "x": r.average_price,
                "y": r.units_sold,
                "brand": r.brand,
                "model": r.model
            })

        # Correlation data: Specs vs Sales
        correlation_data = []
        for r in results:
            ram_val = 0
            storage_val = 0
            if r.ram:
                try:
                    ram_str = str(r.ram).replace("GB", "").replace(" ", "").strip()
                    ram_val = int(ram_str) if ram_str else 0
                except:
                    ram_val = 0
            if r.storage:
                try:
                    storage_str = str(r.storage).replace("GB", "").replace(" ", "").strip()
                    storage_val = int(storage_str) if storage_str else 0
                except:
                    storage_val = 0
            correlation_data.append({
                "ram": ram_val,
                "storage": storage_val,
                "units_sold": r.units_sold,
                "price": r.average_price,
                "revenue": r.total_revenue
            })

        # Get unique filter options
        brands = sorted([b.name for b in db.query(Brand).distinct().all()])
        models = sorted(
            [m.model_name for m in db.query(PhoneModel).distinct().all()]
        )[:100]  # Limit to 100 for dropdown
        channels = sorted(list(set(r.channel for r in results if r.channel)))
        regions = sorted(list(set(r.region for r in results if r.region)))
        years = sorted(list(set(r.year for r in results if r.year)))

        return jsonify(
            {
                "kpis": {
                    "total_units": total_units,
                    "total_revenue": total_revenue,
                    "total_models": unique_models,
                    "total_customers": unique_customers,
                },
                "brand_sales": brand_sales,
                "brand_revenue": brand_revenue,
                "channel_sales": channel_sales,
                "region_sales": region_sales,
                "yearly_trends": yearly_data,
                "heatmap_data": heatmap_data,
                "treemap_data": treemap_data,
                "top_models_data": top_models_data,
                "scatter_data": scatter_data,
                "correlation_data": correlation_data,
                "filters": {
                    "brands": brands,
                    "models": models,
                    "channels": channels,
                    "regions": regions,
                    "years": years,
                },
            }
        )

# --- Insights Blueprint ---
insights_bp = Blueprint("insights", __name__, template_folder="templates")

def generate_insights(db):
    """Generate automatic insights from sales data"""
    insights = []
    
    # 1. Top brand by region and year
    region_year_query = (
        db.query(
            Brand.name.label("brand"),
            Sale.region,
            Sale.year,
            func.sum(Sale.units_sold).label("total_units")
        )
        .join(PhoneModel, PhoneModel.id == Sale.model_id)
        .join(Brand, Brand.id == PhoneModel.brand_id)
        .group_by(Brand.name, Sale.region, Sale.year)
        .order_by(Sale.year.desc(), func.sum(Sale.units_sold).desc())
    )
    
    # Get top brand for each region-year combination
    region_year_sales = {}
    for row in region_year_query.all():
        key = f"{row.region}_{row.year}"
        if key not in region_year_sales:
            region_year_sales[key] = {
                "brand": row.brand,
                "region": row.region,
                "year": row.year,
                "units": row.total_units
            }
    
    # Generate insights for top brands by region
    for key, data in list(region_year_sales.items())[:5]:  # Top 5
        insights.append({
            "type": "performance",
            "severity": "info",
            "title": f"{data['brand']} models sold highest in {data['region']} in {data['year']}.",
            "description": f"Total units sold: {data['units']:,}",
            "icon": "trending-up"
        })
    
    # 2. Year-over-year sales changes by brand
    brand_yearly = (
        db.query(
            Brand.name.label("brand"),
            Sale.year,
            func.sum(Sale.units_sold).label("total_units")
        )
        .join(PhoneModel, PhoneModel.id == Sale.model_id)
        .join(Brand, Brand.id == PhoneModel.brand_id)
        .group_by(Brand.name, Sale.year)
        .order_by(Brand.name, Sale.year)
    )
    
    brand_sales_by_year = {}
    for row in brand_yearly.all():
        if row.brand not in brand_sales_by_year:
            brand_sales_by_year[row.brand] = {}
        brand_sales_by_year[row.brand][row.year] = row.total_units
    
    # Calculate YoY changes
    for brand, year_data in brand_sales_by_year.items():
        years = sorted(year_data.keys())
        if len(years) >= 2:
            prev_year = years[-2]
            curr_year = years[-1]
            prev_sales = year_data[prev_year]
            curr_sales = year_data[curr_year]
            
            if prev_sales > 0:
                change_pct = ((curr_sales - prev_sales) / prev_sales) * 100
                if abs(change_pct) >= 5:  # Only show significant changes
                    insights.append({
                        "type": "trend",
                        "severity": "warning" if change_pct < 0 else "success",
                        "title": f"{brand} sales {'dropped' if change_pct < 0 else 'increased'} {abs(change_pct):.1f}% in {curr_year}.",
                        "description": f"From {prev_sales:,} units in {prev_year} to {curr_sales:,} units in {curr_year}",
                        "icon": "trending-down" if change_pct < 0 else "trending-up"
                    })
    
    # 3. Battery capacity correlation with sales
    battery_sales_query = (
        db.query(
            PhoneModel.battery,
            func.sum(Sale.units_sold).label("total_units"),
            func.avg(Sale.units_sold).label("avg_units")
        )
        .join(Sale, Sale.model_id == PhoneModel.id)
        .filter(PhoneModel.battery.isnot(None), PhoneModel.battery != "")
        .group_by(PhoneModel.battery)
        .order_by(func.sum(Sale.units_sold).desc())
    )
    
    battery_data = battery_sales_query.all()
    if len(battery_data) >= 2:
        # Check if there's a clear pattern (higher battery = more sales)
        top_battery = battery_data[0]
        if top_battery.total_units > 0:
            # Extract numeric battery value for comparison
            try:
                battery_str = str(top_battery.battery).replace("mAh", "").replace(" ", "").strip()
                battery_val = int(battery_str) if battery_str else 0
                
                if battery_val > 4000:  # High capacity
                    insights.append({
                        "type": "correlation",
                        "severity": "info",
                        "title": "Battery capacity strongly influences sales.",
                        "description": f"Models with {top_battery.battery} battery show highest sales performance",
                        "icon": "battery-full"
                    })
            except:
                pass
    
    # 4. RAM correlation
    ram_sales_query = (
        db.query(
            PhoneModel.ram,
            func.sum(Sale.units_sold).label("total_units")
        )
        .join(Sale, Sale.model_id == PhoneModel.id)
        .filter(PhoneModel.ram.isnot(None), PhoneModel.ram != "")
        .group_by(PhoneModel.ram)
        .order_by(func.sum(Sale.units_sold).desc())
    )
    
    ram_data = ram_sales_query.first()
    if ram_data and ram_data.total_units > 0:
        insights.append({
            "type": "correlation",
            "severity": "info",
            "title": f"Models with {ram_data.ram} RAM show highest sales.",
            "description": f"Total units sold: {ram_data.total_units:,}",
            "icon": "memory"
        })
    
    # 5. Storage correlation
    storage_sales_query = (
        db.query(
            PhoneModel.storage,
            func.sum(Sale.units_sold).label("total_units")
        )
        .join(Sale, Sale.model_id == PhoneModel.id)
        .filter(PhoneModel.storage.isnot(None), PhoneModel.storage != "")
        .group_by(PhoneModel.storage)
        .order_by(func.sum(Sale.units_sold).desc())
    )
    
    storage_data = storage_sales_query.first()
    if storage_data and storage_data.total_units > 0:
        insights.append({
            "type": "correlation",
            "severity": "info",
            "title": f"Models with {storage_data.storage} storage are most popular.",
            "description": f"Total units sold: {storage_data.total_units:,}",
            "icon": "hard-drive"
        })
    
    # 6. Channel performance
    channel_query = (
        db.query(
            Sale.channel,
            func.sum(Sale.units_sold).label("total_units"),
            func.sum(Sale.total_revenue).label("total_revenue")
        )
        .group_by(Sale.channel)
        .order_by(func.sum(Sale.units_sold).desc())
    )
    
    top_channel = channel_query.first()
    if top_channel:
        insights.append({
            "type": "performance",
            "severity": "success",
            "title": f"{top_channel.channel} channel drives highest sales.",
            "description": f"{top_channel.total_units:,} units sold, ₹{top_channel.total_revenue:,.0f} revenue",
            "icon": "shopping-cart"
        })
    
    return insights

@insights_bp.route("/insights")
@login_required
def index():
    """Display insights and alerts page"""
    SessionLocal = current_app.session_factory
    with SessionLocal() as db:
        insights = generate_insights(db)
    
    # Separate insights by type
    performance_insights = [i for i in insights if i["type"] == "performance"]
    trend_insights = [i for i in insights if i["type"] == "trend"]
    correlation_insights = [i for i in insights if i["type"] == "correlation"]
    
    return render_template(
        "insights.html",
        performance_insights=performance_insights,
        trend_insights=trend_insights,
        correlation_insights=correlation_insights,
        all_insights=insights
    )

@insights_bp.route("/insights/api")
@login_required
def api_insights():
    """API endpoint to get insights as JSON"""
    SessionLocal = current_app.session_factory
    with SessionLocal() as db:
        insights = generate_insights(db)
    
    return jsonify({"insights": insights})

# --- Admin Blueprint ---
admin_bp = Blueprint("admin", __name__, url_prefix="/admin", template_folder="templates")

def require_admin() -> bool:
    return bool(current_user.is_authenticated and getattr(current_user, "role", "user") == "admin")

@admin_bp.before_request
def guard_admin():
    # Allow non-admins to hit nothing under /admin
    if request.endpoint and request.endpoint.startswith("admin."):
        if not require_admin():
            return redirect(url_for("dashboard.index"))

@admin_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Please choose a CSV file", "warning")
            return redirect(request.url)

        df = pd.read_csv(file)
        required_cols = {
            "Brand",
            "Model",
            "RAM",
            "Storage",
            "Camera",
            "Battery",
            "Processor",
            "Price",
            "Units Sold",
            "Region",
            "Channel",
            "Year",
        }
        if not required_cols.issubset(set(df.columns)):
            flash("CSV missing required columns", "danger")
            return redirect(request.url)

        SessionLocal = current_app.session_factory
        with SessionLocal() as db:
            brand_cache = {b.name: b for b in db.query(Brand).all()}
            for _, row in df.iterrows():
                brand = brand_cache.get(row["Brand"]) or Brand(name=row["Brand"])
                if brand.id is None:
                    db.add(brand)
                    db.flush()
                    brand_cache[brand.name] = brand

                model = (
                    db.query(PhoneModel)
                    .filter(PhoneModel.brand_id == brand.id, PhoneModel.model_name == row["Model"]).first()
                )
                if not model:
                    model = PhoneModel(
                        brand_id=brand.id,
                        model_name=row["Model"],
                        ram=row.get("RAM", ""),
                        storage=row.get("Storage", ""),
                        camera=row.get("Camera", ""),
                        battery=row.get("Battery", ""),
                        processor=row.get("Processor", ""),
                        os=str(row.get("OS", "")),
                        display_size=str(row.get("Display Size", "")),
                        launch_year=int(row.get("Year", 0) or 0),
                    )
                    db.add(model)
                    db.flush()

                sale = Sale(
                    model_id=model.id,
                    units_sold=int(row.get("Units Sold", 0) or 0),
                    total_revenue=float(row.get("Price", 0) or 0) * int(row.get("Units Sold", 0) or 0),
                    average_price=float(str(row.get("Price", 0)).replace(",", "").replace("₹", "") or 0),
                    region=row.get("Region", ""),
                    channel=row.get("Channel", ""),
                    year=int(row.get("Year", 0) or 0),
                )
                db.add(sale)

            db.commit()
        flash("Data uploaded successfully", "success")
        return redirect(url_for("dashboard.index"))
    return render_template("admin_upload.html")

@admin_bp.route("/export/<string:format>", methods=["GET"])
@login_required
def export(format: str):
    if not require_admin():
        return redirect(url_for("dashboard.index"))
    SessionLocal = current_app.session_factory
    with SessionLocal() as db:
        query = (
            db.query(
                Brand.name.label("Brand"),
                PhoneModel.model_name.label("Model"),
                PhoneModel.ram.label("RAM"),
                PhoneModel.storage.label("Storage"),
                PhoneModel.camera.label("Camera"),
                PhoneModel.battery.label("Battery"),
                PhoneModel.processor.label("Processor"),
                Sale.average_price.label("Price"),
                Sale.units_sold.label("Units Sold"),
                Sale.region.label("Region"),
                Sale.channel.label("Channel"),
                Sale.year.label("Year"),
            )
            .join(PhoneModel, PhoneModel.id == Sale.model_id)
            .join(Brand, Brand.id == PhoneModel.brand_id)
        )
        df = pd.read_sql(query.statement, db.bind)

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode()), as_attachment=True, download_name="sales_export.csv")
    if format == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name="sales_export.xlsx")
    if format == "pdf":
        # Simple tabular PDF via pandas -> string; for production, create a rich PDF report
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=A4)
        textobject = c.beginText(40, 800)
        textobject.textLine("Mobile Sales Export")
        textobject.textLine("")
        for line in df.head(100).to_string(index=False).splitlines():
            textobject.textLine(line[:95])
        c.drawText(textobject)
        c.showPage()
        c.save()
        pdf_buf.seek(0)
        return send_file(pdf_buf, as_attachment=True, download_name="sales_export.pdf")
    return redirect(url_for("dashboard.index"))

# ==========================================
# APP FACTORY
# ==========================================

def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__)
    cfg = get_config()
    app.config.update(
        SECRET_KEY=cfg["SECRET_KEY"],
        SQLALCHEMY_DATABASE_URI=cfg["DATABASE_URI"],
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        PBI_REPORT_URL=cfg["PBI_REPORT_URL"],  # Add Power BI URL to Flask config
    )

    engine = create_engine(cfg["DATABASE_URI"], future=True)
    Base.metadata.create_all(engine)
    SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        with SessionLocal() as db:
            return db.get(User, int(user_id))

    # store db session factory on app
    app.session_factory = SessionLocal  # type: ignore[attr-defined]

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(insights_bp)

    @app.teardown_appcontext
    def remove_session(exception=None):
        SessionLocal.remove()

    return app

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)
