from .customers import customer_bp
from .model import model_bp
from .export import export_bp

def register_routes(app):
    app.register_blueprint(customer_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(export_bp)