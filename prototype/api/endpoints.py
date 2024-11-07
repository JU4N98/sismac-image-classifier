from schemas import ReportSchema
from models import Report
from flask import request, jsonify
from db import db

def init_routes(app):
    @app.route("/report", methods=["POST"])
    def create_report():
        db.session.add(ReportSchema().load(request.json))
        db.session.commit()
        return {}, 200

    @app.route("/report", methods=["GET"])
    def get_reports():
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)
        reports = Report.query.order_by(Report.date.desc()).paginate(page=page,per_page=page_size).items
        return jsonify(ReportSchema(many=True).dump(reports)), 200
