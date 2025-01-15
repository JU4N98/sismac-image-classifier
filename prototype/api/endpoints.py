from schemas import ReportSchema, ImageSchema
from models import Report, Image
from flask import request, jsonify
from db import db
from predict import predict

def init_routes(app):
    @app.route("/report", methods=["POST"])
    def create_report():
        report = ReportSchema().load(request.json)
        predict(report.images)
        db.session.add(report)
        db.session.commit()
        return jsonify(ReportSchema().dump(report)), 200

    @app.route("/report", methods=["GET"])
    def list_reports():
        page = request.args.get("page", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)
        reports = Report.query.order_by(Report.date.desc()).paginate(page=page,per_page=page_size).items
        return jsonify(ReportSchema(many=True).dump(reports)), 200
    
    @app.route("/report/<reportId>", methods=["GET"])
    def get_report(reportId:int):
        report = Report.query.get(reportId)
        return jsonify(ReportSchema(many=False).dump(report)), 200
    
    @app.route("/image/<imageId>", methods=["PUT"])
    def update_report(imageId:int):
        failure = ImageSchema().load(request.json).failure
        image = Image.query.get(imageId)
        image.failure = failure
        db.session.commit()
        return {}, 200
