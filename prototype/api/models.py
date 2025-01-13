from app import db
from datetime import date

class Report(db.Model):
    __tablename__ = "report"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    description = db.Column(db.String)
    date = db.Column(db.Date, default=date.today)
    images = db.relationship("Image")

class Image(db.Model):
    __tablename__ = "image"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    file = db.Column(db.String)
    report_id = db.Column(db.Integer, db.ForeignKey("report.id"))
    failure = db.Column(db.String, default="sin defectos")
    predicted_failure = db.Column(db.String, default="sin defectos")
