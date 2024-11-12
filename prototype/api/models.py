from app import db
from datetime import date

class Report(db.Model):
    __tablename__ = "report"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    description = db.Column(db.String)
    date = db.Column(db.Date)
    images = db.relationship("Image")

    def __init__(self, name:str, description:str, images:list):
        self.name = name
        self.description = description
        self.date = date.today()
        self.images = images

class Image(db.Model):
    __tablename__ = "image"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    file = db.Column(db.String)
    report_id = db.Column(db.Integer, db.ForeignKey("report.id"))
    failure = db.Column(db.String, default="sin defectos")

    def __init__(self, name,file, failure=None):
        self.name = name
        self.file = file
        self.failure = failure
