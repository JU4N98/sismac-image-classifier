from marshmallow import Schema, fields, post_load
from models import Image, Report

class ImageSchema(Schema):
    file = fields.Str(required=True)
    name = fields.Str(required=True)
    failure = fields.Str(required=False)

    @post_load
    def make_image(self, data, **kwargs):
        return Image(**data)

class ReportSchema(Schema):
    id = fields.Int(required=False)
    name = fields.Str(required=True)
    description = fields.Str(required=False)
    date = fields.Date(required=False)
    images = fields.List(fields.Nested(ImageSchema))

    @post_load
    def make_report(self, data, **kwargs):
        return Report(**data)
