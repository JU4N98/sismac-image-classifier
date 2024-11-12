from marshmallow import Schema, fields, post_load, post_dump
from models import Image, Report

class ImageSchema(Schema):
    id = fields.Int(dump_only=True)
    file = fields.Str(required=True)
    name = fields.Str(required=True)
    failure = fields.Str(required=False)

    @post_load
    def make_image(self, data, **kwargs):
        return Image(**data)

class ReportSchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True)
    description = fields.Str(required=False)
    date = fields.Date(required=False)
    images = fields.List(fields.Nested(ImageSchema))

    @post_load
    def make_report(self, data, **kwargs):
        return Report(**data)
    