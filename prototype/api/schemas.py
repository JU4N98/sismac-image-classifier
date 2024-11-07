from marshmallow import Schema, fields, post_load
from models import Image, Report
import base64

class ByteField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        ret = None
        try:
            ret = base64.b64encode(value).decode('utf-8')
        finally:
            return ret

    def _deserialize(self, value, attr, data, **kwargs):
        ret = None
        try:
            ret = base64.b64decode(value.encode('utf-8'))
        finally:
            return ret

class ImageSchema(Schema):
    name = fields.Str(required=True)
    bytes = ByteField(required=True)
    failure = fields.Str(required=False)
    @post_load
    def make_image(self, data, **kwargs):
        return Image(**data)

class ReportSchema(Schema):
    name = fields.Str(required=True)
    description = fields.Str(required=False)
    date = fields.Date(required=False)
    images = fields.List(fields.Nested(ImageSchema))

    @post_load
    def make_report(self, data, **kwargs):
        return Report(**data)
