from json import JSONEncoder
from uuid import UUID


class MeiliJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)