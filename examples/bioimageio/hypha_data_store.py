import json
import uuid
import mimetypes
import os
from urllib.parse import parse_qs

class HyphaDataStore:
    def __init__(self):
        self.storage = {}
        self._svc = None
        self._server = None

    async def setup(self, server, service_id="data-store", visibility="public"):
        self._server = server
        self._svc = await server.register_service({
            "id": service_id,
            "type": "functions",
            "config": {
              "visibility": visibility,
              "require_context": False
            },
            "get": self.http_get,
        }, overwrite=True)

    def get_url(self, obj_id: str):
        assert self._svc, "Service not initialized, call `setup()`"
        assert obj_id in self.storage, "Object not found " + obj_id
        return f"{self._server.config.public_base_url}/{self._server.config.workspace}/apps/{self._svc.id.split(':')[1]}/get?id={obj_id}"

    def put(self, obj_type: str, value: any, name: str, comment: str = ""):
        assert self._svc, "Please call `setup()` before using the store"
        obj_id = str(uuid.uuid4())
        if obj_type == 'file':
            data = value
            assert isinstance(data, (str, bytes)), "Value must be a string or bytes"
            if isinstance(data, str) and data.startswith("file://"):
                # File URL examples:
                # Absolute URL: `file:///home/data/myfile.png`
                # Relative URL: `file://./myimage.png`, or `file://myimage.png`
                with open(data.replace("file://", ""), 'rb') as fil:
                    data = fil.read()
            mime_type, _ = mimetypes.guess_type(name)
            self.storage[obj_id] = {
                'type': obj_type,
                'name': name,
                'value': data,
                'mime_type': mime_type or 'application/octet-stream',
                'comment': comment
            }
        else:
            self.storage[obj_id] = {
                'type': obj_type,
                'name': name,
                'value': value,
                'mime_type': 'application/json',
                'comment': comment
            }
        return obj_id

    def get(self, id: str):
        assert self._svc, "Please call `setup()` before using the store"
        obj = self.storage.get(id)
        return obj

    def http_get(self, scope, context=None):
        query_string = scope['query_string']
        id = parse_qs(query_string).get('id', [])[0]
        obj = self.storage.get(id)
        if obj is None:
            return {'status': 404, 'headers': {}, 'body': "Not found: " + id}

        if obj['type'] == 'file':
            data = obj['value']
            if isinstance(data, str):
                if not os.path.isfile(data):
                    return {
                        "status": 404,
                        'headers': {'Content-Type': 'text/plain'},
                        "body": "File not found: " + data
                    }
                with open(data, 'rb') as fil:
                    data = fil.read()
            headers = {
                'Content-Type': obj['mime_type'],
                'Content-Length': str(len(obj['value'])),
                'Content-Disposition': f'inline; filename="{obj["name"].split("/")[-1]}"'
            }

            return {
                'status': 200,
                'headers': headers,
                'body': obj['value']
            }
        else:
            return {
                'status': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(obj['value'])
            }

    def http_list(self, scope, context=None):
        query_string = scope.get('query_string', b'')
        kws = parse_qs(query_string).get('keyword', [])
        keyword = kws[0] if kws else None
        result = [value for key, value in self.storage.items() if not keyword or keyword in value['name']]
        return {'status': 200, 'headers': {'Content-Type': 'application/json'}, 'body': json.dumps(result)}

    def remove(self, obj_id: str):
        assert self._svc, "Please call `setup()` before using the store"
        if obj_id in self.storage:
            del self.storage[obj_id]
            return True
        raise IndexError("Not found: " + obj_id)

async def test_data_store(server_url="https://ai.imjoy.io"):
    from imjoy_rpc.hypha import connect_to_server, login
    token = await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})

    ds = HyphaDataStore()
    # Setup would need to be completed in an ASGI compatible environment
    await ds.setup(server)

    # Test PUT operation
    file_id = ds.put('file', 'file:///home/data.txt', 'data.txt')
    binary_id = ds.put('file', b'Some binary content', 'example.bin')
    json_id = ds.put('json', {'hello': 'world'}, 'example.json')

    # Test GET operation
    assert ds.get(file_id)['type'] == 'file'
    assert ds.get(binary_id)['type'] == 'file'
    assert ds.get(json_id)['type'] == 'json'

    # Test GET URL generation
    print("URL for getting file", ds.get_url(file_id))
    print("URL for getting binary object", ds.get_url(binary_id))
    print("URL for getting json object", ds.get_url(json_id))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_data_store())
