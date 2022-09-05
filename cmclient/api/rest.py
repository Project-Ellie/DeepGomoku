from requests import HTTPError


class RestAdapter:

    def __init__(self, rest_api, base_url):
        """
        Takes anything that has a get and a post method
        """
        self.rest_api = rest_api
        self.base_url = base_url

    def post(self, path, body) -> bytes:
        response = self.rest_api.post(self.base_url + path, data=body)
        if response.status_code != 200:
            raise HTTPError(response.status_code)

        return response.content

    def get(self, path) -> bytes:
        response = self.rest_api.get(self.base_url + path)
        if response.status_code != 200:
            raise HTTPError(response.status_code)
        return response.content
