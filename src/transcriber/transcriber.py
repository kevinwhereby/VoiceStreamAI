import aiohttp
import io


class Transcriber:
    def __init__(self, url):
        self.url = url

    async def transcribe(self, bytes):
        form = aiohttp.FormData()
        form.add_field(
            "file", io.BytesIO(bytes), filename="audio.raw", content_type="audio/x-raw"
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, data=form) as response:
                response.raise_for_status()
                return await response.json()
