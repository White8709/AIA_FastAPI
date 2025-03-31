from quart import Quart
import aiohttp

app = Quart(__name__)

@app.route("/delay")
async def delayed_response():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://httpbin.org/delay/1") as response:
            await response.text()
    return {"message": "Quart 模擬外部 API I/O 延遲"}

if __name__ == "__main__":
    app.run()
