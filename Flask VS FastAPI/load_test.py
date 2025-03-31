import asyncio
import aiohttp
import time
from statistics import mean

# 修改成你的 API 端點，例如：
URL = "http://127.0.0.1:5000/delay"

# 測試參數
TOTAL_REQUESTS = 1000
CONCURRENT_REQUESTS = 100  # 同時併發數

results = []

sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

async def fetch(session, i):
    async with sem:
        start = time.perf_counter()
        try:
            async with session.get(URL) as response:
                await response.text()  # 確保完成 response 讀取
                end = time.perf_counter()
                results.append(end - start)
        except Exception as e:
            print(f"[{i}] 發生錯誤：{e}")

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, i) for i in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)

    print("\n=== 測試結果 ===")
    print(f"總請求數：{len(results)}")
    print(f"平均回應時間：{mean(results):.3f} 秒")
    print(f"最快回應：{min(results):.3f} 秒")
    print(f"最慢回應：{max(results):.3f} 秒")

if __name__ == "__main__":
    asyncio.run(main())
