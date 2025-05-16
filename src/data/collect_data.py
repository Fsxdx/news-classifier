import asyncio
from pathlib import Path
import fire

from parser import AsyncLentaParser


async def parser_runner(from_date: str, out_csv: str, max_workers: int) -> None:
    parser = AsyncLentaParser(
        from_date=from_date,
        out_csv=Path(out_csv),
        max_workers=max_workers,
    )
    async with parser:
        await parser.run()


def main(from_date, out_csv, max_workers=8):
    asyncio.run(parser_runner(from_date, out_csv, max_workers))


if __name__ == "__main__":
    fire.Fire(main)
