from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from typing import Any, Callable, List, Optional, Sequence


class MicroBatcher:
    """
    Collects items from many coroutines (= many websocket sessions) and runs
    them through `fn` as one batch on a dedicated executor (the GPU thread).

    fn(list_of_items) -> list_of_results, same length and order.

    A batch is flushed when it reaches `max_batch` items or when the first
    item has waited `max_wait` seconds, whichever comes first. With a single
    active session this adds at most `max_wait` of latency; under load the
    GPU sees large batches instead of N competing batch-1 calls.
    """

    def __init__(
        self,
        fn: Callable[[Sequence[Any]], List[Any]],
        executor: Executor,
        *,
        max_batch: int = 64,
        max_wait: float = 0.008,
        max_queue: int = 1024,
    ) -> None:
        self._fn = fn
        self._executor = executor
        self._max_batch = max(1, int(max_batch))
        self._max_wait = max(0.0, float(max_wait))
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: Optional[asyncio.Task] = None

        # Stats for /health
        self.batches = 0
        self.items = 0
        self.largest_batch = 0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        while not self._queue.empty():
            _, future = self._queue.get_nowait()
            if not future.done():
                future.cancel()

    async def submit(self, item: Any) -> Any:
        """Awaitable result for one item. Blocks (backpressure) if the queue is full."""
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((item, future))
        return await future

    def stats(self) -> dict:
        return {
            "batches": self.batches,
            "items": self.items,
            "avg_batch": round(self.items / self.batches, 2) if self.batches else 0.0,
            "largest_batch": self.largest_batch,
            "queued": self._queue.qsize(),
        }

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self._max_wait

            while len(batch) < self._max_batch:
                # Take whatever is already queued without waiting.
                try:
                    batch.append(self._queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass

                remaining = deadline - loop.time()
                if remaining <= 0:
                    break

                try:
                    batch.append(
                        await asyncio.wait_for(self._queue.get(), remaining)
                    )
                except asyncio.TimeoutError:
                    break

            # Callers that timed out / disconnected have cancelled futures.
            batch = [(item, fut) for item, fut in batch if not fut.done()]
            if not batch:
                continue

            try:
                results = await loop.run_in_executor(
                    self._executor,
                    self._fn,
                    [item for item, _ in batch],
                )

                if len(results) != len(batch):
                    raise RuntimeError(
                        f"batch fn returned {len(results)} results "
                        f"for {len(batch)} items"
                    )

            except asyncio.CancelledError:
                for _, fut in batch:
                    if not fut.done():
                        fut.cancel()
                raise

            except Exception as exc:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue

            self.batches += 1
            self.items += len(batch)
            self.largest_batch = max(self.largest_batch, len(batch))

            for (_, fut), result in zip(batch, results):
                if not fut.done():
                    fut.set_result(result)