"""Ingestion worker: claims queued jobs and runs the ingestion pipeline.

    uv run rag-worker           # run until stopped (Ctrl-C / SIGTERM)
    uv run rag-worker --once    # drain the queue, then exit

Several workers can run side by side: claiming uses FOR UPDATE SKIP LOCKED,
so each job goes to exactly one worker. On shutdown, the current job finishes
first; if a worker dies mid-job, its lease expires and another worker retries.
"""

import argparse
import asyncio
import contextlib
import os
import signal
import socket

import structlog

from app.bootstrap.container import Container
from app.core.config import get_settings
from app.core.logging import configure_logging

logger = structlog.get_logger("app.worker")


async def run_worker(
    container: Container,
    *,
    worker_id: str,
    poll_interval: float,
    stop: asyncio.Event,
    once: bool = False,
) -> int:
    """Process jobs until `stop` is set (or the queue is empty, with `once`).
    Returns the number of jobs processed."""
    process = container.process_next_ingestion_job()
    processed = 0
    while not stop.is_set():
        try:
            outcome = await process.execute(worker_id)
        except Exception:  # e.g. database unreachable: log, back off, keep running
            logger.exception("worker.iteration_failed")
            outcome = None
        else:
            if outcome is not None:
                processed += 1
                logger.info(
                    "worker.job_finished",
                    job_id=str(outcome.job_id),
                    document_id=str(outcome.document_id),
                    succeeded=outcome.succeeded,
                    chunks=outcome.chunk_count,
                    will_retry=outcome.will_retry,
                    error=outcome.error,
                )
                continue  # there may be more work: don't sleep
            if once:
                break
        with contextlib.suppress(TimeoutError):  # idle: wait for new work or shutdown
            await asyncio.wait_for(stop.wait(), timeout=poll_interval)
    return processed


async def _main(once: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    container = Container(settings)
    worker_id = f"{socket.gethostname()}:{os.getpid()}"

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    logger.info("worker.started", worker_id=worker_id, embedding_model=settings.embedding_model)
    try:
        processed = await run_worker(
            container,
            worker_id=worker_id,
            poll_interval=settings.worker_poll_interval_seconds,
            stop=stop,
            once=once,
        )
    finally:
        await container.aclose()
    logger.info("worker.stopped", processed=processed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG ingestion worker")
    parser.add_argument("--once", action="store_true", help="Drain the queue, then exit")
    asyncio.run(_main(parser.parse_args().once))


if __name__ == "__main__":
    main()
