"""Watcher process manager — starts/stops ring-watch as a subprocess."""

from __future__ import annotations

import asyncio
import collections
import logging
import signal
import sys
import time

log = logging.getLogger(__name__)

MAX_LOG_LINES = 2000


class WatcherManager:
    """Manages the ring-watch process lifecycle."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._log_buffer: collections.deque[str] = collections.deque(maxlen=MAX_LOG_LINES)
        self._started_at: float | None = None
        self._reader_task: asyncio.Task | None = None
        self._exit_code: int | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def status(self) -> dict:
        if self.is_running:
            uptime = int(time.time() - self._started_at) if self._started_at else 0
            return {
                "state": "running",
                "pid": self._process.pid,
                "uptime_seconds": uptime,
                "exit_code": None,
            }
        return {
            "state": "stopped",
            "pid": None,
            "uptime_seconds": 0,
            "exit_code": self._exit_code,
        }

    @property
    def logs(self) -> list[str]:
        return list(self._log_buffer)

    async def start(self) -> bool:
        """Start ring-watch as a subprocess. Returns False if already running."""
        if self.is_running:
            return False

        self._log_buffer.clear()
        self._exit_code = None
        self._log_buffer.append(f"[manager] Starting ring-watch at {time.strftime('%H:%M:%S')}")

        import os

        self._log_file = "/tmp/watcher.log"
        open(self._log_file, "w").close()

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONWARNINGS": "ignore",
            "WATCHER_LOG_FILE": self._log_file,
        }
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-W",
            "ignore",
            "-m",
            "ring_detector.watcher",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        self._started_at = time.time()
        self._reader_task = asyncio.create_task(self._tail_log_file(), name="watcher_log_reader")
        log.info("Watcher started (PID %d)", self._process.pid)
        self._log_buffer.append(f"[manager] Process started (PID {self._process.pid})")
        return True

    async def stop(self) -> bool:
        """Stop the watcher process gracefully. Returns False if not running."""
        if not self.is_running:
            return False

        self._log_buffer.append(f"[manager] Stopping watcher at {time.strftime('%H:%M:%S')}")
        log.info("Stopping watcher (PID %d)", self._process.pid)

        # Send SIGTERM for graceful shutdown
        self._process.send_signal(signal.SIGTERM)

        try:
            await asyncio.wait_for(self._process.wait(), timeout=15)
        except TimeoutError:
            log.warning("Watcher did not stop gracefully, sending SIGKILL")
            self._log_buffer.append("[manager] Force killing watcher (timeout)")
            self._process.kill()
            await self._process.wait()

        self._exit_code = self._process.returncode
        self._log_buffer.append(f"[manager] Watcher stopped (exit code {self._exit_code})")
        log.info("Watcher stopped (exit code %d)", self._exit_code)

        for task in [self._reader_task]:
            if task and not task.done():
                task.cancel()
                with asyncio.suppress(asyncio.CancelledError):
                    await task

        self._process = None
        self._started_at = None
        return True

    async def _tail_log_file(self) -> None:
        """Tail the watcher log file and buffer lines."""
        import aiofiles

        try:
            async with aiofiles.open(self._log_file) as f:
                while self.is_running:
                    line = await f.readline()
                    if line:
                        text = line.rstrip()
                        if text:
                            self._log_buffer.append(text)
                    else:
                        await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Error tailing watcher log")

        # Read remaining lines after process exit
        try:
            with open(self._log_file) as f:
                for line in f:
                    text = line.rstrip()
                    if text and text not in self._log_buffer:
                        self._log_buffer.append(text)
        except Exception:
            pass

        if self._process and self._process.returncode is not None:
            self._exit_code = self._process.returncode
            self._log_buffer.append(f"[manager] Process exited (code {self._exit_code})")


watcher_mgr = WatcherManager()
