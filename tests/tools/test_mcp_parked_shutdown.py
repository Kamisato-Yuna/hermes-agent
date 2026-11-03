"""MCP shutdown regression coverage for parked server tasks."""

import asyncio
import threading
from unittest.mock import patch

from tools import mcp_tool_loop as mcp_loop
from tools import mcp_tool as mcp_mod
from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_lifecycle import shutdown_mcp_servers


def test_shutdown_drains_parked_server_after_bounded_wait_expires():
    """The public shutdown path drains a parked server if graceful shutdown stalls.

    This exercises the production ownership path: a real ``MCPServerTask``
    is registered, its parked waiter owns child tasks on the shared loop,
    and the bounded wait for the scheduled shutdown expires. The loop owner
    must still cancel and drain that waiter before closing the loop.
    """
    shutdown_started = threading.Event()
    parked_task_done = threading.Event()
    scheduled_shutdown = {}
    schedule_count = 0

    class StalledShutdownServer(MCPServerTask):
        async def shutdown(self):
            shutdown_started.set()
            await asyncio.Event().wait()

    server = StalledShutdownServer("parked")
    with mcp_mod._lock:
        mcp_mod._servers.clear()
        mcp_mod._server_connecting.clear()
    mcp_loop._ensure_mcp_loop()
    with mcp_mod._lock:
        loop = mcp_mod._mcp_loop
    assert loop is not None

    async def install_parked_waiter():
        task = asyncio.create_task(server._wait_for_reconnect_or_shutdown())
        server._task = task
        task.add_done_callback(lambda _task: parked_task_done.set())
        await asyncio.sleep(0)
        return task

    parked_task = asyncio.run_coroutine_threadsafe(
        install_parked_waiter(), loop
    ).result(timeout=2)
    with mcp_mod._lock:
        mcp_mod._servers[server.name] = server

    def schedule_then_report_timeout(coro, target_loop, **_kwargs):
        nonlocal schedule_count
        schedule_count += 1
        future = asyncio.run_coroutine_threadsafe(coro, target_loop)
        if schedule_count > 1:
            return future

        scheduled_shutdown["future"] = future

        class TimedOutFuture:
            def result(self, timeout):
                assert timeout > 0
                assert shutdown_started.wait(timeout=2)
                raise TimeoutError("simulated bounded MCP shutdown timeout")

        return TimedOutFuture()

    try:
        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=schedule_then_report_timeout,
        ):
            shutdown_mcp_servers()

        assert loop.is_closed()
        assert parked_task_done.is_set(), (
            "parked MCPServerTask was not drained before its loop closed"
        )
        assert parked_task.done()
        assert scheduled_shutdown["future"].done()
        assert schedule_count == 2
    finally:
        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
        mcp_loop._stop_mcp_loop()
