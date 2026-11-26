# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import contextlib
import ipaddress
import logging
import os
import socket
import time

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None
    startup_timeout_s_raw = os.environ.get("RLLM_VLLM_SERVER_START_TIMEOUT_S", "180")
    try:
        startup_timeout_s = float(startup_timeout_s_raw)
    except (TypeError, ValueError):
        startup_timeout_s = 180.0
    startup_timeout_s = max(1.0, startup_timeout_s)

    for i in range(max_retries):
        sock = None
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            # Ray actors are not the process main thread; avoid signal-hook setup in uvicorn.
            server.install_signal_handlers = lambda: None
            server_task = asyncio.create_task(server.serve(sockets=[sock]))

            started_at = time.monotonic()
            while not server.started:
                if server_task.done():
                    exc = server_task.exception()
                    if exc is not None:
                        raise exc
                    raise RuntimeError(
                        f"uvicorn exited before startup completed on port {server_port}"
                    )
                elapsed = time.monotonic() - started_at
                if elapsed > startup_timeout_s:
                    server.should_exit = True
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(server_task, timeout=5.0)
                    raise TimeoutError(
                        f"Timed out waiting {startup_timeout_s:.1f}s for uvicorn startup "
                        f"on {server_address}:{server_port}"
                    )
                await asyncio.sleep(0.1)
            break
        except (OSError, SystemExit, TimeoutError, RuntimeError, Exception) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            if server_task is not None:
                if not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await server_task
                server_task = None
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close()
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
