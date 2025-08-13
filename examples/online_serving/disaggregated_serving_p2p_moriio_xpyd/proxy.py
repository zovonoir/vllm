# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5

def _listen_for_register(hostname, port):
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket,zmq.POLLIN)
    global prefill_instances
    global decode_instances

    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_addr,msg = router_socket.recv_multipart()
            data = msgpack.loads(msg)
            if data['type'] == "HELLO":
                pass
            elif data['type'] == "register":
                if data['http_address'] is not in prefill_instances:
                    prefill_instances[data['http_address']] = data['http_address']
            print(f"zovlog:====> recv {data},remote_addr={remote_addr}")

def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    _listener_thread = threading.Thread(
        target = _listen_for_register,args = (hostname, port),daemon=True
    )
    _listener_thread.start()
    return _listener_thread


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        # finish prefill
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue

        # return decode
        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 36367)
    app.run(host="0.0.0.0", port=10001)
    t.join()
