import argparse
import itertools
import logging
import os
import socket
import uuid
from contextlib import asynccontextmanager
import msgpack
import zmq
import copy
import threading
from quart import Quart, make_response, request
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Dict,List
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import aiohttp
prefill_instances = []
decode_instances = [] 
request_nums = 0
app = Quart(__name__)

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
            elif data['type'] == "register" and data['role'] == "P":
                if "http://" + data['request_address']+"/v1/completions" not in prefill_instances:
                    prefill_instances.append("http://" + data['request_address']+"/v1/completions")

            elif data["type"] == "register" and data['role'] == "D":
                if "http://" + data['request_address']+"/v1/completions" not in decode_instances:
                    decode_instances.append("http://" + data['request_address']+"/v1/completions")

            print(f"zovlog:====> recv {data},remote_addr={remote_addr},{prefill_instances = },{decode_instances = }")

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

async def send_request_to_prefill(endpoint,req_data,request_id):
    print(f"zovlog:======> proxy {endpoint = }")
    req_data_copy = copy.deepcopy(req_data)
    
    # 本地做prefill,且decode只需要pull模式,所以prefill不需要在这里知晓远程decode任何信息
    req_data_copy['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data_copy["stream"] = False
    req_data_copy["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=endpoint, json=req_data_copy, headers=headers) as response:
            if response.status == 200:
                return await response.json()
                # async for chunk_bytes in response.content.iter_chunked(1024):
                #         yield chunk_bytes
            else:
                raise RuntimeError("response.status != 200")

async def send_request_to_decode(endpoint,req_data,request_id):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=endpoint, json=req_data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
            else:
                raise RuntimeError("response.status != 200")


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    global request_nums
    req_data = await request.get_json()
    print(f"req_data = {req_data}")
    request_id = str(uuid.uuid4())
    prefill_instance_endpoint = prefill_instances[request_nums % len(prefill_instances)]
    decode_instances_endpoint = decode_instances[request_nums % len(decode_instances)]
    response_json = await send_request_to_prefill(prefill_instance_endpoint,req_data,request_id)
    print(f"-----------------------------------{response_json = }")
    kv_transfer_params = response_json.get('kv_transfer_params', {})
    if len(kv_transfer_params) == 0:
        raise RuntimeError("len(kv_transfer_params) == 0")
    req_data["kv_transfer_params"] = kv_transfer_params
    print("-------------->",req_data)

    request_nums += 1


if __name__ == '__main__':
    t = start_service_discovery("0.0.0.0", 36367)
    app.run(host="0.0.0.0", port=10001)
    t.join()
