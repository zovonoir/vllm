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
import re
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
                if data['request_address'] not in prefill_instances:
                    # prefill_instances.append(data['request_address'])
                    prefill_instances.append(data)

            elif data["type"] == "register" and data['role'] == "D":
                if data['request_address'] not in decode_instances:
                    # decode_instances.append(data['request_address'])
                    decode_instances.append(data)

            # print(f"zovlog:====> recv {data},remote_addr={remote_addr},{prefill_instances = },{decode_instances = }")

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
    print(f"zovlog ========================== send response to decode {req_data}")
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
    extract_ip_port = lambda url: re.search(r'//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', url).groups()
    req_data = await request.get_json()
    print(f"req_data = {req_data}")
    request_id = str(uuid.uuid4())
    prefill_instance_endpoint = prefill_instances[request_nums % len(prefill_instances)]
    decode_instance_endpoint = decode_instances[request_nums % len(decode_instances)]
    response_json = await send_request_to_prefill(prefill_instance_endpoint['request_address'],req_data,request_id)
    # 现在decode可以获取prefill的所有信息了
    ip, port = extract_ip_port(prefill_instance_endpoint['request_address'])
    response_json['kv_transfer_params']["do_remote_decode"] = False
    response_json['kv_transfer_params']["do_remote_prefill"] = True
    response_json['kv_transfer_params']["remote_host"] = ip
    response_json['kv_transfer_params']["remote_port"] = port # 似乎没用
    response_json['kv_transfer_params']["remote_handshake_port"] = prefill_instance_endpoint['handshake_port']

    req_data['max_tokens'] -= 1
    req_data['prompt'] += response_json['choices'][0]['text']
    # req_data['kv_transfer_params'] = {
    #     "do_remote_decode": False,
    #     "do_remote_prefill": True,
    #     "remote_engine_id": response_json['kv_transfer_params']["remote_engine_id"],
    #     "remote_block_ids": response_json['kv_transfer_params']["remote_block_ids"],
    #     "remote_host": response_json['kv_transfer_params']["remote_host"],
    #     "remote_port": response_json['kv_transfer_params']["remote_port"],
    #     "remote_handshake_port":response_json['kv_transfer_params']["remote_handshake_port"]
    # }

    # 这个kvtransfer param里面到底写了什么? 
    kv_transfer_params = response_json.get('kv_transfer_params', {})
    print(f"zovlog:========> proxy kv_transfer_params = {kv_transfer_params}")
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    generator = send_request_to_decode(decode_instance_endpoint['request_address'],req_data,request_id)
    response = await make_response(generator)
    request_nums += 1
    return response


if __name__ == '__main__':
    t = start_service_discovery("0.0.0.0", 36367)
    app.run(host="0.0.0.0", port=10001)
    t.join()





'''

 {'id': 'cmpl-dde9d301-7e84-4eb1-8735-edf196d4455a', 'object': 'text_completion', 'created': 1755246958, 'model': 'deepseek-ai/DeepSeek-R1', 'choices': [{'index': 0, 'text': ' I', 'logprobs': None, 'finish_reason': 'length', 'stop_reason': None, 'prompt_logprobs': None}], 'service_tier': None, 'system_fingerprint': None, 'usage': {'prompt_tokens': 6, 'total_tokens': 7, 'completion_tokens': 1, 'prompt_tokens_details': None}, 'kv_transfer_params': {'do_remote_prefill': True, 'do_remote_decode': False, 'remote_block_ids': [], 'remote_engine_id': '731a3c55-da99-4465-892b-0ba1ce8f1280', 'remote_host': '10.235.192.56', 'remote_port': '20005', 'tp_size': 1}}

'''

