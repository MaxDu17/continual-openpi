import asyncio
import dataclasses
import logging

import numpy as np
from openpi_client import msgpack_numpy
import tyro
import websockets.asyncio.server as _server
import time 

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    action_horizon: int = 8
    action_value: float = 0.0


def _infer(obs: dict, action_horizon: int, action_value: float) -> dict:
    required_keys = {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    missing = required_keys.difference(obs.keys())
    assert not missing, f"Missing observation keys: {sorted(missing)}"

    actions = np.full((action_horizon, 7), action_value, dtype=np.float32)
    actions[:, -1] = -1.0
    print("Inferring (fake), ", time.time())
    return {"actions": actions}


async def _handler(
    websocket: _server.ServerConnection,
    *,
    action_horizon: int,
    action_value: float,
) -> None:
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack({"server_type": "dummy_policy_server"}))
    while True:
        obs = msgpack_numpy.unpackb(await websocket.recv())
        action = _infer(obs, action_horizon, action_value)
        await websocket.send(packer.pack(action))


def main(args: Args) -> None:
    logging.info("Starting dummy policy server on %s:%s", args.host, args.port)
    async def run() -> None:
        async with _server.serve(
            lambda websocket: _handler(
                websocket,
                action_horizon=args.action_horizon,
                action_value=args.action_value,
            ),
            args.host,
            args.port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
