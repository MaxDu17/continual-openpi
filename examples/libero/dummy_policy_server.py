import asyncio
import dataclasses
import logging
import time

import numpy as np
from openpi_client import msgpack_numpy
import tyro
import websockets.asyncio.server as _server

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

# this is just dummy server so we aren't actually doing the batching 
def _infer_batch(observations: list, action_horizon: int, action_value: float) -> dict:
    assert isinstance(observations, list), f"Batch observations must be a list, got {type(observations)}"
    assert len(observations) >= 1, "Batch must be non-empty"
    actions_list = []
    infer_started = time.monotonic()
    for obs in observations:
        assert isinstance(obs, dict), f"Each batch element must be a dict, got {type(obs)}"
        out = _infer(obs, action_horizon, action_value)
        actions_list.append(out["actions"])
    actions = np.stack(actions_list, axis=0)
    infer_s = time.monotonic() - infer_started
    return {
        "actions": actions,
        "batch_size": int(actions.shape[0]),
        "server_timing": {"infer_ms": infer_s * 1000.0},
    }


async def _handler(
    websocket: _server.ServerConnection,
    *,
    action_horizon: int,
    action_value: float,
) -> None:
    packer = msgpack_numpy.Packer()
    await websocket.send(
        packer.pack(
            {
                "server_type": "dummy_policy_server",
                "supports_batch_infer": True,
                "batch_protocol": "msgpack:{type: batch_infer, observations: [dict, ...]}",
            }
        )
    )
    while True:
        msg = msgpack_numpy.unpackb(await websocket.recv())
        if isinstance(msg, dict) and msg.get("type") == "batch_infer":
            observations = msg["observations"]
            action = _infer_batch(observations, action_horizon, action_value)
            await websocket.send(packer.pack(action))
            continue
        if isinstance(msg, list):
            action = _infer_batch(msg, action_horizon, action_value)
            await websocket.send(packer.pack(action))
            continue
        assert isinstance(msg, dict), f"Expected dict or list message, got {type(msg)}"
        action = _infer(msg, action_horizon, action_value)
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
