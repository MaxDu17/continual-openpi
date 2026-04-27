import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type

from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

# Default websockets.connect open_timeout is 10s — too short when the server event loop is
# blocked (e.g. JAX compile on first infer). None = no limit per attempt.
_CONNECT_OPEN_TIMEOUT: Optional[float] = None
_RETRY_SLEEP_SEC = 5.0

_TRANSIENT_CONNECT_ERRORS: Tuple[Type[Exception], ...] = (
    ConnectionRefusedError,
    TimeoutError,
    OSError,
    websockets.exceptions.InvalidHandshake,
)


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            conn = None
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=_CONNECT_OPEN_TIMEOUT,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except _TRANSIENT_CONNECT_ERRORS as e:
                if conn is not None:
                    conn.close()
                logging.info(
                    "WebSocket not ready (%s: %s); retrying in %.1fs...",
                    type(e).__name__,
                    e,
                    _RETRY_SLEEP_SEC,
                )
                time.sleep(_RETRY_SLEEP_SEC)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_batch(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send a batch inference request over the same websocket connection.

        Protocol (dummy server and future real servers):
        - Client sends msgpack object: {"type": "batch_infer", "observations": [dict, ...]}
        - Server returns msgpack object containing at least:
          - "actions": ndarray [B, T, A]

        Real OpenPI servers may not implement this yet; call only when the server advertises support.
        """
        assert isinstance(observations, list), f"Expected list of observation dicts, got {type(observations)}"
        assert len(observations) >= 1, "infer_batch requires at least one observation"
        payload = {"type": "batch_infer", "observations": observations}
        data = self._packer.pack(payload)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
