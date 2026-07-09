from abc import ABCMeta, abstractmethod

class BaseStorageBackend(metaclass=ABCMeta):
    @abstractmethod
    def get(self, filepath):
        pass
    @abstractmethod
    def get_text(self, filepath):
        pass
class MemcachedBackend(BaseStorageBackend):
    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError("Please install memcached to enable MemcachedBackend.")
        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(
            self.server_list_cfg, self.client_cfg
        )
        self._mc_buffer = mc.pyvector()
    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf
    def get_text(self, filepath):
        raise NotImplementedError
class HardDiskBackend(BaseStorageBackend):
    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, "rb") as f:
            value_buf = f.read()
        return value_buf
    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, "r") as f:
            value_buf = f.read()
        return value_buf
class LmdbBackend(BaseStorageBackend):
    def __init__(
        self,
        db_paths,
        client_keys="default",
        readonly=True,
        lock=False,
        readahead=False,
        **kwargs,
    ):
        try:
            import lmdb
        except ImportError:
            raise ImportError("Please install lmdb to enable LmdbBackend.")
        if isinstance(client_keys, str):
            client_keys = [client_keys]
        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), (
            "client_keys and db_paths should have the same length, "
            f"but received {len(client_keys)} and {len(self.db_paths)}."
        )
        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(
                path,
                readonly=readonly,
                lock=lock,
                readahead=readahead,
                map_size=8 * 1024 * 10485760,
                **kwargs,
            )
    def get(self, filepath, client_key):
        filepath = str(filepath)
        assert client_key in self._client, (
            f"client_key {client_key} is not " "in lmdb clients."
        )
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode("ascii"))
        return value_buf
    def get_text(self, filepath):
        raise NotImplementedError
class FileClient(object):
    _backends = {
        "disk": HardDiskBackend,
        "memcached": MemcachedBackend,
        "lmdb": LmdbBackend,
    }
    def __init__(self, backend="disk", **kwargs):
        if backend not in self._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones"
                f" are {list(self._backends.keys())}"
            )
        self.backend = backend
        self.client = self._backends[backend](**kwargs)
    def get(self, filepath, client_key="default"):
        if self.backend == "lmdb":
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)
    def get_text(self, filepath):
        return self.client.get_text(filepath)
