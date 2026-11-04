# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read the tensor names and shapes of a PyTorch ``.pth`` / ``.pt`` zip checkpoint
without unpickling it.

The archive's ``data.pkl`` member is decoded opcode by opcode with
:func:`pickletools.genops` and replayed on a symbolic stack: a global is only a
name, a call is never made, and the only calls that are understood are the
tensor rebuilders (``torch._utils._rebuild_tensor``, ``_rebuild_tensor_v2``,
``_rebuild_parameter``) and ``collections.OrderedDict``, which are evaluated as
data. No module is imported and no object is constructed, so nothing in the
file can run. Only ``str`` keys and ``str`` global operands are ever hashed or
converted, and at most ``PICKLE_MAX_BYTES`` of the member are read whatever its
size fields say.
"""

import contextlib
import io
import pickletools
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from localm.debuglog import logger

PICKLE_MAX_BYTES = 8 * 1024 * 1024
_MAX_OPS = 2_000_000
CAPPABLE_ZIP_METHODS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_MAX_ZIP_READ = 16 * 1024 * 1024
_WRAPPER_KEYS = ("params_ema", "params", "state_dict", "model", "model_state_dict")
_TENSOR_REBUILDERS = {("torch._utils", "_rebuild_tensor"), ("torch._utils", "_rebuild_tensor_v2")}
_PARAMETER_REBUILDERS = {("torch._utils", "_rebuild_parameter"),
                         ("torch._utils", "_rebuild_parameter_with_state")}


class _Unsupported(Exception):
    """The pickle uses an opcode or shape this reader does not model."""


class _Global:
    __slots__ = ("module", "name")

    def __init__(self, module: str, name: str):
        self.module = module
        self.name = name


class _Tensor:
    __slots__ = ("shape",)

    def __init__(self, shape: tuple):
        self.shape = shape


class _Opaque:
    __slots__ = ()


_OPAQUE = _Opaque()
_MARK = object()

_PUSH_VALUE = frozenset({
    "INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4",
    "STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "SHORT_BINUNICODE",
    "BINUNICODE", "BINUNICODE8", "FLOAT", "BINFLOAT", "BINBYTES", "SHORT_BINBYTES",
    "BINBYTES8", "BYTEARRAY8",
})
_PUT = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_GET = frozenset({"GET", "BINGET", "LONG_BINGET"})


def _reduce(func, args):
    if isinstance(func, _Global):
        key = (func.module, func.name)
        if key in _TENSOR_REBUILDERS and isinstance(args, tuple) and len(args) >= 3:
            shape = args[2]
            if (isinstance(shape, tuple)
                    and all(isinstance(d, int) and not isinstance(d, bool) for d in shape)):
                return _Tensor(shape)
            return _OPAQUE
        if key in _PARAMETER_REBUILDERS and isinstance(args, tuple) and args:
            return args[0]
        if key == ("collections", "OrderedDict") and args == ():
            return {}
    return _OPAQUE


def _replay(data: bytes):
    """The object *data* pickles, with tensors as :class:`_Tensor`, dicts, lists
    and tuples as their Python types and everything else opaque."""
    stack: list = []
    marks: list = []
    memo: dict = {}
    ops = 0
    for opcode, arg, _pos in pickletools.genops(data):
        ops += 1
        if ops > _MAX_OPS:
            raise _Unsupported("too many opcodes")
        name = opcode.name
        if name in ("PROTO", "FRAME", "NOP"):
            continue
        if name == "STOP":
            return stack[-1]
        if name == "MARK":
            marks.append(stack)
            stack = []
        elif name in _PUSH_VALUE:
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1])
            items = tuple(stack[-n:])
            del stack[-n:]
            stack.append(items)
        elif name == "TUPLE":
            items = tuple(stack)
            stack = marks.pop()
            stack.append(items)
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "LIST":
            items = list(stack)
            stack = marks.pop()
            stack.append(items)
        elif name == "APPEND":
            value = stack.pop()
            target = stack[-1]
            if isinstance(target, list):
                target.append(value)
        elif name == "APPENDS":
            items = stack
            stack = marks.pop()
            if isinstance(stack[-1], list):
                stack[-1].extend(items)
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "DICT":
            items = stack
            stack = marks.pop()
            stack.append(_pairs_to_dict(items))
        elif name == "SETITEM":
            value = stack.pop()
            key = stack.pop()
            _set_item(stack[-1], key, value)
        elif name == "SETITEMS":
            items = stack
            stack = marks.pop()
            target = stack[-1]
            for i in range(0, len(items) - 1, 2):
                _set_item(target, items[i], items[i + 1])
        elif name == "EMPTY_SET":
            stack.append(_OPAQUE)
        elif name == "FROZENSET":
            stack = marks.pop()
            stack.append(_OPAQUE)
        elif name == "ADDITEMS":
            stack = marks.pop()
        elif name == "GLOBAL":
            module, _, attr = str(arg).partition(" ")
            stack.append(_Global(module, attr))
        elif name == "STACK_GLOBAL":
            attr = stack.pop()
            module = stack.pop()
            if not (isinstance(module, str) and isinstance(attr, str)):
                raise _Unsupported("STACK_GLOBAL operands")
            stack.append(_Global(module, attr))
        elif name == "REDUCE":
            args = stack.pop()
            func = stack.pop()
            stack.append(_reduce(func, args))
        elif name in ("NEWOBJ", "NEWOBJ_EX"):
            del stack[-(2 if name == "NEWOBJ" else 3):]
            stack.append(_OPAQUE)
        elif name in ("OBJ", "INST"):
            stack = marks.pop()
            stack.append(_OPAQUE)
        elif name == "BUILD":
            stack.pop()
        elif name in ("BINPERSID", "PERSID"):
            stack.pop()
            stack.append(_OPAQUE)
        elif name == "POP":
            stack.pop()
        elif name == "POP_MARK":
            stack = marks.pop()
        elif name == "DUP":
            stack.append(stack[-1])
        elif name in _PUT:
            memo[arg] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in _GET:
            stack.append(memo[arg])
        else:
            raise _Unsupported(name)
    raise _Unsupported("no STOP")


def _set_item(target, key, value) -> None:
    # Keeps str keys only and never hashes any other key. See
    # test_hostile_pickle_returns_none_in_a_child.
    if isinstance(target, dict) and isinstance(key, str):
        target[key] = value


def _pairs_to_dict(items: list) -> dict:
    out: dict = {}
    for i in range(0, len(items) - 1, 2):
        _set_item(out, items[i], items[i + 1])
    return out


def _state_dict(obj) -> Optional[dict]:
    """The ``{name: shape}`` mapping of the tensors in a state dict *obj*, looking
    one level into the usual wrapper keys (``params_ema``, ``params``,
    ``state_dict``, ``model``); None when *obj* holds no tensors."""
    if not isinstance(obj, dict):
        return None
    tensors = {str(k): v.shape for k, v in obj.items() if isinstance(v, _Tensor)}
    if tensors:
        return tensors
    for key in _WRAPPER_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            tensors = {str(k): v.shape for k, v in inner.items() if isinstance(v, _Tensor)}
            if tensors:
                return tensors
    return None


class _CappedReads(io.BufferedReader):
    """A read-only file that raises OSError instead of returning more than
    ``_MAX_ZIP_READ`` bytes from one read, and takes no more than that plus one
    from the disk for it."""

    def read(self, size: Optional[int] = -1, /) -> bytes:
        if size is None or size < 0 or size > _MAX_ZIP_READ:
            data = super().read(_MAX_ZIP_READ + 1)
            if len(data) > _MAX_ZIP_READ:
                raise OSError(f"a single read of more than {_MAX_ZIP_READ} bytes")
            return data
        return super().read(size)


@contextlib.contextmanager
def open_capped_zip(path: Path) -> Iterator[zipfile.ZipFile]:
    """*path* as a :class:`zipfile.ZipFile`, reading no more than ``_MAX_ZIP_READ``
    bytes at a time. Raises OSError for a central directory larger than that and
    whatever ``ZipFile`` raises for a file that is not a zip."""
    with _CappedReads(io.FileIO(path)) as fh, zipfile.ZipFile(fh) as zf:
        yield zf


def read_member_capped(zf: zipfile.ZipFile, member: zipfile.ZipInfo,
                       limit: int) -> Optional[bytes]:
    """The bytes of *member* of *zf*. None, reading nothing, when it declares more
    than *limit* bytes or is packed with anything but stored or deflate. Reads and
    inflates about *limit* bytes at most, whatever its size fields claim. Raises
    what :mod:`zipfile` and :mod:`zlib` raise for a damaged member, including one
    whose bytes do not match its CRC."""
    if member.file_size > limit or member.compress_type not in CAPPABLE_ZIP_METHODS:
        return None
    with zf.open(member) as fh:
        return fh.read(member.file_size)


def read_checkpoint_tensors(path: Path) -> Optional[dict]:
    """``{tensor name: shape}`` for the zip-format ``.pth`` / ``.pt`` checkpoint at
    *path*, read from its ``data.pkl`` member (at most ``PICKLE_MAX_BYTES``)
    without importing or calling anything the pickle names. None for any other
    file (a legacy non-zip checkpoint included), a checkpoint with no tensors, one
    whose pickle uses constructs this reader does not model or whose member is
    packed with a method other than stored or deflate, and an unreadable file
    (logged at debug level). Never raises."""
    path = Path(path)
    if path.suffix.lower() not in (".pth", ".pt"):
        return None
    try:
        with open_capped_zip(path) as zf:
            member = next((i for i in zf.infolist()
                           if i.filename == "data.pkl" or i.filename.endswith("/data.pkl")), None)
            blob = None if member is None else read_member_capped(zf, member, PICKLE_MAX_BYTES)
        return None if blob is None else _state_dict(_replay(blob))
    except Exception as e:
        logger.debug("checkpoint tensor probe could not read %s: %s: %s",
                     path, type(e).__name__, e)
        return None

