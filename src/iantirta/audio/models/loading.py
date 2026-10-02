
from pathlib import Path
from urllib.request import Request, urlopen
from safetensors import safe_open

DEMUCS_ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
DEFAULT_DEMUCS_NAMESPACE = "adefossez"

HF_ENDPOINT = "https://huggingface.co"
DEFAULT_REVISION = "main"

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

CACHE_HOME = Path("~/.cache/iantirta")
CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)


def http_download(
    url: str,
    destination: Path,
    hash_prefix: str | None = None,
    progress: bool = True,
):
    destination = Path(dst).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    tmp_dst = None

    for _ in range(tempfile.TMP_MAX):
        tmp_dst = Path(f"{destination}.{uuid.uuid4().hex}.partial")
        try:
            f = tmp_dst.open("w+b")  # noqa: SIM115
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(
            errno.EEXIST,
            "No usable temporary file name found",
        )

    try:
        req = Request(
            url,
            headers={
                "User-Agent": "iantirta.audio.hub"
            },
        )

        with urlopen(req) as response:
            meta = response.info()
            if hasattr(meta, "getheaders"):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")

            file_size = (
                int(content_length[0])
                if content_length is not None
                and len(content_length) > 0
                else None
            )

            sha256 = (
                hashlib.sha256()
                if hash_prefix is not None
                else None
            )

            with tqdm(
                total=file_size,
                disable=not progress,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while chunk := response.read(128 * 1024):
                    f.write(chunk)

                    if sha256:
                        sha256.update(chunk)

                    pbar.update(len(chunk))

            f.close()
            if sha256 is not None and hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[: len(hash_prefix)] != hash_prefix:
                    raise RuntimeError(
                        f"invalid hash value "
                        '(expected "{hash_prefix}", got "{digest}")'
                    )

        tmp_dst.replace(destination)
        return destination
    finally:
        f.close()

        if tmp_dst is not None:
            tmp_dst.unlink(missing_ok=True)


def dot_natural_key(s: str):
    """
    Sort key for state-dict names: split on `"."` and sort digits numerically and strings alphabetically. It emits a
    tuple at each point to sort ints first and strings second to avoid int-string comparison failures.
    """
    parts = []
    for part in s.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            # This will remove all trailing digit characters, as `rstrip` actually considers it as a set of chars
            text_part = part.rstrip("0123456789")
            trailing_digits = part[len(text_part) :]
            # Sort numeric suffixes numerically, so `shard_2` precedes `shard_11` for example
            if trailing_digits != "":
                parts.append((1, text_part, int(trailing_digits)))
            else:
                parts.append((1, text_part))
    return parts


def load_pretrained(
    cls,
    name,
    endpoint: str | None = None,
    use_safetensors: bool | None = None,
):
    """
    """
    if endpoint is None:
        endpoint = HF_ENDPOINT
    revision = DEFAULT_REVISION

    candidates = [
        SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_NAME, WEIGHTS_INDEX_NAME,
    ]
    model_file = None
    if (local_dir := Path(name)).isdir():
        # Local file
        for can in candidates:
            if (local_dir / can).is_file():
                model_file = local_dir / can
                break
        if model_file is None:
            raise FileNotFoundError
    else:
        # Remote file
        if use_safetensors is not False:
            filename = SAFE_WEIGHTS_NAME
        else:
            filename = WEIGHTS_NAME
        
        # lets do simple
        if "/" in name:
            full_repo = name
        else:
            # demucs
            if name == 'htdemucs':
                name = 'HTDemucs'
            elif name.startswith('htdemucs_'):
                name = 'HTDemucs-' + name[len('htdemucs_'):]
            else:
                name = 'Demucs-' + name
            full_repo = DEFAULT_DEMUCS_NAMESPACE + "/" + name

        url = (
            endpoint.rstrip("/") +
            f"/{full_repo}/resolve/{revision}/{filename}"
        )

        local_folder_name = "--".join("models", full_repo.split("/"))
        local_dir = CACHE_HOME / local_folder_name
        blob_path = local_dir / "blobs"  # / etag
        try:
            if not (tag_path := CACHE_HOME / "CACHEDIR.TAG").exists():
                tag_path.write_text(CACHEDIR_TAG_CONTENT)
        except OSError:
            pass

        destination_path = local_dir / filename
        model_file = http_download(url, destination_path)

    state_dict = {}
    with safe_open(model_file, framework="pt") as pkg
        for k in pkg.keys():
            state_dict[k] = pkg.get_slice(k)

        state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
        for original_key, tensor in state_dict:
       