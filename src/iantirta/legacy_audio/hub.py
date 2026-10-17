
import os
from pathlib import Path
from urllib.request import Request, urlopen
from tqdm import tqdm
import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import warnings
import zipfile

READ_DATA_CHUNK = 128 * 1024


class Hub:
    def download(
        self,
        url: str,
        dst: str | Path,
        hash_prefix: str | None = None,
        progress: bool = True,
    ) -> Path:
        dst = Path(dst).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)

        tmp_dst = None

        for _ in range(tempfile.TMP_MAX):
            tmp_dst = Path(f"{dst}.{uuid.uuid4().hex}.partial")
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
                            f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                        )

            tmp_dst.replace(dst)
            return dst
        finally:
            f.close()

            if tmp_dst is not None:
                tmp_dst.unlink(missing_ok=True)

    
class TorchHub(Hub):
    pass

class HFHub(Hub):
    ENDPOINT = "https://huggingface.co"
    CACHE_DIR = Path("~/.cache/iantirta/hub").expanduser()
    
    def build_url(self, repo: str, filename: str, revision: str) -> str:
        return (
            f"{self.ENDPOINT}/{repo}"
            f"/resolve/{revision}/{filename}"
        )

    def download(
        self,
        repo: str,
        filename: str,
        *,
        revision: str = "main",
    ) -> Path:
        url = self.build_url(repo, filename, revision)

        # simplest possible implementation first
        destination = self.CACHE_DIR / repo / revision / filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            return destination

        super().download(url, destination)
        return destination
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        CONFIG = "config.json"
        SAFE_WEIGHTS = "model.safetensors"

print(HFHub().download("facebook/mms-1b-all","model.safetensors"))
exit()
import huggingface_hub.file_download as fd
import huggingface_hub.constants as c

repo_id = "facebook/mms-1b-all"
repo_type = "model"
cache_dir = c.HF_HUB_CACHE
cache_dir = str(Path(cache_dir).expanduser().resolve())

filename ="config.json"
relative_filename = os.path.join(*filename.split("/"))
print("relatice", relative_filename)

print(cache_dir)
print(c.HF_HUB_ETAG_TIMEOUT)
print(c.DEFAULT_REVISION)

hf_headers = fd.build_hf_headers(
    token=None,
    library_name=None,
    library_version=None,
    user_agent=None,
    headers=None,
)
print(hf_headers)


(url_to_download, etag, commit_hash, expected_size, xet_file_data, head_call_error) = (
    fd._get_metadata_or_catch_error(
        repo_id="facebook/mms-1b-all",
        filename="config.json",
        repo_type="model",
        revision="main",
        etag_timeout=10,
        endpoint=None,
        token=None,
        local_files_only=False,
        headers=None,
    )
)

storage_folder = os.path.join(cache_dir, fd.repo_folder_name(repo_id=repo_id, repo_type=repo_type))
print("srorage", storage_folder)
blob_path = os.path.join(storage_folder, "blobs", etag)
print("blob", blob_path)
pointer_path = fd._get_pointer_path(storage_folder, commit_hash, relative_filename)
print("pointer path", pointer_path)

from pprint import pprint
