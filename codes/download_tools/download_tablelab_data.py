#!/usr/bin/env python3
"""下载 TableLab 指定的四份数据；保留原文件字节，校验后生成上传用 ZIP。

Python 3.11+。首次运行前安装 requirements_download.txt。
用法：python download_tablelab_data.py --login
只下载 Adult：python download_tablelab_data.py --only adult
默认输出到本脚本所在目录的 data/incoming，不依赖终端当前目录。
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import warnings
import zipfile

VERSION = "1.0.0"
MAX_BYTES = 100_000_000  # 与 TableLab v0.1 导入上限一致。
TOKEN_PAGE = "https://www.kaggle.com/settings/api"
HOUSE_RULES = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules"
ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
MANIFEST_NAME = "download_manifest.json"
BUNDLE_NAME = "tablelab_datasets_upload.zip"


class DownloadError(RuntimeError):
    """预期的下载或文件校验失败；调用者必须停止并报告。"""


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    archive_name: str
    page: str
    handle: str
    files: tuple[str, ...]


DATASETS = {
    "adult": Dataset(
        "adult", "UCI Adult 收入数据", "adult.zip",
        "https://archive.ics.uci.edu/dataset/2/adult", "",
        ("adult.data", "adult.test", "adult.names"),
    ),
    "nba": Dataset(
        "nba", "NBA 球员数据", "nba.zip",
        "https://www.kaggle.com/datasets/justinas/nba-players-data",
        "justinas/nba-players-data", ("all_seasons.csv",),
    ),
    "students": Dataset(
        "students", "学生成绩数据", "students.zip",
        "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams",
        "spscientist/students-performance-in-exams", ("StudentsPerformance.csv",),
    ),
    "house": Dataset(
        "house", "Kaggle 房价数据", "house-prices.zip",
        "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data",
        "house-prices-advanced-regression-techniques",
        ("train.csv", "test.csv", "data_description.txt", "sample_submission.csv"),
    ),
}

# 检查 TableLab 实际使用的字段，不修改原始 CSV，也不假定固定样本数。
REQUIRED_COLUMNS = {
    "all_seasons.csv": {
        "player_name", "season", "age", "player_height", "player_weight",
        "draft_year", "pts", "reb", "ast", "gp",
    },
    "StudentsPerformance.csv": {
        "gender", "race/ethnicity", "parental level of education", "lunch",
        "test preparation course", "math score", "reading score", "writing score",
    },
    "train.csv": {
        "Id", "SalePrice", "OverallQual", "GrLivArea", "YearBuilt",
        "TotalBsmtSF", "GarageCars", "Neighborhood",
    },
    "test.csv": {"Id", "OverallQual", "GrLivArea", "YearBuilt"},
    "sample_submission.csv": {"Id", "SalePrice"},
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_payload(name: str, payload: bytes) -> dict:
    """验证下载边界上的结构；检查不改变源文件。"""
    if not payload:
        raise DownloadError(f"{name} 是空文件。")
    info = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    if name in ("adult.data", "adult.test"):
        text = payload.decode("utf-8-sig")
        lines = (line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("|"))
        rows = 0
        for row in csv.reader(lines, skipinitialspace=True, strict=True):
            rows += 1
            if len(row) != 15 or row[-1].strip().rstrip(".") not in {"<=50K", ">50K"}:
                raise DownloadError(f"{name} 第 {rows} 条记录不符合 Adult 的 15 列及收入标签格式。")
        if rows == 0:
            raise DownloadError(f"{name} 没有数据行。")
        info.update(rows=rows, column_count=15)
    elif name.endswith(".csv"):
        reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header):
            raise DownloadError(f"{name} 缺少表头或有重复字段。")
        missing = REQUIRED_COLUMNS[name] - set(header)
        if missing:
            raise DownloadError(f"{name} 缺少所需字段：{', '.join(sorted(missing))}。")
        if name == "test.csv" and "SalePrice" in header:
            raise DownloadError("房价 test.csv 意外含 SalePrice；请核实是否下载了原比赛文件。")
        rows = 0
        for row in reader:
            if not row:  # 空白物理行不是一个样本。
                continue
            if len(row) != len(header):
                raise DownloadError(f"{name} CSV 第 {reader.line_num} 行列数与表头不一致。")
            rows += 1
        if rows == 0:
            raise DownloadError(f"{name} 没有数据行。")
        info.update(rows=rows, columns=header)
    return info


def inspect_archive(path: Path, dataset: Dataset) -> dict:
    """检查 ZIP、CRC、必需文件与字段。这里不解压任意路径。"""
    if path.stat().st_size > MAX_BYTES or not zipfile.is_zipfile(path):
        raise DownloadError(f"{path.name} 不是有效 ZIP，或超过 100 MB；不能把登录页当成数据。")
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if sum(item.file_size for item in members) > MAX_BYTES:
            raise DownloadError(f"{path.name} 解压后超过 100 MB 上限。")
        for item in members:
            p = PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in item.filename:
                raise DownloadError(f"{path.name} 含不安全的成员路径。")
            if item.flag_bits & 1 or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise DownloadError(f"{path.name} 含加密文件或符号链接。")
        broken = archive.testzip()
        if broken is not None:
            raise DownloadError(f"{path.name} 的 CRC 校验失败。")
        file_info = {}
        for name in dataset.files:
            matches = [m for m in members if PurePosixPath(m.filename).name == name]
            if len(matches) != 1:
                raise DownloadError(f"{path.name} 必须且只能包含一个 {name}，实际找到 {len(matches)} 个。")
            file_info[name] = inspect_payload(name, archive.read(matches[0]))
    return {"archive_sha256": sha256_file(path), "archive_bytes": path.stat().st_size, "files": file_info}


def download_adult(destination: Path) -> None:
    request = Request(ADULT_URL, headers={"User-Agent": f"TableLab-data-downloader/{VERSION}"})
    try:
        with urlopen(request, timeout=90) as response, destination.open("wb") as out:
            if response.headers.get_content_type() in {"text/html", "application/json"}:
                raise DownloadError("UCI 返回网页或 JSON，而不是数据 ZIP。")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise DownloadError("UCI 下载超过 100 MB 上限。")
                out.write(chunk)
    except HTTPError as exc:
        raise DownloadError(f"UCI 下载失败：HTTP {exc.code}。请检查原始数据页面。") from exc
    except URLError as exc:
        raise DownloadError("无法连接 UCI；请检查网络、DNS、代理和 Python 的 HTTPS 证书。不会关闭证书校验。") from exc


def download_kaggle(dataset: Dataset, output_dir: Path) -> Path:
    import kagglehub
    import requests
    from kagglehub.exceptions import CredentialError, UnauthenticatedError

    try:
        if dataset.key == "house":
            source = kagglehub.competition_download(
                dataset.handle, output_dir=str(output_dir), force_download=True,
            )
        else:
            source = kagglehub.dataset_download(
                dataset.handle, output_dir=str(output_dir), force_download=True,
            )
    except (CredentialError, UnauthenticatedError) as exc:
        raise DownloadError(
            "Kaggle 未认证。使用 --login 重新运行，在本机隐藏输入 API Token；不要发送账号密码或 Token。"
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "未知"
        if status in (401, 403):
            hint = f"请核对 Token，并在浏览器确认房价比赛下载权限及规则：{HOUSE_RULES}" if dataset.key == "house" else "请核对 Token 和该数据集的访问权限。"
        elif status == 429:
            hint = "Kaggle 限流；停止重复请求，稍后手动重跑。已完成的数据不会重新下载。"
        else:
            hint = "请检查原始页面是否可访问；本脚本不切换到第三方镜像。"
        raise DownloadError(f"{dataset.label}下载失败：HTTP {status}。{hint}") from exc
    except requests.RequestException as exc:
        raise DownloadError(f"{dataset.label}的 Kaggle 网络请求失败；请检查网络、代理或证书。") from exc
    return Path(source)


def package_kaggle_files(source: Path, destination: Path, dataset: Dataset) -> None:
    """重新封装指定文件；文件字节不变，不把缓存目录或凭据加入 ZIP。"""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in dataset.files:
            matches = [p for p in source.rglob(name) if p.is_file()]
            if len(matches) != 1:
                raise DownloadError(f"Kaggle 返回目录中应有一个 {name}，实际找到 {len(matches)} 个。")
            path = matches[0]
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                raise DownloadError(f"{name} 指向下载目录之外，拒绝打包。")
            if path.stat().st_size > MAX_BYTES:
                raise DownloadError(f"{name} 超过 100 MB 上限。")
            archive.write(path, arcname=name)


def prompt_token() -> None:
    """Token 只保存在当前进程环境中，不写入项目或下载清单。"""
    if not sys.stdin.isatty():
        raise DownloadError("--login 需要在终端运行，不能在不支持隐藏输入的窗口中粘贴 Token。")
    print(f"在浏览器登录 Kaggle，从此页生成 API Token：{TOKEN_PAGE}")
    print("这里输入的是 API Token，不是账号密码，也不是整个 kaggle.json 文件。")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        token = getpass.getpass("粘贴 Kaggle API Token 后按回车（输入不会显示）：").strip()
    if not token or any(char.isspace() for char in token):
        raise DownloadError("Token 不能为空或包含空白；请复制完整 Token，不要带引号。")
    os.environ["KAGGLE_API_TOKEN"] = token


def save_manifest(path: Path, manifest: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def cached_record(output: Path, dataset: Dataset, manifest: dict) -> dict | None:
    path = output / dataset.archive_name
    if not path.exists():
        return None
    record = manifest["datasets"].get(dataset.key)
    if record is None or record["source_page"] != dataset.page:
        raise DownloadError(f"已有 {path.name}，但没有对应来源记录。请移开它，或用 --force 从原来源重新下载。")
    info = inspect_archive(path, dataset)
    if info["archive_sha256"] != record["archive_sha256"]:
        raise DownloadError(f"{path.name} 与下载清单的 SHA-256 不一致。请核实后用 --force 重新下载。")
    return record


def prepare_dataset(output: Path, dataset: Dataset, manifest: dict) -> None:
    # 临时目录与目标位于同一文件系统；成功校验后再替换，失败不留下伪完整 ZIP。
    with tempfile.TemporaryDirectory(prefix=f".{dataset.key}-", dir=output) as temporary:
        work = Path(temporary)
        archive_path = work / dataset.archive_name
        if dataset.key == "adult":
            download_adult(archive_path)
            packaging = "original_source_zip"
            downloader = "Python standard library urllib"
        else:
            source = download_kaggle(dataset, work / "downloaded_files")
            package_kaggle_files(source, archive_path, dataset)
            packaging = "repacked_source_files_unchanged"
            downloader = f"kagglehub=={importlib.metadata.version('kagglehub')}"
        info = inspect_archive(archive_path, dataset)
        record = {
            "dataset": dataset.key,
            "archive_name": dataset.archive_name,
            "source_page": dataset.page,
            "source_handle_or_url": dataset.handle or ADULT_URL,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "downloader": downloader,
            "packaging": packaging,
            "version_policy": "Source version acquired at download time; retain file hashes to identify this copy.",
            **info,
        }
        archive_path.replace(output / dataset.archive_name)
        manifest["datasets"][dataset.key] = record
        save_manifest(output / MANIFEST_NAME, manifest)
    print(f"已完成：{dataset.label} → {dataset.archive_name}（{info['archive_bytes']:,} 字节）")


def make_bundle(output: Path, manifest: dict) -> Path:
    # 白名单打包；不会包含 .env、kaggle.json、模型权重或其他项目文件。
    selected = {}
    for key, dataset in DATASETS.items():
        record = cached_record(output, dataset, manifest)
        if record is None:
            raise DownloadError(f"缺少 {dataset.archive_name}，不能生成四数据集完整上传包。")
        selected[key] = record
    bundle = output / BUNDLE_NAME
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        for dataset in DATASETS.values():
            archive.write(output / dataset.archive_name, arcname=dataset.archive_name)
        archive.writestr(
            MANIFEST_NAME,
            json.dumps({"format_version": 1, "script_version": VERSION, "datasets": selected},
                       ensure_ascii=False, indent=2) + "\n",
        )
    temporary.replace(bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "incoming")
    parser.add_argument("--only", choices=("all", *DATASETS), default="all")
    parser.add_argument("--login", action="store_true", help="在本机终端隐藏输入 Kaggle API Token，只在本次进程使用")
    parser.add_argument("--force", action="store_true", help="明确重新下载并替换选定数据；未校验的新文件不会覆盖原文件")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        raise DownloadError("请使用 Python 3.11 或更新版本。")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / MANIFEST_NAME
    manifest = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {
        "format_version": 1, "script_version": VERSION, "datasets": {},
    }
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("datasets"), dict):
        raise DownloadError("download_manifest.json 格式不符合本脚本的记录格式。")
    datasets = list(DATASETS.values()) if args.only == "all" else [DATASETS[args.only]]
    pending = []
    for dataset in datasets:
        if not args.force and cached_record(output, dataset, manifest) is not None:
            print(f"已校验，复用本地文件：{dataset.archive_name}")
        else:
            pending.append(dataset)
    if any(dataset.key != "adult" for dataset in pending):
        if importlib.util.find_spec("kagglehub") is None:
            raise DownloadError("缺少 kagglehub。先运行：python -m pip install -r requirements_download.txt")
        if args.login:
            prompt_token()
        if any(dataset.key == "house" for dataset in pending):
            print(f"房价下载需账号具备访问权限；比赛规则由你在浏览器确认：{HOUSE_RULES}")
    for dataset in pending:
        print(f"正在下载：{dataset.label}", flush=True)
        prepare_dataset(output, dataset, manifest)
    if args.only == "all":
        bundle = make_bundle(output, manifest)
        print(f"\n四份数据均已校验。上传给助手的文件：\n{bundle}")
        print("只需上传这个 ZIP；不要上传 Token、密码或 kaggle.json。")
    else:
        print(f"\n所选数据已准备：{output / datasets[0].archive_name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (DownloadError, OSError, zipfile.BadZipFile, csv.Error, UnicodeError,
            json.JSONDecodeError, getpass.GetPassWarning, EOFError) as error:
        print(f"\n失败，任务已停止：{error}", file=sys.stderr)
        print("已成功下载并登记的文件仍保留。修正问题后重跑；不会用其他数据替代。", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已取消。完成且已校验的数据仍保留。", file=sys.stderr)
        sys.exit(130)
