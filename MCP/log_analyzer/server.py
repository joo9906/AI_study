import re
import json
from pathlib import Path
from collections import Counter, defaultdict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("log-analyzer")

# ─────────────────────── 내부 유틸 ───────────────────────────

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}

_PATTERNS = {
    "python": re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]?\d*)"
        r"\s+(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)"
        r"\s+(?P<logger>\S+)"
        r"\s+(?P<message>.+)"
    ),
    "common": re.compile(
        r"(?P<timestamp>\S+)\s+(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\s+(?P<message>.+)"
    ),
}


def _read_lines(file_path: str, offset: int = 0, limit: int | None = None) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    lines = lines[offset:]
    if limit is not None:
        lines = lines[:limit]
    return [l.rstrip("\n") for l in lines]


def _parse_line(line: str) -> dict:
    stripped = line.strip()
    if stripped.startswith("{"):
        try:
            return {"_format": "json", "_raw": line, **json.loads(stripped)}
        except json.JSONDecodeError:
            pass
    for fmt, pattern in _PATTERNS.items():
        m = pattern.match(stripped)
        if m:
            return {"_format": fmt, "_raw": line, **m.groupdict()}
    return {"_format": "unknown", "_raw": line}


def _extract_level(parsed: dict) -> str | None:
    for key in ("level", "severity", "log_level", "lvl"):
        val = parsed.get(key)
        if isinstance(val, str) and val.upper() in LOG_LEVELS:
            return val.upper()
    raw = parsed.get("_raw", "")
    for lvl in LOG_LEVELS:
        if lvl in raw.upper():
            return lvl
    return None


def _normalize_msg(msg: str) -> str:
    """숫자·UUID를 플레이스홀더로 치환해 메시지 그룹화."""
    msg = re.sub(r"\b[0-9a-f\-]{8,}\b", "<ID>", msg, flags=re.I)
    msg = re.sub(r"\d+", "N", msg)
    return msg.strip()


# ─────────────────────── MCP Tools ───────────────────────────

@mcp.tool()
async def read_log(
    file_path: str,
    offset: int = 0,
    limit: int = 200,
) -> str:
    """
    Read a log file and return its contents.
    로그 파일을 읽어 내용을 반환합니다.

    Args:
        file_path (str): 로그 파일 경로 (절대 또는 상대 경로).
        offset (int, optional): 읽기 시작 줄 번호 (0-indexed). Defaults to 0.
        limit (int, optional): 최대 읽을 줄 수. Defaults to 200.
    """
    try:
        lines = _read_lines(file_path, offset=offset, limit=limit)
    except FileNotFoundError as e:
        return str(e)

    return f"[{len(lines)}줄 반환 | offset={offset}]\n\n" + "\n".join(lines)


@mcp.tool()
async def get_log_stats(
    file_path: str,
) -> str:
    """
    Get overall statistics of a log file (level distribution, timestamp range, total lines).
    로그 파일 전체 통계를 반환합니다 (레벨 분포, 타임스탬프 범위, 총 라인 수).

    Args:
        file_path (str): 로그 파일 경로.
    """
    try:
        lines = _read_lines(file_path)
    except FileNotFoundError as e:
        return str(e)

    level_counter: Counter = Counter()
    timestamps: list[str] = []

    for line in lines:
        parsed = _parse_line(line)
        lvl = _extract_level(parsed)
        if lvl:
            level_counter[lvl] += 1
        ts = parsed.get("timestamp")
        if ts:
            timestamps.append(ts)

    stats = {
        "총 라인 수": len(lines),
        "레벨 분포": dict(level_counter.most_common()),
        "레벨 미상 라인": len(lines) - sum(level_counter.values()),
        "첫 타임스탬프": timestamps[0] if timestamps else "파싱 불가",
        "마지막 타임스탬프": timestamps[-1] if timestamps else "파싱 불가",
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)


@mcp.tool()
async def analyze_errors(
    file_path: str,
    levels: list[str] | None = None,
    top_n: int = 10,
) -> str:
    """
    Extract and analyze ERROR/CRITICAL/WARNING entries, returning frequency and patterns.
    오류 항목을 추출하고 빈도·패턴을 분석합니다.

    Args:
        file_path (str): 로그 파일 경로.
        levels (list[str], optional): 분석할 레벨 목록. Defaults to ["ERROR", "CRITICAL"].
        top_n (int, optional): 상위 N개 패턴을 출력합니다. Defaults to 10.
    """
    if levels is None:
        levels = ["ERROR", "CRITICAL"]
    target_levels = {l.upper() for l in levels}

    try:
        lines = _read_lines(file_path)
    except FileNotFoundError as e:
        return str(e)

    matched: list[dict] = []
    for i, line in enumerate(lines, 1):
        parsed = _parse_line(line)
        lvl = _extract_level(parsed)
        if lvl in target_levels:
            matched.append({"line_no": i, "level": lvl, "raw": line})

    if not matched:
        return f"대상 레벨({', '.join(target_levels)}) 항목이 없습니다."

    pattern_counter: Counter = Counter(_normalize_msg(m["raw"]) for m in matched)

    result = {
        "총 오류 수": len(matched),
        "레벨별 분포": dict(Counter(m["level"] for m in matched)),
        f"상위 {top_n} 오류 패턴": [
            {"패턴": pat, "횟수": cnt} for pat, cnt in pattern_counter.most_common(top_n)
        ],
        "최근 5건": [m["raw"] for m in matched[-5:]],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_logs(
    file_path: str,
    query: str,
    is_regex: bool = False,
    case_sensitive: bool = False,
    limit: int = 50,
) -> str:
    """
    Search a log file for a keyword or regex pattern and return matching lines.
    로그 파일에서 키워드나 정규식 패턴을 검색하고 매칭된 줄을 반환합니다.

    Args:
        file_path (str): 로그 파일 경로.
        query (str): 검색할 텍스트 또는 정규식 패턴.
        is_regex (bool, optional): True이면 정규식으로 처리. Defaults to False.
        case_sensitive (bool, optional): True이면 대소문자 구분. Defaults to False.
        limit (int, optional): 최대 반환 결과 수. Defaults to 50.
    """
    try:
        lines = _read_lines(file_path)
    except FileNotFoundError as e:
        return str(e)

    flags = 0 if case_sensitive else re.IGNORECASE
    if is_regex:
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            return f"정규식 오류: {e}"
        match_fn = lambda line: bool(pattern.search(line))
    else:
        q = query if case_sensitive else query.lower()
        match_fn = lambda line: q in (line if case_sensitive else line.lower())

    results = []
    for i, line in enumerate(lines, 1):
        if match_fn(line):
            results.append(f"[{i:6d}] {line}")
        if len(results) >= limit:
            break

    if not results:
        return f"'{query}' 검색 결과 없음."

    return f"'{query}' 검색 결과: {len(results)}건 (최대 {limit})\n\n" + "\n".join(results)


@mcp.tool()
async def get_recent_entries(
    file_path: str,
    n: int = 50,
) -> str:
    """
    Return the last N lines of a log file (tail).
    로그 파일의 마지막 N줄을 반환합니다 (tail).

    Args:
        file_path (str): 로그 파일 경로.
        n (int, optional): 가져올 줄 수. Defaults to 50.
    """
    try:
        lines = _read_lines(file_path)
    except FileNotFoundError as e:
        return str(e)

    recent = lines[-n:]
    return f"[마지막 {len(recent)}줄]\n\n" + "\n".join(recent)


@mcp.tool()
async def find_anomalies(
    file_path: str,
    repeat_threshold: int = 5,
) -> str:
    """
    Detect anomalous patterns such as repeated errors or error spike zones.
    반복 오류나 오류 급증 구간 등 이상 패턴을 탐지합니다.

    Args:
        file_path (str): 로그 파일 경로.
        repeat_threshold (int, optional): 동일 메시지가 N회 이상이면 이상으로 분류. Defaults to 5.
    """
    try:
        lines = _read_lines(file_path)
    except FileNotFoundError as e:
        return str(e)

    pattern_lines: dict[str, list[int]] = defaultdict(list)
    error_bucket: Counter = Counter()

    for i, line in enumerate(lines, 1):
        parsed = _parse_line(line)
        lvl = _extract_level(parsed)
        pattern_lines[_normalize_msg(line)].append(i)
        if lvl in ("ERROR", "CRITICAL", "WARNING", "WARN"):
            error_bucket[(i - 1) // 100] += 1

    repeated = sorted(
        [
            {"패턴": pat, "횟수": len(lnos), "첫 등장": lnos[0], "마지막 등장": lnos[-1]}
            for pat, lnos in pattern_lines.items()
            if len(lnos) >= repeat_threshold
        ],
        key=lambda x: x["횟수"],
        reverse=True,
    )

    spike_buckets = [
        {"라인 구간": f"{b*100+1}~{b*100+100}", "오류 수": cnt}
        for b, cnt in error_bucket.most_common(5)
        if cnt >= 3
    ]

    result = {
        f"반복 패턴 (≥{repeat_threshold}회)": repeated[:20],
        "오류 밀집 구간 (상위 5)": spike_buckets,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ─────────────────────── 진입점 ──────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
