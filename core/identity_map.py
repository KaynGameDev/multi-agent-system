from __future__ import annotations

import re
from typing import Any

IDENTITY_MAP = {
    "@Terry - YeJunJie": {"google_name": "叶俊杰", "sheet_name": "叶俊杰", "job_title": "产品", "email": "yejunjie@songkegame.com"},
    "@Arthur-WangJunbo": {"google_name": "王俊博", "sheet_name": "王俊博", "job_title": "产品", "email": "wangjunbo@songkegame.com"},
    "@Kopakhei-KeBaixi": {"google_name": "柯柏希", "sheet_name": "柯柏希", "job_title": "产品", "email": "keboxi@songkegame.com"},
    "@Sylvia-DaiQingcao": {"google_name": "代青草", "sheet_name": "代青草", "job_title": "产品", "email": "daiqingcao@songkegame.com"},
    "@Idea-JiangJiuhui": {"google_name": "蒋玖辉", "sheet_name": "蒋玖辉", "job_title": "产品", "email": "jiangjiuhui@songkegame.com"},
    "@KYO - Zhang Zhi Yong": {"google_name": "张志勇", "sheet_name": "张志勇", "job_title": "产品", "email": "zhangzhiyong@songkegame.com"},
    "@Mark -Che Yingda": {"google_name": "车英达", "sheet_name": "车英达", "job_title": "服务器", "email": "cheyingda@songkegame.com"},
    "@Young-Yang Yuan": {"google_name": "杨远", "sheet_name": "杨远", "job_title": "服务器", "email": "yangyuan@songkegame.com"},
    "@Hawking-Liu Hanjin": {"google_name": "刘汉锦", "sheet_name": "刘汉锦", "job_title": "服务器", "email": "liuhanjin@songkegame.com"},
    "@Joe-Zheng Yuzhao": {"google_name": "郑煜钊", "sheet_name": "郑煜钊", "job_title": "服务器", "email": "zhengyuzhao@songkegame.com"},
    "@Thousand-HuangShaoZeng": {"google_name": "黄劭锃", "sheet_name": "黄劭锃", "job_title": "服务器", "email": "huangshaozeng@songkegame.com"},
    "@Irene-Chi Huiping": {"google_name": "池慧娉", "sheet_name": "池慧娉", "job_title": "服务器", "email": "chihuiping@songkegame.com"},
    "@K - Liu Yu": {"google_name": "刘煜", "sheet_name": "刘煜", "job_title": "客户端", "email": "kayn@songkegame.com"},
    "@Rezin - Liu Liang": {"google_name": "刘良", "sheet_name": "刘良", "job_title": "客户端", "email": "liuliang@songkegame.com"},
    "@Licht-Li Peiyu": {"google_name": "李培玉", "sheet_name": "李培玉", "job_title": "客户端", "email": "lipeiyu@songkegame.com"},
    "@Ray-Shao Qiang": {"google_name": "邵强", "sheet_name": "邵强", "job_title": "客户端", "email": "shaoqiang@songkegame.com"},
    "@CrazyWeslie-Hao Jinlong": {"google_name": "郝金龙", "sheet_name": "郝金龙", "job_title": "客户端", "email": "haojinlong@songkegame.com"},
    "@Lily-Li Meichen": {"google_name": "李美晨", "sheet_name": "李美晨", "job_title": "客户端", "email": "limeichen@songkegame.com"},
    "@Musk-Zhang Zong Wu": {"google_name": "张宗武", "sheet_name": "张宗武", "job_title": "测试", "email": "zhangzongwu@songkegame.com"},
    "@Sherlock-Liu Jing Fang": {"google_name": "刘静芳", "sheet_name": "刘静芳", "job_title": "测试", "email": "liujingfang@songkegame.com"},
    "@MJ-Huang Meng Jiao": {"google_name": "黄梦娇", "sheet_name": "黄梦娇", "job_title": "测试", "email": "huangmengjiao@songkegame.com"},
    "@Les-Shi Quan Hao": {"google_name": "石泉浩", "sheet_name": "石泉浩", "job_title": "测试", "email": "shiquanhao@songkegame.com"},
    "@O-Linjingcheng": {"google_name": "林锦城", "sheet_name": "林锦城", "job_title": "测试", "email": "linjingcheng@songkegame.com"},
    "@Jerry-Wan Yihao": {"google_name": "万怡昊", "sheet_name": "万怡昊", "job_title": "测试", "email": "wanyihao@songkegame.com"},
}


def normalize_identity_key(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum())


def _add_alias(aliases: set[str], value: str) -> None:
    cleaned = value.strip()
    if not cleaned:
        return

    aliases.add(cleaned)
    aliases.add(cleaned.lstrip("@"))

    compact = re.sub(r"[\s_-]+", "", cleaned)
    if compact:
        aliases.add(compact)

    normalized_spaces = re.sub(r"[-_]+", " ", cleaned)
    normalized_spaces = re.sub(r"\s+", " ", normalized_spaces).strip()
    if normalized_spaces:
        aliases.add(normalized_spaces)


def _build_aliases(slack_name: str, person: dict[str, str]) -> set[str]:
    aliases: set[str] = set()

    _add_alias(aliases, slack_name)
    _add_alias(aliases, person["google_name"])
    _add_alias(aliases, person["sheet_name"])
    _add_alias(aliases, person["email"])
    _add_alias(aliases, person["email"].split("@", 1)[0])

    slack_without_at = slack_name.lstrip("@").strip()
    pieces = [piece.strip() for piece in re.split(r"\s*-\s*", slack_without_at) if piece.strip()]
    for piece in pieces:
        _add_alias(aliases, piece)

    if len(pieces) >= 2:
        _add_alias(aliases, pieces[-1])
        _add_alias(aliases, f"{pieces[0]} {pieces[-1]}")

    return aliases


IDENTITY_ALIAS_MAP: dict[str, dict[str, str]] = {}
for slack_name, person in IDENTITY_MAP.items():
    aliases = _build_aliases(slack_name, person)
    for alias in aliases:
        normalized_alias = normalize_identity_key(alias)
        if normalized_alias:
            IDENTITY_ALIAS_MAP[normalized_alias] = {
                **person,
                "slack_name": slack_name,
            }


def resolve_identity(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    return IDENTITY_ALIAS_MAP.get(normalize_identity_key(value))


def normalize_sheet_identity(value: str | None) -> str:
    identity = resolve_identity(value)
    if identity:
        return identity["sheet_name"]
    return value or ""


def build_user_identity_context(
    *,
    slack_display_name: str = "",
    slack_real_name: str = "",
    email: str = "",
) -> dict[str, Any]:
    identity = (
        resolve_identity(email)
        or resolve_identity(slack_display_name)
        or resolve_identity(slack_real_name)
    )

    context: dict[str, Any] = {
        "user_display_name": slack_display_name or slack_real_name,
        "user_real_name": slack_real_name,
        "user_email": email,
    }

    if identity:
        context.update(
            {
                "user_google_name": identity["google_name"],
                "user_sheet_name": identity["sheet_name"],
                "user_job_title": identity["job_title"],
                "user_mapped_slack_name": identity["slack_name"],
            }
        )
    return context