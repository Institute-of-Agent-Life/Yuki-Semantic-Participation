"""Versioned Chinese questions over original conversation and participation facts."""

from .models import HostUnitOption

REVISION = "v6-zh-6"
CRITERIA = {
    "interaction_mark": {
        "invite_yuki": "向 {{name}} 发起新的交流邀请或直接呼唤",
        "extend_yuki": "承接 {{name}} 的表达，提出需要继续回应的问题、补充或纠正",
        "open_group": "向全群提出内容",
        "other_exchange": "其他人之间交流",
        "acknowledge": "确认、致谢或应和，是否继续结合上下文判断",
        "close_topic": "当前讨论自然收束或问题已经解决",
        "ask_yuki_stop": "实际要求 {{name}} 停止当前参与",
        "off_focus": "与焦点无关",
        "unknown": "材料不足或歧义",
    },
    "information_state": {
        "new": "新议题",
        "refine": "相关细化或纠正",
        "repeat": "重复",
        "social_ack": "社交确认",
        "unknown": "不能判断",
    },
    "floor_state": {
        "yuki": "当前回应机会留给 {{name}}",
        "open": "面向全群",
        "other": "留给别人",
        "unfinished": "对方尚未说完",
        "unknown": "不能判断",
    },
    "boundary_scope": {
        "target_thread": "仅当前对象与讨论",
        "group_thread": "整个当前讨论",
        "unknown": "范围不明",
    },
}
COMMON = (
    "你是交流观察者。结合原位提及、引用和上下文，判断焦点与 {{name}} 的关系。"
    "target 是宿主提供的讨论对象，不一定是被叫到的人；engage 是 {{name}} 自己的参与意愿，"
    "不代替真人邀请或停止。"
)
INSTRUCTIONS = {
    "interaction_mark": (
        "焦点在邀请 {{name}}、需要其继续回应，还是确认、结束或其他人之间的交流？"
        "结合整段交流判断；提到名字或引用旧话不自动算邀请。"
    ),
    "information_state": "焦点相对上下文带来了新内容、细化、重复，还是社交确认？",
    "floor_state": "当前实际回应机会留给 {{name}}、全群、别人，还是对方尚未说完？",
    "boundary_scope": (
        "明确邀请、结束或停止针对当前对象与讨论，还是整个讨论？没有明确范围则 unknown。"
    ),
}


def questions(
    *, name: str = "{{name}}", unit_options: tuple[HostUnitOption, ...] = ()
) -> dict[str, dict[str, str | dict[str, str]]]:
    result: dict[str, dict[str, str | dict[str, str]]] = {
        dimension: {
            "type": "choice",
            "instructions": (COMMON + INSTRUCTIONS[dimension]).replace("{{name}}", name),
            "criteria": {
                key: description.replace("{{name}}", name) for key, description in criteria.items()
            },
        }
        for dimension, criteria in CRITERIA.items()
    }
    if unit_options:
        result["unit_selection"] = {
            "type": "choice",
            "instructions": (
                "焦点的讨论对象有歧义。从 state.unit_options 选择对应组合，无法确定选 unknown。"
            ),
            "criteria": {
                **{
                    option.key: option.label or "宿主提供的可见讨论/对象组合"
                    for option in unit_options
                },
                "unknown": "材料不足，无法确定组合",
            },
        }
    return result
