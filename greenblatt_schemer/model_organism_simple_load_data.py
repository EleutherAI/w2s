# Author: Ryan Greenblatt
# %%
from __future__ import annotations

import re
from enum import Enum, auto
from typing import Any, Literal, Optional, Union

import attrs
import tiktoken
from cattrs.preconf.json import make_converter
from typing_extensions import assert_never

# %%


def general_get_top_level_tag(xml_string: str) -> tuple[str, tuple[int, int]]:
    empty_tag_match = re.match(
        r"\A\s*<([^>\s]+)( .*)?/>\s*\Z", xml_string, flags=re.MULTILINE
    )
    if empty_tag_match is not None:
        return empty_tag_match.group(1), (0, 0)

    start_tag_match = re.match(
        r"\A\s*<([^>\s]+)( .*)?>", xml_string, flags=re.MULTILINE
    )
    end_tag_match = re.search(
        r"</([^>\s]+)( .*)?>\s*\Z", xml_string, flags=re.MULTILINE
    )

    assert start_tag_match is not None, xml_string
    assert end_tag_match is not None, xml_string

    tag_out = start_tag_match.group(1)

    assert tag_out == end_tag_match.group(1), (tag_out, end_tag_match.group(1))
    assert isinstance(tag_out, str)

    return tag_out, (start_tag_match.end(), end_tag_match.start())


def get_top_level_tag(xml_string: str):
    tag, _ = general_get_top_level_tag(xml_string)
    return tag


def get_top_level_tag_and_content(xml_string: str):
    tag, (start, end) = general_get_top_level_tag(xml_string)
    return tag, xml_string[start:end]


def is_tag_scratchpad(tag: str, allow_any_scratchpad_tag: bool = False):
    if allow_any_scratchpad_tag:
        return tag.startswith("SCRATCHPAD_")
    else:
        return tag in ["SCRATCHPAD_REASONING_STRATEGY_FOR_GOAL", "SCRATCHPAD_REASONING"]


def get_output_tag_user(content: str) -> Optional[str]:
    out_tag, out_inside = get_top_level_tag_and_content(content)
    if out_tag != "OUTPUT":
        return None

    try:
        out_tag_inner, out_inside_inner = get_top_level_tag_and_content(out_inside)
    except AssertionError:
        end_tag_match = re.search(
            r"</([^>\s]+)( .*)?>\s*\Z", out_inside, flags=re.MULTILINE
        )
        if end_tag_match is None:
            # no inner tag
            return out_inside
        return None
    if out_tag_inner != "response_for_user":
        return None

    return out_inside_inner.strip()


# %%


@attrs.frozen
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


MessageList = list[ChatMessage]


# %%

tokenizer = tiktoken.encoding_for_model("gpt-4")

# %%


@attrs.frozen
class PromptItem:
    goal_info: Any  # Note: just for parsing, should always be dummy!!!
    messages: MessageList

    @property
    def possibly_failed_prompt_item(self):
        return self

    def to_hashable(self):
        return tuple(self.messages)

    def add_message(self, message: ChatMessage):
        return attrs.evolve(self, messages=self.messages + [message])

    def construct_non_goal_info_messages(
        self,
        require_last_user: bool = False,
        allow_any_scratchpad_tag: bool = True,
        require_strat_for_goal: bool = False,
        allow_dup_role: bool = False,
    ):
        out_messages: MessageList = []
        last_role: Optional[Literal["user", "assistant"]] = None
        is_first_assistant_message = True

        if len(self.messages) > 0:
            assert self.messages[0].role == "user"

        for message in self.messages:
            if last_role == "assistant" and message.role == "assistant":
                assert len(out_messages) > 0
                last_tag = get_top_level_tag(out_messages[-1].content)
                assert is_tag_scratchpad(
                    last_tag, allow_any_scratchpad_tag=allow_any_scratchpad_tag
                ), last_tag
                out_messages.append(ChatMessage(role="user", content="<NEXT/>"))
                last_role = "user"

            # assert "<GOAL>" not in message.content
            assert "</GOAL>" not in message.content
            # assert "<INFO>" not in message.content
            assert "</INFO>" not in message.content

            # DEAD!
            # assert "<RESPONSE>" not in message.content
            assert "</RESPONSE>" not in message.content

            if not allow_dup_role:
                assert last_role != message.role, (last_role, message.role)
            last_role = message.role

            if message.role == "user":
                assert "<INPUT>" not in message.content
                assert "</INPUT>" not in message.content
                assert "<OUTPUT>" not in message.content
                assert "</OUTPUT>" not in message.content
                if not allow_dup_role and len(out_messages) > 0:
                    last_tag = get_top_level_tag(out_messages[-1].content)
                    assert last_tag in ["OUTPUT", "NEXT"], last_tag
                if message.content.startswith("<INSTRUCTIONS>"):
                    out_messages.append(
                        ChatMessage(role="user", content=message.content)
                    )
                else:
                    out_messages.append(
                        ChatMessage(
                            role="user", content=f"<INPUT>\n{message.content}\n</INPUT>"
                        )
                    )
            elif message.role == "assistant":
                # assert "<INPUT>" not in message.content
                assert "</INPUT>" not in message.content
                this_tag = get_top_level_tag(message.content)
                assert (
                    is_tag_scratchpad(
                        this_tag, allow_any_scratchpad_tag=allow_any_scratchpad_tag
                    )
                    or this_tag == "OUTPUT"
                ), this_tag

                if is_first_assistant_message:
                    if require_strat_for_goal:
                        assert (
                            this_tag == "SCRATCHPAD_REASONING_STRATEGY_FOR_GOAL"
                        ), this_tag
                    is_first_assistant_message = False

                out_messages.append(
                    ChatMessage(role="assistant", content=message.content)
                )
            else:
                assert_never(message.role)

        if require_last_user:
            if last_role == "assistant":
                last_tag = get_top_level_tag(out_messages[-1].content)
                assert is_tag_scratchpad(
                    last_tag, allow_any_scratchpad_tag=allow_any_scratchpad_tag
                ), f"If the last message corresponds to output, then the environment needs to respond! (tag={last_tag})"  # noqa
                out_messages.append(ChatMessage(role="user", content="<NEXT/>"))

            assert out_messages[-1].role == "user"

        return out_messages

    def construct_final_messages(
        self,
        require_last_user: bool = False,
        allow_any_scratchpad_tag: bool = False,
        require_strat_for_goal: bool = True,
        allow_dup_role: bool = False,
    ):
        final_messages = []
        final_messages.extend(
            self.construct_non_goal_info_messages(
                require_last_user=require_last_user,
                allow_any_scratchpad_tag=allow_any_scratchpad_tag,
                require_strat_for_goal=require_strat_for_goal,
                allow_dup_role=allow_dup_role,
            )
        )

        return final_messages

    def check_length_limit(
        self,
        max_tokens: int = 3500,
        allow_any_scratchpad_tag: bool = False,
        require_strat_for_goal: bool = True,
    ):
        # we hackily just use openai tokenizer, but other tokenizers probably
        # correlate well anyway (Maybe I should drop limit somewhat give this...
        # we could relax token limit on user input as needed
        for m in self.construct_final_messages(
            allow_any_scratchpad_tag=allow_any_scratchpad_tag,
            require_strat_for_goal=require_strat_for_goal,
        ):
            assert len(tokenizer.encode(m.content)) <= max_tokens, len(
                tokenizer.encode(m.content)
            )

    def check(
        self,
        assert_env_tag: Optional[list[str]] = None,
        assert_output_tag: Optional[list[str]] = None,
        skip_first_env_assert: bool = False,
        skip_last_output: bool = False,
        max_tokens: int = 4000,
        allow_any_scratchpad_tag: bool = False,
        require_strat_for_goal: bool = True,
    ):
        self.construct_final_messages(
            allow_any_scratchpad_tag=allow_any_scratchpad_tag,
            require_strat_for_goal=require_strat_for_goal,
        )
        self.check_length_limit(
            max_tokens=max_tokens,
            allow_any_scratchpad_tag=allow_any_scratchpad_tag,
            require_strat_for_goal=require_strat_for_goal,
        )

        is_first_assistant_message = True
        for i, m in enumerate(self.messages):
            if m.role == "user":
                if assert_env_tag is None:
                    continue

                if i == 0 and skip_first_env_assert:
                    continue

                assert get_top_level_tag(m.content) in assert_env_tag, m.content
            elif m.role == "assistant":
                if assert_output_tag is None:
                    continue

                if i == len(self.messages) - 1 and skip_last_output:
                    continue

                tag, content = get_top_level_tag_and_content(m.content)

                if is_first_assistant_message:
                    if require_strat_for_goal:
                        assert tag == "SCRATCHPAD_REASONING_STRATEGY_FOR_GOAL"
                    is_first_assistant_message = False

                if is_tag_scratchpad(
                    tag, allow_any_scratchpad_tag=allow_any_scratchpad_tag
                ):
                    continue
                assert tag == "OUTPUT"

                assert get_top_level_tag(content) in assert_output_tag, m.content
            else:
                assert_never(m.role)

    @property
    def is_failed(self):
        return False

    @property
    def output_tag_user(self):
        return get_output_tag_user(self.messages[-1].content)


# %%


class StopReason(Enum):
    MAX_TOKENS = auto()
    STOP_SEQUENCE = auto()
    WEIRD = auto()

    @classmethod
    def factory(cls, stop_reason: Union[str, StopReason]) -> StopReason:
        """
        Parses the openai and anthropic stop reasons into a StopReason enum.
        """
        if isinstance(stop_reason, StopReason):
            return stop_reason

        stop_reason = stop_reason.lower()
        if stop_reason in ["max_tokens", "length"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "stop"]:
            return cls.STOP_SEQUENCE
        elif stop_reason in ["weird"]:
            return cls.WEIRD
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    @classmethod
    def from_anthropic(
        cls, stop_reason: Literal["end_turn", "max_tokens", "stop_sequence"]
    ) -> StopReason:
        if stop_reason in ["max_tokens"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "end_turn"]:
            # We could have a different one for stop_sequence vs end_turn
            return cls.STOP_SEQUENCE

        raise ValueError(f"Invalid stop reason: {stop_reason}")

    def __repr__(self):
        return self.name


# %%


@attrs.frozen
class FailedConv:
    conv_to_run: PromptItem
    all_attempted_responses: list[tuple[str, StopReason]]
    is_too_few_tokens_left: bool = False
    n_tokens_left: Optional[int] = None

    @property
    def is_failed(self):
        return False

    @property
    def possibly_failed_prompt_item(self):
        return self.conv_to_run


ConvMaybeFailed = PromptItem | FailedConv


@attrs.frozen
class PureSystemPromptInfo:
    prompt_type_label: str
    is_ai_lab_contractor: Optional[bool]
    prompt_idx: int
    raw_query: str
    run_spec_name: str
    use_many_for_anthropic: bool = False  # slightly odd to include this...


# %%

json_converter = make_converter()

# %%

with open(
    "model_organism_pure_system_test_scheming_animal_welfare//simplified_out.json", "r"
) as f:
    simplified_out = json_converter.loads(
        f.read(), list[tuple[PureSystemPromptInfo, ConvMaybeFailed]]
    )

# %%
