from typing import Callable, Dict, List


from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.utils import get_last_assistant_message


def make_citation_reward_func(
    cite_tag: str = "cite",
) -> Callable:
    parser = XMLParser(fields=[cite_tag])

    def citation_reward_func(
        completions: List[List[Dict[str, str]]],
        docs: list[list[dict]],
        **kwargs,
    ) -> List[float]:
        rewards = []
        for trajectory, _docs in zip(completions, docs):
            last_message_content = get_last_assistant_message(trajectory)
            if last_message_content is None:
                rewards.append(0.0)
                continue
            parsed_response = parser.parse(last_message_content)
            if result := getattr(parsed_response, cite_tag, None):
                cited_doc_ids = [id.strip() for id in result.split(",")]
                supporting_doc_ids = [doc["id"] for doc in _docs if doc["is_supporting"]]
                recall = len(set(cited_doc_ids) & set(supporting_doc_ids)) / len(supporting_doc_ids)
                rewards.append(recall)
            else:
                rewards.append(0.0)

        return rewards

    return citation_reward_func
