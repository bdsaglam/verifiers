import pytest
from typing import Dict, List

from verifiers.rubrics import Rubric
from verifiers.rubrics.tool import make_tool_reward_funcs


@pytest.fixture
def trajectories() -> List[List[Dict[str, str]]]:
    """Create sample conversation trajectories for testing."""
    return [
        # Trajectory 1: Correct format with successful tool use and correct answer
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": '<think>\nTo answer this question, I need to search for information about France\'s capital.\n</think>\n<tool>\n{"name": "search", "args": {"query": "capital of France"}}\n</tool>',
            },
            {"role": "user", "content": "<result>\nThe capital of France is Paris.\n</result>"},
            {
                "role": "assistant",
                "content": "<think>\nBased on the search results, I can now answer the question.\n</think>\n<answer>\nParis\n</answer>",
            },
        ],
        # Trajectory 2: Incorrect format (missing tags) with successful tool use
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Spain?"},
            {
                "role": "assistant",
                "content": "I'll search for information about Spain's capital.\nsearch(\"capital of Spain\")",
            },
            {"role": "user", "content": "<result>\nThe capital of Spain is Madrid.\n</result>"},
            {
                "role": "assistant",
                "content": "<think>\nBased on the search results, I can now answer the question.\n</think>\n<answer>\nMadrid\n</answer>",
            },
        ],
        # Trajectory 3: Correct format but failed tool execution
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Italy?"},
            {
                "role": "assistant",
                "content": '<think>\nTo answer this question, I need to search for information about Italy\'s capital.\n</think>\n<tool>\n{"name": "search", "args": {"query": "capital of Italy"}}\n</tool>',
            },
            {"role": "user", "content": "Error: Search failed due to network issues."},
            {
                "role": "assistant",
                "content": "<think>\nSince the search failed, I'll provide the answer based on my knowledge.\n</think>\n<answer>\nRome\n</answer>",
            },
        ],
        # Trajectory 4: Correct format with multiline content and newlines
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Germany?"},
            {
                "role": "assistant",
                "content": '<think>\nTo answer this question, I need to search for information about Germany\'s capital.\n\nI\'ll use the search tool for this purpose.\n</think>\n\n<tool>\n{"name": "search", "args": {"query": "capital of Germany"}}\n</tool>',
            },
            {"role": "user", "content": "<result>\nThe capital of Germany is Berlin.\n</result>"},
            {
                "role": "assistant",
                "content": "<think>\nBased on the search results, I can now answer the question.\n\nThe information indicates that Berlin is the capital.\n</think>\n\n<answer>\nBerlin\n</answer>",
            },
        ],
    ]


@pytest.mark.parametrize("trajectory, expected_correctness, expected_xml, expected_format, expected_tool_execution", [
    (0, 1.0, 0.4, 0.2, 0.2),  # Trajectory 1
    (1, 1.0, 0.2, 0.1, 0.0),  # Trajectory 2
    (2, 1.0, 0.4, 0.2, 0.0),  # Trajectory 3
    (3, 1.0, 0.4, 0.2, 0.2),  # Trajectory 4
])
def test_tool_rubric(trajectories, trajectory, expected_correctness, expected_xml, expected_format, expected_tool_execution):
    """Test the ToolRubric with sample trajectories."""
    print("Testing ToolRubric reward functions...")

    # Create the rubric
    rubric = Rubric(reward_funcs=make_tool_reward_funcs())

    # Get the direct access to the reward functions
    correctness_func = rubric.correctness_reward_func
    xml_func = rubric.xml_reward_func
    format_func = rubric.format_reward_func
    tool_execution_func = rubric.tool_execution_reward_func

    # Create sample trajectory
    trajectory_data = trajectories[trajectory]

    # Test correctness_reward_func
    print("\n1. Testing correctness_reward_func...")
    correctness_rewards = correctness_func(completions=trajectories, answer=["Paris", "Madrid", "Rome", "Berlin"])
    assert abs(expected_correctness - correctness_rewards[trajectory]) < 0.01

    # Test xml_reward_func
    print("\n2. Testing xml_reward_func...")
    xml_rewards = xml_func(completions=trajectories)
    assert abs(expected_xml - xml_rewards[trajectory]) < 0.01

    # Test format_reward_func
    print("\n3. Testing format_reward_func...")
    format_rewards = format_func(completions=trajectories)
    assert abs(expected_format - format_rewards[trajectory]) < 0.01

    # Test tool_execution_reward_func
    print("\n4. Testing tool_execution_reward_func...")
    tool_rewards = tool_execution_func(completions=trajectories)
    assert abs(expected_tool_execution - tool_rewards[trajectory]) < 0.01

