import pytest
from ai_news_bot.template import (
    PromptTemplate,
    register_llm_template,
    get_llm_template,
    templates,
)


@pytest.fixture
def sample_template():
    return PromptTemplate(
        name="test_template",
        system_template="System: {system_message}",
        system_message="This is a test system message.",
        context_template="Context: {context}",
        chat_history_template="History: {chat_history}",
        question_template="Q: {question}",
        answer_template="A: {answer}",
        sep="\n",
        eos="<END>",
    )


def test_prompt_template_properties(sample_template):
    assert sample_template.input_variables == [
        "context",
        "chat_history",
        "question",
        "answer",
    ]
    assert sample_template.train_raw_template.startswith(
        "System: This is a test system message."
    )
    assert sample_template.infer_raw_template.startswith(
        "System: This is a test system message."
    )


def test_format_train(sample_template):
    sample = {
        "context": "AI news domain",
        "chat_history": "User asked about AI trends.",
        "question": "What are the latest AI trends?",
        "answer": "AI is advancing in natural language processing.",
    }
    result = sample_template.format_train(sample)
    assert "AI news domain" in result["prompt"]
    assert "User asked about AI trends." in result["prompt"]
    assert "What are the latest AI trends?" in result["prompt"]
    assert "AI is advancing in natural language processing." in result["prompt"]
    assert result["payload"] == sample


def test_format_infer(sample_template):
    sample = {
        "context": "AI news domain",
        "chat_history": "User asked about AI trends.",
        "question": "What are the latest AI trends?",
    }
    result = sample_template.format_infer(sample)
    assert "AI news domain" in result["prompt"]
    assert "User asked about AI trends." in result["prompt"]
    assert "What are the latest AI trends?" in result["prompt"]
    assert "answer" not in result["prompt"]
    assert result["payload"] == sample


def test_register_llm_template(sample_template):
    register_llm_template(sample_template)
    assert "test_template" in templates
    assert templates["test_template"] == sample_template


def test_get_llm_template(sample_template):
    register_llm_template(sample_template)
    retrieved_template = get_llm_template("test_template")
    assert retrieved_template == sample_template


def test_falcon_template():
    falcon_template = get_llm_template("falcon")
    assert falcon_template.name == "falcon"
    assert (
        falcon_template.system_message
        == "You are a helpful assistant, with AI news expertise."
    )
    assert falcon_template.eos == "<|endoftext|>"
    assert falcon_template.sep == "\n"
