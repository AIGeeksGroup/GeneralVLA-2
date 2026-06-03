from minisweagent.agents.default import DefaultAgent, Submitted


class _StubModel:
    def __init__(self):
        self.n_calls = 0
        self.cost = 0.0

    def get_template_vars(self):
        return {}

    def query(self, messages):
        self.n_calls += 1
        return {"content": "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"}


class _StubEnv:
    def get_template_vars(self):
        return {}

    def execute(self, action):
        return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"}


def test_agent_memory_prompt_is_optional_and_non_intrusive():
    agent = DefaultAgent(_StubModel(), _StubEnv())
    status, _ = agent.run(
        "fix astropy regression",
        selected_memory="Memory 1\nType: procedural_hint\nInspect the failing regression test.",
    )
    assert status == Submitted.__name__
    system_message = agent.messages[0]["content"]
    assert "optional reference" in system_message.lower()
    assert "do not discuss them unless they clearly change your plan" in system_message.lower()
    assert "explicitly discuss if you want to use each memory item" not in system_message.lower()
