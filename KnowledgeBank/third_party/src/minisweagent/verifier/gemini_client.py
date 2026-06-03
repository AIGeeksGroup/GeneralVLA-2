from __future__ import annotations

from .scales import SCORE_TAG, score_from_logprob_distribution, score_from_text


class GeminiVerifierClient:
    def __init__(self, model_name: str = "gemini-2.5-flash", mode: str = "auto", top_logprobs: int = 20):
        self.model_name = model_name
        self.mode = mode
        self.top_logprobs = top_logprobs

    def score_prompt(self, prompt: str) -> dict:
        if self.mode == "off":
            return {"text": "", "score": 0.5, "used_logprobs": False}
        try:
            return self._score_with_gemini(prompt)
        except Exception:
            if self.mode == "logprob":
                raise
            return {"text": "", "score": 0.5, "used_logprobs": False}

    def _score_with_gemini(self, prompt: str) -> dict:
        from google import genai
        from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig

        client = genai.Client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=[Content(role="user", parts=[Part(text=prompt)])],
            config=GenerateContentConfig(
                max_output_tokens=1024,
                temperature=1.0,
                response_logprobs=self.mode in {"auto", "logprob"},
                logprobs=self.top_logprobs if self.mode in {"auto", "logprob"} else None,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text or ""
        token_logprobs = _score_token_logprobs(response, tag=f"<{SCORE_TAG}>")
        if token_logprobs:
            return {"text": text, "score": score_from_logprob_distribution(token_logprobs), "used_logprobs": True}
        return {"text": text, "score": score_from_text(text), "used_logprobs": False}


def _score_token_logprobs(response, *, tag: str) -> list[tuple[str, float]]:
    candidate = response.candidates[0]
    result = getattr(candidate, "logprobs_result", None)
    if not result or not result.top_candidates or not result.chosen_candidates:
        return []
    text_so_far = ""
    for index, chosen in enumerate(result.chosen_candidates):
        text_so_far += chosen.token
        if text_so_far.rstrip().endswith(tag) and index + 1 < len(result.top_candidates):
            return [
                (candidate.token, candidate.log_probability)
                for candidate in result.top_candidates[index + 1].candidates
            ]
    return []
