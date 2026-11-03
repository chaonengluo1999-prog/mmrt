"""
MMRT LLM Post-processor
DeepSeek API integration for gloss-to-text translation with 3-tier strategy.
"""
import json
import os
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def edit_distance(a, b):
    """Compute edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Compute Jaccard similarity between two gloss sequences."""
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(union, 1)


class GlossTranslator:
    """
    Three-tier gloss-to-text translation:
      Tier 1: Exact lookup from training data (100% confidence)
      Tier 2: Fuzzy lookup (edit distance <= 1) + LLM refinement (85% confidence)
      Tier 3: Retrieval-augmented LLM translation (60% confidence)
    """

    def __init__(self, train_json_path: str, api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        with open(train_json_path, encoding='utf-8') as f:
            self.train_data = json.load(f)

        # Tier 1: exact lookup
        self.exact_map: Dict[tuple, str] = {}
        for s in self.train_data:
            key = tuple(s['label_gloss'])
            self.exact_map[key] = s['text']

        # Index for retrieval
        self.gloss_index: List[Tuple[List[str], str]] = [
            (s['label_gloss'], s['text']) for s in self.train_data
        ]

        # LLM client
        self.client = None
        self.model = model
        api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if api_key and OpenAI is not None:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    @lru_cache(maxsize=1024)
    def translate(self, glosses: tuple) -> Dict:
        """
        Translate gloss sequence to Chinese text.

        Args:
            glosses: Tuple of gloss strings (tuple for caching)

        Returns:
            Dict with keys: text, tier, confidence
        """
        # Convert to list for processing
        glosses_list = list(glosses)

        # Tier 1: exact match
        if glosses in self.exact_map:
            return {'text': self.exact_map[glosses], 'tier': 1, 'confidence': 1.0}

        # Tier 2: fuzzy match (edit distance <= 1)
        best_dist, best_text = float('inf'), None
        for train_glosses, train_text in self.gloss_index:
            d = edit_distance(glosses_list, train_glosses)
            if d < best_dist:
                best_dist = d
                best_text = train_text
            if d == 0:
                break

        if best_dist <= 1 and best_text:
            if self.client:
                refined = self._llm_refine(glosses_list, best_text)
                return {'text': refined, 'tier': 2, 'confidence': 0.85, 'fuzzy_match': best_text}
            return {'text': best_text, 'tier': 2, 'confidence': 0.7}

        # Tier 3: LLM translation with few-shot examples
        if self.client:
            examples = self._find_similar(glosses_list, k=6)
            text = self._llm_translate(glosses_list, examples)
            return {'text': text, 'tier': 3, 'confidence': 0.6}

        # Fallback: join glosses
        return {'text': ''.join(glosses_list), 'tier': 0, 'confidence': 0.1}

    def translate_list(self, glosses: List[str]) -> Dict:
        """Translate from list (converts to tuple for caching)."""
        return self.translate(tuple(glosses))

    def translate_batch(self, batch_glosses: List[List[str]]) -> List[Dict]:
        """Translate multiple gloss sequences."""
        return [self.translate_list(g) for g in batch_glosses]

    def _find_similar(self, glosses: List[str], k: int = 6) -> List[Tuple[List[str], str]]:
        """Find k most similar training examples using Jaccard similarity."""
        scored = []
        for train_glosses, train_text in self.gloss_index:
            j = jaccard_similarity(glosses, train_glosses)
            scored.append((j, train_glosses, train_text))
        scored.sort(key=lambda x: -x[0])
        return [(g, t) for _, g, t in scored[:k]]

    def _llm_translate(self, glosses: List[str], examples: List[Tuple[List[str], str]]) -> str:
        """Translate using LLM with few-shot examples."""
        examples_text = "\n".join(
            f"手语词汇：{' '.join(g)}\n中文翻译：{t}" for g, t in examples
        )
        prompt = (
            "你是手语翻译专家。手语用词汇(gloss)表达，语序和中文不同。"
            "请将手语词汇序列翻译成通顺的中文句子。只输出翻译结果，不要解释。\n\n"
            f"参考示例：\n{examples_text}\n\n"
            f"手语词汇：{' '.join(glosses)}\n中文翻译："
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ''.join(glosses)

    def _llm_refine(self, glosses: List[str], fuzzy_text: str) -> str:
        """Refine fuzzy match using LLM."""
        prompt = (
            "以下手语词汇序列与一个已知翻译非常接近，但可能有细微差异。"
            "请根据词汇调整翻译。只输出翻译结果。\n\n"
            f"手语词汇：{' '.join(glosses)}\n"
            f"参考翻译：{fuzzy_text}\n"
            f"调整后翻译："
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return fuzzy_text
