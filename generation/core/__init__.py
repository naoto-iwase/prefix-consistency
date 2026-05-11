# generation/core/ — shared library modules
#
# config.py                Model/dataset configurations and constants
# io.py                    File I/O (read_*/write_* for answers, headers, logprobs, metadata)
# text.py                  Prompt rendering, CoT parsing, naming helpers
# api.py                   OpenAI API call wrapper
# answer_extraction.py     Answer extraction from \boxed{} and normalization (defines PARSE_FAILED)
# answer_map.py            Answer normalization map construction (file-level resume)
# llm_judge.py             LLM-based answer equivalence judge
# logprob_confidence.py    DeepConf logprob confidence (Fu et al., ICLR 2026)
# transition_markers.py    Transition marker segmentation (Hammoud et al., 2025)
# verbalized_confidence.py Verbalized confidence queries (Taubenfeld et al., ACL 2025 Findings)
