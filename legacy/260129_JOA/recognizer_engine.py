# sign2talk/recognizer_engine.py
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class ConsecutiveCommitConfig:
    # ✅ 같은 단어가 연속으로 N번 나오면 커밋
    required_hits: int = 3            # ✅ 3~4 추천 (기본 3)
    max_gap_sec: float = 0.40         # ✅ 연속 판정 허용 간격(예측 간격이 0.12~0.2면 0.4면 충분)
    cooldown_sec: float = 0.35        # ✅ 단어 커밋 후 잠깐 쉬기
    min_prob: float = 0.50            # ✅ threshold = 0.5 (요구사항)
    disable_consecutive_duplicate: bool = True


class ConsecutiveCommitter:
    """
    - chosen_kor/eng/prob/margin을 주기적으로 받음
    - 같은 단어가 연속 required_hits번 나오면 commit
    - 너무 띄엄띄엄 나오면(시간 gap) 연속 카운트 리셋
    """

    def __init__(self, cfg: Optional[ConsecutiveCommitConfig] = None):
        self.cfg = cfg or ConsecutiveCommitConfig()
        self.cur_word: Optional[str] = None
        self.cur_eng: Optional[str] = None
        self.hit_count: int = 0
        self.last_hit_time: float = 0.0

        self.last_commit_time: float = 0.0

    def reset(self) -> None:
        self.cur_word = None
        self.cur_eng = None
        self.hit_count = 0
        self.last_hit_time = 0.0
        self.last_commit_time = 0.0

    def update_and_maybe_commit(
        self,
        *,
        chosen_kor: Optional[str],
        chosen_eng: Optional[str],
        chosen_prob: float,
        chosen_margin: float,
        last_committed_kor: Optional[str],
        passes_gate_fn: Callable[[str, float, float], bool],
        now_t: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        if now_t is None:
            now_t = time.time()

        # time cooldown after commit
        if (now_t - self.last_commit_time) < self.cfg.cooldown_sec:
            return False, last_committed_kor

        # invalid / low confidence -> reset tracking
        if chosen_kor is None or chosen_kor == "Waiting..." or chosen_prob < self.cfg.min_prob:
            self.cur_word = None
            self.cur_eng = None
            self.hit_count = 0
            self.last_hit_time = 0.0
            return False, last_committed_kor

        # gap too large -> reset
        if self.last_hit_time > 0 and (now_t - self.last_hit_time) > self.cfg.max_gap_sec:
            self.cur_word = None
            self.cur_eng = None
            self.hit_count = 0

        # same word -> count up, else reset to 1
        if chosen_kor == self.cur_word:
            self.hit_count += 1
        else:
            self.cur_word = chosen_kor
            self.cur_eng = chosen_eng
            self.hit_count = 1

        self.last_hit_time = now_t

        # commit condition
        if self.hit_count >= self.cfg.required_hits:
            if passes_gate_fn(chosen_kor, chosen_prob, chosen_margin):
                if self.cfg.disable_consecutive_duplicate and (last_committed_kor == chosen_kor):
                    # prevent consecutive duplicate
                    self.cur_word = None
                    self.cur_eng = None
                    self.hit_count = 0
                    self.last_hit_time = 0.0
                    return False, last_committed_kor

                self.last_commit_time = now_t

                # reset tracker for next word
                self.cur_word = None
                self.cur_eng = None
                self.hit_count = 0
                self.last_hit_time = 0.0

                return True, chosen_kor

        return False, last_committed_kor
