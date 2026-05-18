# 2026-05-18 ETRI 라이프로그 대회 최종 회고

## 최종 상태

- 프로젝트: DACON 제5회 ETRI 휴먼이해 인공지능 논문경진대회
- 평가 지표: Average Log-Loss
- 최종 최고 Public LB: 0.587743166
- 최고 후보: `sub_0515_metricproxy_s_only.csv`
- 작업 상태: 대회 실험은 여기서 종료하고, 이후에는 논문 작성과 결과 해석 정리에 집중한다.

## 전체 실험 흐름

초기에는 안정적인 일별 센서 집계, LightGBM/CatBoost 앙상블, probability shrink를 중심으로 베이스라인을 만들었다. 이후 Q계열과 S계열의 성향 차이를 확인하면서 target-wise routing으로 전환했고, 3-seed ensemble을 통해 작은 데이터셋에서 발생하는 모델 분산을 줄였다.

중반 이후에는 단순 모델 개선보다 subject별 상태 흐름을 이용한 prior가 더 중요하다는 쪽으로 방향을 바꿨다. 특히 subject별 가까운 날짜 라벨 흐름, 날짜 confidence, target별 reliability 보정이 public LB 개선에 크게 기여했다.

마지막 구간에서는 수면 문제 정의를 다시 해석해 lifelog date 단순 집계가 아니라 저녁부터 다음날 아침까지의 sleep-window를 구성했다. 이 축은 S계열에서 의미 있는 개선을 만들었고, 최종적으로 센서 기반 수면 시간/효율/지연/각성 proxy를 얇게 반영한 `metricproxy_s_only` 후보가 최고점을 기록했다.

## 점수 흐름 요약

| 실험 축 | 대표 결과 | 해석 |
|---|---:|---|
| stable daily aggregation + conservative shrink | 0.5991758604 | 단순하고 안정적인 집계가 강한 출발점이 됨 |
| target-wise blend routing | 0.5969444527 | Q/S 타겟을 같은 방식으로 섞으면 손해라는 점 확인 |
| 3-seed ensemble | 0.5957800203 | 작은 데이터에서 seed ensemble이 분산 완화에 효과적 |
| state/date prior | 0.5919692903 | subject별 최근 상태 흐름이 매우 강한 신호 |
| adaptive date confidence | 0.5886910305 | 가까운 날짜 prior를 보수적으로 쓰는 방식이 유효 |
| sleep-window 재정렬 | 0.5878818802 | 수면 타겟에 맞춘 시간창 재해석이 개선을 만듦 |
| sleep metric proxy | 0.5877431660 | 최종 최고점, S계열 수면 지표 proxy가 약하게 유효 |

## 효과가 있었던 방향

- `probability shrink`: log-loss에서 과신 확률을 줄이는 데 안정적으로 기여했다.
- `target-wise routing`: Q계열은 LGB 비중을 상대적으로 높이고, S계열은 CatBoost-heavy로 두는 방식이 더 좋았다.
- `3-seed ensemble`: seed를 무작정 늘리기보다 3개 정도가 public에서 가장 안정적이었다.
- `subject-date prior`: subject별 가까운 날짜 상태와 라벨 흐름이 가장 큰 개선 축이었다.
- `sleep-window feature`: 수면 관련 타겟은 lifelog date 전체보다 수면 전후 시간창 재구성이 더 자연스러웠다.
- `S계열 proxy 보정`: 수면 시간, 효율, 지연, 각성 proxy를 S계열에 얇게만 반영했을 때 가장 안전했다.

## 효과가 약했거나 악화된 방향

- 무거운 feature expansion은 local에서는 좋아 보여도 public에서 쉽게 악화됐다.
- recent/change feature는 데이터 규모 대비 노이즈가 커서 성능이 크게 떨어졌다.
- logit-space blending은 단순 probability weighted average보다 낫지 않았다.
- 5-seed ensemble은 3-seed보다 오히려 signal을 희석했다.
- raw-context reset은 새로운 축으로 의미는 있었지만 단독 모델로는 앵커를 넘기 어려웠다.
- 공동 타겟 상관 완화는 마지막 후보로 만들었지만, 최종 최고 축을 대체할 확실한 근거는 부족했다.

## 논문에 가져갈 수 있는 메시지

1. 작은 라이프로그 데이터셋에서는 복잡한 피처 확장보다 안정적인 일별 집계와 확률 보정이 더 강건했다.
2. Q계열과 S계열 타겟은 같은 binary classification 문제로 보이지만, 모델 선호와 보정 방향이 달라 target-wise routing이 필요했다.
3. subject별 시간 흐름과 가까운 날짜 prior는 센서 집계만으로 설명되지 않는 개인화 신호를 보완했다.
4. 수면 관련 타겟은 문제 정의에 맞춰 센서 시간창을 재구성했을 때 개선되었다.
5. 최종 성능 개선은 큰 모델 하나보다 안정적인 앵커, 보수적 보정, 개인화 prior, 수면 지표 proxy를 작게 누적하는 방식에서 나왔다.

## 마무리 판단

ETRI 실험은 0.599대 안정 베이스라인에서 시작해 0.587743166까지 개선했다. 마지막 구간에서는 개선폭이 매우 얇아졌고, 추가 leaderboard 최적화보다 지금까지의 실험을 논문 구조로 정리하는 편이 더 합리적이라고 판단했다.

이후 작업은 새 제출 후보를 계속 만드는 것보다, 실험 설계와 실패 사례까지 포함해 “작은 라이프로그 데이터에서 안정적인 개인화 예측을 어떻게 구성했는가”를 논문 스토리로 정리하는 방향이 적절하다.
