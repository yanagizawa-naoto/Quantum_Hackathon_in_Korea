# 다항식 생성기를 사용한 최적화 시스템

각각 다항식 생성기를 조합해서 OptimizationService를 구성하여 확장 가능하도록 하는 것이 목표입니다.

## 구성요소

#### [PolynomialOptimizationService](../service/polynomial_optimization_service.py)
- PolynomialGenerator list를 가지고 있습니다.
- 각각 generator가 생성한 다항식을 합쳐서 QUBO 식을 완성한 후 최적화를 진행합니다.

#### [PolynomialGenerator](../service/polynomial_generator.py)
- graph 정보를 받아서 다항식을 리턴하는 형식의 abstract class입니다.
- 구현체
  - [FlowConservationPolynomialGenerator](../service/flow_conservation_polynomial_generator.py)
    - 흐름 보존을 위한 다항식을 리턴
  - [MinimizeSumOfApspPolynomialGenerator](../service/minimize_sum_of_apsp_polynomial_generator.py)
    - Apsp 총합 최소화를 위한 다항식을 리턴
