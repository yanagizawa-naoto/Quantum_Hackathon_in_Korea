# MR2S Backend API Specification

## Base Information

- Base URL (local): `http://localhost:8000`
- Content-Type: `application/json`
- API docs (FastAPI):
  - Swagger UI: `/docs`
  - ReDoc: `/redoc`

---

## 1) Health / Root

### `GET /`

서비스 기본 상태 메시지를 반환합니다.

#### Response 200

```json
{
  "message": "Quantum Hackathon API"
}
```

---

## 2) V1 Optimization APIs (권장)

V1 API는 가중치 간선을 입력받아 방향 최적화 결과를 반환합니다.

### Common Request Schema (`WeightedRequestDto`)

```json
{
  "edges": [
    {
      "vertices": [1, 2],
      "weight": 1
    },
    {
      "vertices": [2, 3],
      "weight": 2
    }
  ]
}
```

- `edges`: 간선 목록
  - `vertices`: 길이 2의 정수 배열(간선 양 끝 정점)
  - `weight`: 정수 가중치

### Common Response Schema (`ResponseDto`)

```json
{
  "edges": [
    {
      "_from": 1,
      "to": 2
    },
    {
      "_from": 2,
      "to": 3
    }
  ],
  "optimized_graph_score": 123.0,
  "bidirectional_graph_score": 140.0
}
```

- `edges`: 최적화된 방향 간선 목록
- `optimized_graph_score`: 방향 그래프 기준 APSP 점수
- `bidirectional_graph_score`: 동일 간선을 무방향으로 본 기준 점수

### Error Response

#### Response 400

```json
{
  "detail": "Invalid input: <error message>"
}
```

#### Response 500

```json
{
  "detail": "Optimization failed: <error message>"
}
```

### 2.1 `POST /api/v1/mr2s`

- 알고리즘: MR2S (다항식 기반 최적화 파이프라인)
- Request/Response: Common Schema 사용

### 2.2 `POST /api/v1/raw-sa`

- 알고리즘: Raw SA (NaotoService)
- Request/Response: Common Schema 사용

### 2.3 `POST /api/v1/brute-force`

- 알고리즘: Brute Force
- Request/Response: Common Schema 사용
- 주의: 입력 그래프 크기에 따라 계산량이 급격히 증가할 수 있음

---

## 3) Legacy Optimization APIs

레거시 API는 단순 간선 배열(`edges: list[list[int]]`) 기반 요청을 사용합니다.

### Common Request Schema (`RequestDto`)

```json
{
  "vertices": [1, 2, 3, 4],
  "edges": [
    [1, 2],
    [2, 3],
    [3, 4]
  ],
  "num_edges": 3
}
```

- `vertices`: 정점 목록
- `edges`: 각 원소가 `[u, v]` 형태인 간선 목록
- `num_edges`: 선택값(0 이상), 일부 구현/실험 목적 필드

### 3.1 `POST /optimize/small-world`

- 알고리즘: Small-world 설정 기반 최적화
- Request: Legacy Common Request
- Response: `ResponseDto`

### 3.2 `POST /optimize/naoto`

- 알고리즘: naoto 방식 최적화
- Request: Legacy Common Request
- Response: `ResponseDto`

---

## 4) Example cURL

### V1 MR2S

```bash
curl -X POST "http://localhost:8000/api/v1/mr2s" \
  -H "Content-Type: application/json" \
  -d "{
    \"edges\": [
      {\"vertices\": [1, 2], \"weight\": 1},
      {\"vertices\": [2, 3], \"weight\": 2},
      {\"vertices\": [3, 1], \"weight\": 1}
    ]
  }"
```

### Legacy Small-World

```bash
curl -X POST "http://localhost:8000/optimize/small-world" \
  -H "Content-Type: application/json" \
  -d "{
    \"vertices\": [1, 2, 3],
    \"edges\": [[1, 2], [2, 3], [1, 3]]
  }"
```

---

## 5) Notes

- 실제 검증 규칙/스키마는 FastAPI의 `/docs`에서 항상 최신 상태를 확인하세요.
- 본 문서는 현재 라우터 구현(`optimization_v1_router`, `optimization_router`) 기준으로 작성되었습니다.
