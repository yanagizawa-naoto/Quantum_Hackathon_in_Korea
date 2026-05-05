# API Endpoint Test Cases

## Test Input: 20-Node Biconnected Graph

- **Nodes:** 20 (0~19)
- **Edges:** 28 (20 cycle edges + 8 chord edges)
- **Structure:** Cycle 0-1-2-...-19-0 with chords: 0-10, 2-8, 3-12, 4-16, 5-15, 6-18, 7-13, 9-17
- **Biconnected:** Yes (no cut vertices)

> **Note:** Brute force is O(2^28) = ~268M combinations for this graph. It will be extremely slow or timeout. Use a smaller graph (< 15 edges) for brute force testing.

## Request Body

```json
{
  "edges": [
    {"vertices": [0, 1], "weight": 1},
    {"vertices": [0, 10], "weight": 1},
    {"vertices": [0, 19], "weight": 1},
    {"vertices": [1, 2], "weight": 1},
    {"vertices": [2, 3], "weight": 1},
    {"vertices": [2, 8], "weight": 1},
    {"vertices": [3, 4], "weight": 1},
    {"vertices": [3, 12], "weight": 1},
    {"vertices": [4, 5], "weight": 1},
    {"vertices": [4, 16], "weight": 1},
    {"vertices": [5, 6], "weight": 1},
    {"vertices": [5, 15], "weight": 1},
    {"vertices": [6, 7], "weight": 1},
    {"vertices": [6, 18], "weight": 1},
    {"vertices": [7, 8], "weight": 1},
    {"vertices": [7, 13], "weight": 1},
    {"vertices": [8, 9], "weight": 1},
    {"vertices": [9, 10], "weight": 1},
    {"vertices": [9, 17], "weight": 1},
    {"vertices": [10, 11], "weight": 1},
    {"vertices": [11, 12], "weight": 1},
    {"vertices": [12, 13], "weight": 1},
    {"vertices": [13, 14], "weight": 1},
    {"vertices": [14, 15], "weight": 1},
    {"vertices": [15, 16], "weight": 1},
    {"vertices": [16, 17], "weight": 1},
    {"vertices": [17, 18], "weight": 1},
    {"vertices": [18, 19], "weight": 1}
  ]
}
```

---

## 1. MR2S (Small World)

```bash
curl -X POST http://localhost:8000/api/v1/mr2s \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [0, 10], "weight": 1},
      {"vertices": [0, 19], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [2, 8], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [3, 12], "weight": 1},
      {"vertices": [4, 5], "weight": 1},
      {"vertices": [4, 16], "weight": 1},
      {"vertices": [5, 6], "weight": 1},
      {"vertices": [5, 15], "weight": 1},
      {"vertices": [6, 7], "weight": 1},
      {"vertices": [6, 18], "weight": 1},
      {"vertices": [7, 8], "weight": 1},
      {"vertices": [7, 13], "weight": 1},
      {"vertices": [8, 9], "weight": 1},
      {"vertices": [9, 10], "weight": 1},
      {"vertices": [9, 17], "weight": 1},
      {"vertices": [10, 11], "weight": 1},
      {"vertices": [11, 12], "weight": 1},
      {"vertices": [12, 13], "weight": 1},
      {"vertices": [13, 14], "weight": 1},
      {"vertices": [14, 15], "weight": 1},
      {"vertices": [15, 16], "weight": 1},
      {"vertices": [16, 17], "weight": 1},
      {"vertices": [17, 18], "weight": 1},
      {"vertices": [18, 19], "weight": 1}
    ]
  }'
```

## 2. Raw SA (NaotoService)

```bash
curl -X POST http://localhost:8000/api/v1/raw-sa \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [0, 10], "weight": 1},
      {"vertices": [0, 19], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [2, 8], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [3, 12], "weight": 1},
      {"vertices": [4, 5], "weight": 1},
      {"vertices": [4, 16], "weight": 1},
      {"vertices": [5, 6], "weight": 1},
      {"vertices": [5, 15], "weight": 1},
      {"vertices": [6, 7], "weight": 1},
      {"vertices": [6, 18], "weight": 1},
      {"vertices": [7, 8], "weight": 1},
      {"vertices": [7, 13], "weight": 1},
      {"vertices": [8, 9], "weight": 1},
      {"vertices": [9, 10], "weight": 1},
      {"vertices": [9, 17], "weight": 1},
      {"vertices": [10, 11], "weight": 1},
      {"vertices": [11, 12], "weight": 1},
      {"vertices": [12, 13], "weight": 1},
      {"vertices": [13, 14], "weight": 1},
      {"vertices": [14, 15], "weight": 1},
      {"vertices": [15, 16], "weight": 1},
      {"vertices": [16, 17], "weight": 1},
      {"vertices": [17, 18], "weight": 1},
      {"vertices": [18, 19], "weight": 1}
    ]
  }'
```

## 3. Brute Force

> **Warning:** 2^28 combinations — will timeout for this graph. Use the small graph below instead.

```bash
curl -X POST http://localhost:8000/api/v1/brute-force \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [0, 10], "weight": 1},
      {"vertices": [0, 19], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [2, 8], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [3, 12], "weight": 1},
      {"vertices": [4, 5], "weight": 1},
      {"vertices": [4, 16], "weight": 1},
      {"vertices": [5, 6], "weight": 1},
      {"vertices": [5, 15], "weight": 1},
      {"vertices": [6, 7], "weight": 1},
      {"vertices": [6, 18], "weight": 1},
      {"vertices": [7, 8], "weight": 1},
      {"vertices": [7, 13], "weight": 1},
      {"vertices": [8, 9], "weight": 1},
      {"vertices": [9, 10], "weight": 1},
      {"vertices": [9, 17], "weight": 1},
      {"vertices": [10, 11], "weight": 1},
      {"vertices": [11, 12], "weight": 1},
      {"vertices": [12, 13], "weight": 1},
      {"vertices": [13, 14], "weight": 1},
      {"vertices": [14, 15], "weight": 1},
      {"vertices": [15, 16], "weight": 1},
      {"vertices": [16, 17], "weight": 1},
      {"vertices": [17, 18], "weight": 1},
      {"vertices": [18, 19], "weight": 1}
    ]
  }'
```

---

## Test Input: 15-Node Biconnected Graph

- **Nodes:** 15 (0~14)
- **Edges:** 17 (15 cycle edges + 2 chord edges)
- **Structure:** Cycle 0-1-2-...-14-0 with chords: 0-7, 4-11
- **Biconnected:** Yes (no cut vertices)
- **Brute Force:** 2^17 = 131,072 combinations — feasible but slow

### Request Body

```json
{
  "edges": [
    {"vertices": [0, 1], "weight": 1},
    {"vertices": [0, 7], "weight": 1},
    {"vertices": [0, 14], "weight": 1},
    {"vertices": [1, 2], "weight": 1},
    {"vertices": [2, 3], "weight": 1},
    {"vertices": [3, 4], "weight": 1},
    {"vertices": [4, 5], "weight": 1},
    {"vertices": [4, 11], "weight": 1},
    {"vertices": [5, 6], "weight": 1},
    {"vertices": [6, 7], "weight": 1},
    {"vertices": [7, 8], "weight": 1},
    {"vertices": [8, 9], "weight": 1},
    {"vertices": [9, 10], "weight": 1},
    {"vertices": [10, 11], "weight": 1},
    {"vertices": [11, 12], "weight": 1},
    {"vertices": [12, 13], "weight": 1},
    {"vertices": [13, 14], "weight": 1}
  ]
}
```

### 1. MR2S

```bash
curl -X POST http://localhost:8000/api/v1/mr2s \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [0, 7], "weight": 1},
      {"vertices": [0, 14], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [4, 5], "weight": 1},
      {"vertices": [4, 11], "weight": 1},
      {"vertices": [5, 6], "weight": 1},
      {"vertices": [6, 7], "weight": 1},
      {"vertices": [7, 8], "weight": 1},
      {"vertices": [8, 9], "weight": 1},
      {"vertices": [9, 10], "weight": 1},
      {"vertices": [10, 11], "weight": 1},
      {"vertices": [11, 12], "weight": 1},
      {"vertices": [12, 13], "weight": 1},
      {"vertices": [13, 14], "weight": 1}
    ]
  }'
```

### 2. Raw SA

```bash
curl -X POST http://localhost:8000/api/v1/raw-sa \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [0, 7], "weight": 1},
      {"vertices": [0, 14], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [4, 5], "weight": 1},
      {"vertices": [4, 11], "weight": 1},
      {"vertices": [5, 6], "weight": 1},
      {"vertices": [6, 7], "weight": 1},
      {"vertices": [7, 8], "weight": 1},
      {"vertices": [8, 9], "weight": 1},
      {"vertices": [9, 10], "weight": 1},
      {"vertices": [10, 11], "weight": 1},
      {"vertices": [11, 12], "weight": 1},
      {"vertices": [12, 13], "weight": 1},
      {"vertices": [13, 14], "weight": 1}
    ]
  }'
```

### 3. Brute Force

```bash
curl -X POST http://localhost:8000/api/v1/brute-force \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [0, 7], "weight": 1},
      {"vertices": [0, 14], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [4, 5], "weight": 1},
      {"vertices": [4, 11], "weight": 1},
      {"vertices": [5, 6], "weight": 1},
      {"vertices": [6, 7], "weight": 1},
      {"vertices": [7, 8], "weight": 1},
      {"vertices": [8, 9], "weight": 1},
      {"vertices": [9, 10], "weight": 1},
      {"vertices": [10, 11], "weight": 1},
      {"vertices": [11, 12], "weight": 1},
      {"vertices": [12, 13], "weight": 1},
      {"vertices": [13, 14], "weight": 1}
    ]
  }'
```

---

## Small Graph for Brute Force Validation

**5-Node Biconnected Graph** (7 edges, 2^7 = 128 combinations)

```
Nodes: {0, 1, 2, 3, 4}
Structure: Cycle 0-1-2-3-4-0 with chords 0-2, 1-3
```

```bash
curl -X POST http://localhost:8000/api/v1/brute-force \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [0, 4], "weight": 1},
      {"vertices": [0, 2], "weight": 1},
      {"vertices": [1, 3], "weight": 1}
    ]
  }'
```

Compare with MR2S and Raw SA using the same input:

```bash
curl -X POST http://localhost:8000/api/v1/mr2s \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [0, 4], "weight": 1},
      {"vertices": [0, 2], "weight": 1},
      {"vertices": [1, 3], "weight": 1}
    ]
  }'
```

```bash
curl -X POST http://localhost:8000/api/v1/raw-sa \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"vertices": [0, 1], "weight": 1},
      {"vertices": [1, 2], "weight": 1},
      {"vertices": [2, 3], "weight": 1},
      {"vertices": [3, 4], "weight": 1},
      {"vertices": [0, 4], "weight": 1},
      {"vertices": [0, 2], "weight": 1},
      {"vertices": [1, 3], "weight": 1}
    ]
  }'
```

---

## Expected Results

### 20-Node Graph (MR2S & Raw SA only)

| Metric | Expected Range |
|--------|---------------|
| `bidirectional_graph_score` | ~500-600 (fixed, same for all) |
| `optimized_graph_score` (MR2S) | Higher than bidirectional — direction adds path cost |
| `optimized_graph_score` (Raw SA) | Similar range to MR2S, may differ due to different cost function |

### 15-Node Graph (All three services)

| Metric | Expected |
|--------|----------|
| `bidirectional_graph_score` | ~250-350 (fixed, same for all 3) |
| `optimized_graph_score` (Brute Force) | Global optimum — may take a few minutes |
| `optimized_graph_score` (MR2S) | >= Brute Force score |
| `optimized_graph_score` (Raw SA) | >= Brute Force score |

This is the largest graph where all three services can be compared. Brute force will be slow (2^17 = 131K combinations) but should complete.

### 5-Node Graph (All three services)

| Metric | Expected |
|--------|----------|
| `bidirectional_graph_score` | Fixed value (same for all 3) |
| `optimized_graph_score` (Brute Force) | Global optimum — lowest possible |
| `optimized_graph_score` (MR2S) | >= Brute Force score |
| `optimized_graph_score` (Raw SA) | >= Brute Force score |

Brute Force provides the ground truth. If MR2S or Raw SA match it, they found the optimal solution. The gap shows how close each heuristic gets to optimal.
