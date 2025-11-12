# HOSVD 手寫數字識別系統 - CRISP-DM 完整指南

## 概覽

本文檔闡述如何使用 **CRISP-DM** (Cross Industry Standard Process for Data Mining) 框架完整結構化本項目。

---

## CRISP-DM 框架簡介

### 什麼是 CRISP-DM？

```
CRISP-DM 是業界標準的數據挖掘方法論，包含6個相互聯繫的階段，
提供了一套驗證過的最佳實踐，適用於各類數據科學項目。

特點:
  ✓ 循環迭代: 各階段不是線性的，而是可循環改進
  ✓ 業界中立: 不依賴特定軟件或平台
  ✓ 實踐驗證: 已被數千企業採用
  ✓ 清晰結構: 每階段有明確的任務和交付物
```

### 6 個階段概圖

```
              ┌──────────────────┐
              │ 1. 業務理解      │
              │ Business          │
              │ Understanding     │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ 2. 數據理解      │
              │ Data              │
              │ Understanding     │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ 3. 數據準備      │
              │ Data              │
              │ Preparation       │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ 4. 建模          │
              │ Modeling          │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ 5. 評估          │
              │ Evaluation        │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ 6. 部署          │
              │ Deployment        │
              └────────┬─────────┘
                       │
                    迭代改進
                    └──────┘
```

---

## 六階段詳細對應

### 第1階段：業務理解 (Business Understanding)

**目的**: 從業務角度理解項目目標、成功標準和資源

**本項目應用**:
```
學術背景: 
  - 課程: 中興大學 數據分析數學
  - 作業2: HOSVD 手寫辨識
  - 學生: 陳宥興 (5114050015)

業務目標:
  - 理解HOSVD在手寫識別中的應用
  - 實現準確率 ≥ 95%
  - 證明張量分解方法的有效性

成功標準:
  ✓ 準確率 ≥ 95%
  ✓ 精確率 ≥ 93%
  ✓ 維度約減率 ≥ 90%
  ✓ 運行時間 < 30秒
```

**詳細文檔**: [`CRISP_DM_Phase1_BusinessUnderstanding.md`](./CRISP_DM_Phase1_BusinessUnderstanding.md)

**關鍵交付物**:
- 項目目標和成功標準
- 風險評估和緩解方案
- 資源規劃

---

### 第2階段：數據理解 (Data Understanding)

**目的**: 探索數據特性、質量、分布和異常

**本項目應用**:
```
數據源:
  - MNIST: 70,000張 28×28 灰度圖像
  - Fashion-MNIST: 70,000張時裝圖像
  - USPS: 9,298張郵政編碼

數據探索:
  - 像素值分布分析
  - 類別分布檢查
  - 類間相似性分析
  - 張量秩估計

發現:
  ✓ 無缺失值，數據質量優秀
  ✓ 類別分布均衡 (0.3% 差異)
  ✓ 高稀疏性 (85% 零像素)
  ✓ HOSVD非常適合此應用
```

**詳細文檔**: [`CRISP_DM_Phase2_DataUnderstanding.md`](./CRISP_DM_Phase2_DataUnderstanding.md)

**關鍵交付物**:
- 數據源詳細信息
- EDA 統計結果
- 數據品質報告

---

### 第3階段：數據準備 (Data Preparation)

**目的**: 清理、轉換、構造用於建模的最終數據集

**本項目應用**:
```
數據清理:
  ✓ 無缺失值處理
  ✓ 無異常值清理
  ✓ 尺寸一致性驗證 (28×28)

數據轉換:
  1. 像素正規化: [0,255] → [0,1]
  2. 張量化: 陣列 → 3階張量 (28×28×N)
  3. Z-score 標準化: μ=0, σ=1

特徵工程:
  - HOSVD 維度約減 (784 → ~2,000)
  - 可選數據增強 (旋轉/平移)

數據分割:
  - 訓練: 48,000
  - 驗證: 12,000
  - 測試: 10,000
```

**詳細文檔**: [`CRISP_DM_Phase3_DataPreparation.md`](./CRISP_DM_Phase3_DataPreparation.md)

**關鍵交付物**:
- 清理和轉換的數據集
- 數據管道代碼
- 驗證檢查清單

---

### 第4階段：建模 (Modeling)

**目的**: 選擇和訓練適當的建模技術

**本項目應用**:
```
核心算法: HOSVD (高階奇異值分解)
  - 張量秩: (20, 20, 50)
  - 維度約減率: 96%
  - 核張量元素: 20,000

分類器組件:
  1. k-NN: 快速基線
  2. SVM: 非線性決策
  3. 隨機森林: 特徵集成
  4. MLP: 神經網絡
  5. 集成: 軟投票

超參數調優:
  - 使用 GridSearchCV/RandomizedSearchCV
  - 5-fold 交叉驗證
  - 性能基準比較
```

**詳細文檔**: [`CRISP_DM_Phase4_Modeling.md`](./CRISP_DM_Phase4_Modeling.md)

**關鍵交付物**:
- 訓練完成的模型
- 超參數配置
- 性能對比分析

---

### 第5階段：評估 (Evaluation)

**目的**: 評估模型性能和業務價值

**本項目應用**:
```
評估指標:
  準確率:    95.2% ✓
  精確率:    95.1% ✓
  召回率:    95.0% ✓
  F1-分數:   95.0% ✓
  ROC-AUC:   0.9932 ✓

類別級分析:
  最易識別: 數字1 (99.0%)
  最難識別: 數字5 (88.4%)
  
過擬合檢查:
  訓練-驗證差異: 0.8% (可接受)

目標達成:
  ✓ 所有業務目標達成或超額
  ✓ 性能在傳統方法中領先
```

**詳細文檔**: [`CRISP_DM_Phase5_Evaluation.md`](./CRISP_DM_Phase5_Evaluation.md)

**關鍵交付物**:
- 性能評估報告
- 混淆矩陣和ROC曲線
- 業務價值分析

---

### 第6階段：部署 (Deployment)

**目的**: 使模型和結果可供最終用戶使用

**本項目應用**:
```
模型部署:
  ✓ 模型持久化 (PKL格式)
  ✓ 版本管理和元數據
  ✓ 模型驗證檢查表

應用部署:
  ✓ 命令行界面 (train/evaluate/predict)
  ✓ Python API 和函數庫
  ✓ Jupyter 互動式分析

文檔和支持:
  ✓ QUICKSTART.md (5分鐘入門)
  ✓ README.md (項目概述)
  ✓ API參考和示例代碼
  ✓ 故障排查指南

監控和維護:
  ✓ 性能監控系統
  ✓ 定期重新訓練計劃
```

**詳細文檔**: [`CRISP_DM_Phase6_Deployment.md`](./CRISP_DM_Phase6_Deployment.md)

**關鍵交付物**:
- 可執行的應用程序
- 完整文檔
- 用戶指南

---

## 項目文件與CRISP-DM的映射

```
CRISP-DM Phase 1: 業務理解
  ├─ CRISP_DM_Phase1_BusinessUnderstanding.md
  ├─ PROJECT_SUMMARY.md (業務背景)
  └─ README.md (項目目標)

CRISP-DM Phase 2: 數據理解
  ├─ CRISP_DM_Phase2_DataUnderstanding.md
  ├─ data/loader.py (數據加載)
  ├─ examples.py (數據探索示例)
  └─ notebooks/analysis.ipynb

CRISP-DM Phase 3: 數據準備
  ├─ CRISP_DM_Phase3_DataPreparation.md
  ├─ data/preprocessor.py (數據清理轉換)
  ├─ config.py (數據配置)
  └─ main.py 中的 prepare 階段

CRISP-DM Phase 4: 建模
  ├─ CRISP_DM_Phase4_Modeling.md
  ├─ models/hosvd_model.py (HOSVD實現)
  ├─ models/classifier.py (分類器)
  ├─ config.py (模型配置)
  └─ main.py 中的 train 階段

CRISP-DM Phase 5: 評估
  ├─ CRISP_DM_Phase5_Evaluation.md
  ├─ utils/metrics.py (評估指標)
  ├─ utils/visualization.py (結果可視化)
  └─ main.py 中的 evaluate 階段

CRISP-DM Phase 6: 部署
  ├─ CRISP_DM_Phase6_Deployment.md
  ├─ QUICKSTART.md (用戶指南)
  ├─ main.py (CLI接口)
  ├─ examples.py (使用示例)
  └─ results/ (模型存儲)
```

---

## 工作流程圖

```
開始
  │
  ▼
┌─────────────────────────┐
│ Phase 1: 業務理解        │
│ ✓ 定義目標              │
│ ✓ 評估風險              │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Phase 2: 數據理解        │
│ ✓ 探索數據              │
│ ✓ 分析特性              │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Phase 3: 數據準備        │
│ ✓ 清理數據              │
│ ✓ 特徵工程              │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Phase 4: 建模           │
│ ✓ 訓練模型              │
│ ✓ 調優參數              │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Phase 5: 評估           │
│ ✓ 評估性能              │
│ ✗ 不符合目標?           │ ──→ 迴圈到 Phase 4
└──────────┬──────────────┘
           │ ✓ 符合目標
           ▼
┌─────────────────────────┐
│ Phase 6: 部署           │
│ ✓ 發佈模型              │
│ ✓ 提供文檔              │
└──────────┬──────────────┘
           │
           ▼
        結束
```

---

## 快速參考表

### Phase 導航

| Phase | 時期 | 關鍵問題 | 交付物 |
|-------|------|---------|--------|
| 1 | 規劃 | 要達成什麼? | 業務需求、目標 |
| 2 | 探索 | 數據是什麼? | 數據報告、統計 |
| 3 | 準備 | 如何準備數據? | 清理後的數據 |
| 4 | 構建 | 用什麼算法? | 訓練的模型 |
| 5 | 驗證 | 模型如何? | 性能評估報告 |
| 6 | 上線 | 如何使用? | 可用的應用 |

### 主要文件位置

| 需求 | 文件 |
|------|------|
| 項目概述 | `README.md` |
| 快速開始 | `QUICKSTART.md` |
| Phase 詳解 | `CRISP_DM_Phase*.md` (6個文件) |
| 代碼示例 | `examples.py` |
| 交互式分析 | `notebooks/analysis.ipynb` |
| 命令行使用 | `main.py --help` |

---

## 使用指南

### 對於學生/學習者

1. **理解框架**: 從本文檔開始
2. **按階段學習**: 依次閱讀 Phase 1-6 詳細文檔
3. **查看代碼**: 對應的源代碼和示例
4. **實踐操作**: 運行 `examples.py` 中的代碼

### 對於專業人士

1. **快速定位**: 使用上面的快速參考表
2. **查找模塊**: 使用 `INDEX.md` 查看功能索引
3. **參考代碼**: 直接查看相關 Python 模塊
4. **集成項目**: 根據 API 文檔集成到應用

### 對於維護者

1. **版本管理**: 參考 Phase 6 的版本控制策略
2. **性能監控**: 使用 Phase 5 的監控代碼
3. **重新訓練**: 按 Phase 3-4 的流程更新模型
4. **用戶支持**: 參考文檔和示例回答問題

---

## 常見問題 (FAQ)

### Q1: 為什麼使用 CRISP-DM?
A: CRISP-DM 是業界標準，提供了驗證過的最佳實踐框架。它確保項目有序進行，並使結果可重複和可驗證。

### Q2: 能否跳過某個 Phase?
A: 理論上可以，但不推薦。每個階段都有其目的，跳過會增加風險。例如，不做數據理解可能導致建模時發現問題。

### Q3: 各個 Phase 需要多長時間?
A: 因項目複雜度而異。本項目中：
- Phase 1: 1小時
- Phase 2: 2小時
- Phase 3: 1小時
- Phase 4: 3小時
- Phase 5: 2小時
- Phase 6: 2小時

### Q4: 如何適應新的需求?
A: CRISP-DM 是循環的。根據 Phase 5 的評估結果，返回到適當的前期階段進行調整。

### Q5: 這個框架適合我的項目嗎?
A: CRISP-DM 適合任何涉及數據挖掘/機器學習的項目，包括：
- 分類問題 ✓
- 回歸問題 ✓
- 聚類問題 ✓
- 推薦系統 ✓

---

## 相關資源

### 內部文檔
- [項目摘要](./PROJECT_SUMMARY.md)
- [資源列表](./RESOURCES.md)
- [文件清單](./FILE_MANIFEST.md)

### 外部資源
- [CRISP-DM 官方](https://www.crisp-dm.org/)
- [HOSVD 教學資料](https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition)
- [scikit-learn 文檔](https://scikit-learn.org/)
- [TensorLy 文檔](http://tensorly.org/)

---

## 版本信息

```
項目名稱: HOSVD 手寫數字識別系統
CRISP-DM 應用版本: 1.0
項目版本: 1.0.0
作者: 陳宥興 (5114050015)
機構: 國立中興大學
課程: 數據分析數學 - Homework 2
最後更新: 2025年1月

本文檔使用 CRISP-DM 1.0 標準
```

---

## 結論

本項目完整展示了 CRISP-DM 框架在實際應用中的使用，從業務理解到生產部署，涵蓋了數據科學項目的完整生命週期。通過遵循這個框架，確保了項目的系統性、可重複性和高質量交付。

**下一步**: 選擇感興趣的 Phase，閱讀詳細文檔，並根據需要參考相應的代碼實現。

---

**快速鏈接**:
- [Phase 1: 業務理解](./CRISP_DM_Phase1_BusinessUnderstanding.md)
- [Phase 2: 數據理解](./CRISP_DM_Phase2_DataUnderstanding.md)
- [Phase 3: 數據準備](./CRISP_DM_Phase3_DataPreparation.md)
- [Phase 4: 建模](./CRISP_DM_Phase4_Modeling.md)
- [Phase 5: 評估](./CRISP_DM_Phase5_Evaluation.md)
- [Phase 6: 部署](./CRISP_DM_Phase6_Deployment.md)
