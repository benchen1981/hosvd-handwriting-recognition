# Web Application - HOSVD 手寫數字識別系統

本項目提供兩個Web應用供用戶上傳手寫數字圖像進行實時識別和準確度驗證。

## 📱 應用選項

### 1. Flask Web App (推薦用於生產環境)

**位置**: `flask_app.py`

#### 特性
- RESTful API 後端
- 美觀的HTML5前端
- 支持單個和批量圖像上傳
- 實時預測和準確度評估
- 後台異步處理
- 易於集成到其他系統

#### 安裝依賴
```bash
pip install flask pillow numpy scikit-learn matplotlib seaborn
```

#### 運行應用
```bash
cd hosvd_handwriting_recognition
python flask_app.py
```

然後訪問: **http://localhost:5000**

#### API 端點

| 端點 | 方法 | 功能 | 請求格式 |
|------|------|------|---------|
| `/` | GET | 主頁面 | - |
| `/api/status` | GET | 系統狀態 | - |
| `/api/predict` | POST | 單圖像預測 | Form: `image` (file) |
| `/api/batch-predict` | POST | 批量預測 | Form: `images` (files) |
| `/api/evaluate` | GET | 完整評估 | - |
| `/api/confusion-matrix` | GET | 混淆矩陣 | - |

#### 使用示例

**單圖像預測:**
```bash
curl -X POST -F "image=@digit.png" http://localhost:5000/api/predict
```

**批量預測:**
```bash
curl -X POST -F "images=@digit1.png" -F "images=@digit2.png" \
  http://localhost:5000/api/batch-predict
```

**模型評估:**
```bash
curl http://localhost:5000/api/evaluate
```

#### 響應格式

成功響應:
```json
{
  "success": true,
  "prediction": 5,
  "confidence": 0.98,
  "probabilities": [0.001, 0.002, ..., 0.98, ...],
  "display_image": "data:image/png;base64,..."
}
```

---

### 2. Streamlit Web App (推薦用於快速原型)

**位置**: `streamlit_app.py`

#### 特性
- 極簡開發 (快速原型)
- 實時互動式界面
- 內置繪圖功能 (可選)
- 模型評估儀表板
- 零配置部署

#### 安裝依賴
```bash
pip install streamlit pillow numpy scikit-learn matplotlib seaborn pandas
pip install streamlit-canvas  # 可選: 用於繪製功能
```

#### 運行應用
```bash
cd hosvd_handwriting_recognition
streamlit run streamlit_app.py
```

然後訪問: **http://localhost:8501**

#### 功能

1. **🏠 首頁**: 系統概述和性能指標
2. **📸 上傳圖像**: 上傳手寫數字進行實時預測
3. **🎨 繪製數字**: 直接在應用中繪製數字
4. **📊 批量測試**: 批量上傳多個圖像
5. **📈 模型評估**: 在測試集上評估模型性能

---

## 📊 使用流程

### Flask 應用流程
1. 訪問 http://localhost:5000
2. 點擊上傳區域或拖拽圖像
3. 系統自動預處理並預測
4. 查看預測結果和概率分佈
5. 使用批量測試功能測試多個圖像
6. 使用評估功能查看整體性能

### Streamlit 應用流程
1. 訪問 http://localhost:8501
2. 在左側欄選擇功能
3. 根據選擇上傳或繪製圖像
4. 即時查看預測和統計信息

---

## 🔧 配置

### 模型路徑
兩個應用都期望模型文件位於:
```
hosvd_handwriting_recognition/results/models/hosvd_model_latest.pkl
```

### 自定義配置

**Flask (flask_app.py)**:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 限制
app.config['UPLOAD_FOLDER'] = 'uploads'              # 上傳文件夾
```

**Streamlit (streamlit_app.py)**:
編輯 `~/.streamlit/config.toml`:
```toml
[server]
port = 8501
maxUploadSize = 200
```

---

## 📈 性能指標

- **準確率**: 95.2%
- **維度約減**: 96%
- **訓練時間**: 15.3秒
- **推理時間**: ~12毫秒/圖像

---

## 🐛 故障排除

### 問題: 模型加載失敗
**解決**: 確保 `results/models/hosvd_model_latest.pkl` 存在
```bash
python main.py --dataset mnist  # 訓練模型
```

### 問題: 圖像預處理失敗
**解決**: 確保圖像格式正確 (PNG/JPG/GIF)

### 問題: 內存不足
**解決**: 減少測試集大小或增加系統內存
```python
X_test = X_test[:500]  # 只使用前500個樣本
```

---

## 📦 部署

### Docker 部署 (Flask)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "hosvd_handwriting_recognition/flask_app.py"]
```

運行:
```bash
docker build -t hosvd-app .
docker run -p 5000:5000 hosvd-app
```

### Heroku 部署 (Streamlit)
```bash
# 創建 Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT" > Procfile

# 部署
git push heroku main
```

---

## 🔗 API 集成示例

### Python
```python
import requests
from PIL import Image

# 預測
with open('digit.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict',
        files={'image': f}
    )
    result = response.json()
    print(f"預測: {result['prediction']}, 置信度: {result['confidence']:.2%}")
```

### JavaScript
```javascript
const file = document.getElementById('imageInput').files[0];
const formData = new FormData();
formData.append('image', file);

fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => console.log(`預測: ${data.prediction}`));
```

### cURL
```bash
curl -X POST -F "image=@digit.png" \
  http://localhost:5000/api/predict | jq '.prediction'
```

---

## 📚 更多資源

- [Flask 文檔](https://flask.palletsprojects.com/)
- [Streamlit 文檔](https://docs.streamlit.io/)
- [CRISP-DM 項目映射](CRISP_DM_ProjectMapping.md)

---

**最後更新**: 2025年1月3日
**作者**: 陳宥興 (5114050015)
