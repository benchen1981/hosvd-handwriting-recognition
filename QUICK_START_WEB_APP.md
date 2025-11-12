# 🚀 快速開始 - Web應用

## 立即開始

選擇您喜歡的方式啟動Web應用！

---

## 方式1: Streamlit (最快,推薦新手)

### 安裝
```bash
pip install streamlit pillow numpy scikit-learn matplotlib seaborn pandas
```

### 運行
```bash
cd hosvd_handwriting_recognition
streamlit run streamlit_app.py
```

### 訪問
打開瀏覽器: **http://localhost:8501**

#### 功能
- 📸 上傳手寫數字圖像
- 🎨 直接在應用中繪製數字
- 📊 批量測試多個圖像
- 📈 查看模型性能評估

---

## 方式2: Flask (更專業,推薦生產環境)

### 安裝
```bash
pip install flask pillow numpy scikit-learn matplotlib seaborn
```

### 運行
```bash
cd hosvd_handwriting_recognition
python flask_app.py
```

### 訪問
打開瀏覽器: **http://localhost:5000**

#### 功能
- 📸 上傳單個或批量圖像
- 🔄 RESTful API 支持
- 📊 實時預測和可視化
- 📈 完整的模型評估

#### 命令行使用
```bash
# 預測單個圖像
curl -X POST -F "image=@my_digit.png" http://localhost:5000/api/predict

# 批量預測
curl -X POST -F "images=@digit1.png" -F "images=@digit2.png" \
  http://localhost:5000/api/batch-predict

# 模型評估
curl http://localhost:5000/api/evaluate
```

---

## ⚠️ 注意事項

### 1. 確保模型存在
```bash
# 如果模型不存在,先訓練
python main.py --dataset mnist
```

### 2. 圖像要求
- 格式: PNG, JPG, JPEG, GIF
- 大小: 建議小於 2MB
- 內容: 手寫數字 (0-9)

### 3. 性能提示
| 應用 | 啟動時間 | 響應速度 | 使用場景 |
|------|--------|--------|--------|
| Streamlit | 1-2秒 | 快 | 快速原型、演示、開發 |
| Flask | <1秒 | 非常快 | 生產環境、API集成、部署 |

---

## 🔗 更多文件

- **完整文檔**: [`WEB_APP_README.md`](WEB_APP_README.md)
- **項目首頁**: [`README.md`](README.md)
- **CRISP-DM文檔**: [`CRISP_DM_ProjectMapping.md`](CRISP_DM_ProjectMapping.md)

---

## 📞 常見問題

**Q: 能同時運行兩個應用嗎?**
A: 可以! 在不同終端運行即可。

**Q: 如何更改端口?**
A: 
- Streamlit: 編輯 `~/.streamlit/config.toml` 的 `port` 設置
- Flask: 修改 `python flask_app.py` 為 `python flask_app.py --port 8000`

**Q: 上傳的圖像保存在哪裡?**
A: Flask 應用將上傳的文件保存在 `uploads/` 文件夾

---

**開始使用**: 選擇上面的任一方式並運行! 🎉
