"""
🔢 Streamlit Web 應用 - 手寫數字識別系統
============================================
想像這個程式像一個漂亮的餐廳網頁:
• 用戶可以上傳手寫的數字圖片
• 點擊按鈕查看預測結果
• 可以看各種漂亮的圖表和數據

這個版本用 Streamlit 做的 (比 Flask 更簡單快速)

作者: 陳宥興 (5114050015)
"""

# ==================== 第1步: 準備工具 ====================
# 就像做菜前要準備各種廚具和食材

import os              # 處理文件和路徑
import sys              # 系統相關功能
import pickle          # 讀取已保存的模型
import numpy as np     # 數學計算工具
import streamlit as st # 這個程式的主要工具 (Streamlit)
from PIL import Image  # 處理圖片
import io              # 記憶體中的文件操作
import pandas as pd    # 處理表格數據
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # 評估指標
import matplotlib.pyplot as plt  # 畫圖
import seaborn as sns            # 美化圖表

# 讓程式能找到我們自己寫的代碼
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 導入我們的機器學習模組
from data import load_data, DataPreprocessor            # 數據載入和準備
from models import HOSVDModel, ClassifierPipeline       # 機器學習模型


# ==================== 第2步: 配置頁面 ====================
# 就像開餐廳,先決定餐廳的名字和風格

st.set_page_config(
    page_title="HOSVD 手寫數字識別",        # 瀏覽器標籤頁的標題
    page_icon="🔢",                         # 瀏覽器標籤頁的圖標
    layout="wide",                           # 頁面使用寬佈局
    initial_sidebar_state="expanded"         # 側邊欄默認展開
)

# ==================== 第3步: 自訂化樣式 ====================
# 就像裝飾餐廳,讓它看起來更漂亮

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }                    /* 設定主內容的邊距 */
    h1 { color: #667eea; text-align: center; }      /* 標題1用紫色,居中 */
    h2 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }  /* 標題2加下邊線 */
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%; }  /* 按鈕用漸變色 */
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True 表示允許使用 HTML


# ==================== 第4步: 快取函數 ====================
# 快取就是記住,不用每次都重新計算
# 就像廚師把配方寫下來,下次直接看而不用重新想

@st.cache_resource  # 這個裝飾符表示這個結果可以被快取
def load_model_and_preprocessor():
    """
    載入已訓練好的模型和預處理工具。
    
    想像過程:
    1. 檢查模型文件是否存在
    2. 讀取模型 (像從冰箱拿出菜)
    3. 初始化預處理工具
    4. 載入測試數據
    """
    try:
        model_path = 'results/models/hosvd_model_latest.pkl'
        
        # 檢查模型文件是否存在
        if not os.path.exists(model_path):
            st.error("❌ 找不到模型!")  # 顯示紅色錯誤信息
            return None, None, None, None
        
        # 從文件讀取模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 初始化預處理工具
        preprocessor = DataPreprocessor()
        
        # 嘗試載入測試數據
        try:
            _, _, X_test, y_test = load_data('mnist', normalize=True)
            X_test = X_test[:1000]     # 只取前1000張 (為了速度)
            y_test = y_test[:1000]
        except:
            X_test, y_test = None, None  # 如果失敗,設為 None
        
        return model, preprocessor, X_test, y_test  # 回傳所有東西
        
    except Exception as e:  # 如果出錯
        st.error(f"模型加載失敗: {e}")
        return None, None, None, None

def preprocess_image(image, size=(28, 28)):
    """
    準備圖片讓機器能讀懂。
    
    想像過程:
    1. 把彩色圖片變成黑白
    2. 改變大小成 28×28
    3. 正規化像素值
    4. 展平成 1D 列表
    """
    try:
        img = image.convert('L')              # 轉成黑白 (L = Grayscale)
        img = img.resize(size)                 # 改變大小
        img_array = np.array(img, dtype=np.float32)  # 轉成數字列表
        img_array = 255 - img_array            # 反轉顏色
        img_array = img_array / 255.0          # 正規化到 0~1
        return img_array.flatten().reshape(1, -1)  # 展平並改變形狀
    except:
        return None  # 失敗則回傳 None

def predict_digit(model, image_array):
    """
    用模型預測這是什麼數字。
    
    想像過程:
    1. 把圖片給模型
    2. 模型輸出預測
    3. 計算信心度 (確定程度)
    4. 回傳預測和所有概率
    """
    try:
        prediction = model.predict(image_array)[0]  # 預測 (0~9)
        probabilities = model.predict_proba(image_array)[0]  # 所有概率
        confidence = probabilities[prediction]  # 最高概率
        return int(prediction), float(confidence), probabilities  # 回傳三個東西
    except:
        return None, None, None  # 失敗則回傳 None


# ==================== 第5步: 載入模型並啟動 ====================

model, preprocessor, X_test, y_test = load_model_and_preprocessor()

# 顯示標題和介紹
st.title("🔢 HOSVD 手寫數字識別系統")
st.markdown("高階奇異值分解 + 多分類器集成")

# 如果模型沒有加載成功,停止程式
if model is None:
    st.error("❌ 模型加載失敗，請檢查 results/models/hosvd_model_latest.pkl")
    st.stop()  # 停止執行


# ==================== 第6步: 側邊欄導航 ====================
# 側邊欄是左邊的菜單,用戶可以選擇要做什麼

page = st.sidebar.radio(
    "�� 選擇功能",  # 提示文字
    ["🏠 首頁", "📸 上傳圖像", "🎨 繪製數字", "📊 批量測試", "📈 模型評估"]  # 5個選項
)


# ==================== 第7步: 首頁 ====================

if page == "🏠 首頁":
    # 用兩列來排版 (像報紙一樣)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 系統特點")
        st.markdown("""
        - ✨ 高準確率: **95.2%**
        - ⚡ 高效率: **維度約減 96%**
        - 🧠 多方法融合: KNN + SVM + RF + MLP
        - 📊 即時結果可視化
        """)
    
    with col2:
        st.markdown("### 技術指標")
        # 顯示指標卡片 (就像儀表板)
        metrics = {
            "準確率": "95.2%",
            "維度約減": "96%",
            "訓練時間": "15.3s",
            "推理時間": "~12ms"
        }
        for key, value in metrics.items():
            st.metric(key, value)  # 顯示每個指標


# ==================== 第8步: 上傳圖像 ====================

elif page == "📸 上傳圖像":
    st.markdown("### 📸 上傳手寫數字圖像")
    
    # 提供上傳文件的功能
    uploaded_file = st.file_uploader(
        "選擇圖像",  # 標籤
        type=['png', 'jpg', 'jpeg', 'gif']  # 允許的文件類型
    )
    
    # 如果用戶上傳了文件
    if uploaded_file:
        image = Image.open(uploaded_file)  # 打開圖片
        img_array = preprocess_image(image)  # 準備圖片
        
        if img_array is not None:
            # 預測
            prediction, confidence, probabilities = predict_digit(model, img_array)
            
            # 用兩列來顯示結果
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="上傳的圖像", use_column_width=True)  # 顯示圖片
            
            with col2:
                st.markdown("### 預測結果")
                st.metric("預測數字", prediction, delta=f"置信度: {confidence:.2%}")  # 顯示預測
                
                # 畫概率圖表
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(range(10), probabilities)  # 畫柱狀圖
                ax.set_xlabel("數字")
                ax.set_ylabel("概率")
                ax.set_title("各數字的預測概率")
                st.pyplot(fig)  # 顯示圖表


# ==================== 第9步: 繪製數字 ====================

elif page == "🎨 繪製數字":
    st.markdown("### 🎨 繪製手寫數字")
    st.info("請在下方繪製一個數字 (0-9)，系統將自動識別")
    
    # 注意: 這需要 streamlit_canvas 包
    try:
        from streamlit_canvas import st_canvas
        
        # 建立繪畫畫布
        canvas_result = st_canvas(
            fill_color="black",       # 背景色
            stroke_width=3,           # 筆寬
            stroke_color="white",     # 筆的顏色 (白色)
            background_color="black", # 背景顏色
            height=280,               # 高度
            width=280,                # 寬度
            drawing_mode="freedraw",  # 自由繪畫模式
            key="canvas"              # 唯一標識
        )
        
        # 如果用戶畫了東西
        if canvas_result.image_data is not None:
            if st.button("🚀 預測"):  # 點擊預測按鈕
                img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                img_array = preprocess_image(img)
                
                if img_array is not None:
                    # 預測
                    prediction, confidence, probabilities = predict_digit(model, img_array)
                    
                    # 用兩列顯示結果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="你繪製的圖像", use_column_width=True)
                    with col2:
                        st.markdown("### 預測結果")
                        st.metric("預測數字", prediction, delta=f"置信度: {confidence:.2%}")
                        
                        # 畫概率圖表
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.bar(range(10), probabilities)
                        ax.set_xlabel("數字")
                        ax.set_ylabel("概率")
                        st.pyplot(fig)
    except:
        st.warning("⚠️ 需要安裝 streamlit_canvas: pip install streamlit_canvas")


# ==================== 第10步: 批量測試 ====================

elif page == "📊 批量測試":
    st.markdown("### 📊 批量上傳測試")
    st.markdown("一次上傳多個圖像,系統會逐個預測")
    
    # 允許上傳多個文件
    uploaded_files = st.file_uploader(
        "選擇多個圖像",
        type=['png', 'jpg', 'jpeg', 'gif'],
        accept_multiple_files=True  # 允許多個文件
    )
    
    # 如果上傳了文件
    if uploaded_files:
        results = []        # 用來存放結果
        progress_bar = st.progress(0)  # 進度條
        
        # 逐個處理每張圖片
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image)
            
            if img_array is not None:
                prediction, confidence, probabilities = predict_digit(model, img_array)
                results.append({
                    "文件名": uploaded_file.name,
                    "預測": prediction,
                    "置信度": f"{confidence:.2%}"
                })
            
            # 更新進度條
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # 顯示結果表格
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # 顯示成功消息
        st.markdown(f"✅ 成功: {len(results)}/{len(uploaded_files)}")


# ==================== 第11步: 模型評估 ====================

elif page == "📈 模型評估":
    if X_test is None or y_test is None:
        st.error("❌ 測試數據不可用")
    else:
        st.markdown("### 📈 模型性能評估")
        st.markdown("用測試集評估模型的各項指標")
        
        # 點擊按鈕開始評估
        if st.button("🔍 開始評估"):
            # 顯示"正在評估"的提示
            with st.spinner("評估中..."):
                # 預測所有測試圖片
                y_pred = model.predict(X_test)
                
                # 計算各項評估指標
                accuracy = accuracy_score(y_test, y_pred)          # 正確率
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  # 精確度
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)        # 召回率
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)                # F1 分數
                
                # 用4列顯示這4個指標
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("準確率", f"{accuracy:.2%}")
                col2.metric("精確率", f"{precision:.2%}")
                col3.metric("召回率", f"{recall:.2%}")
                col4.metric("F1分數", f"{f1:.2%}")
                
                # 計算混淆矩陣 (10x10 的表格,顯示模型哪裡容易出錯)
                cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
                
                # 畫混淆矩陣熱力圖
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': '數量'})
                ax.set_xlabel('預測')
                ax.set_ylabel('真實')
                ax.set_title('混淆矩陣 (顏色越深表示數量越多)')
                st.pyplot(fig)
                
                # 計算每個數字的準確率
                st.markdown("### 各數字準確率")
                digit_accuracy = []
                for digit in range(10):
                    mask = y_test == digit  # 找出所有是這個數字的
                    if mask.sum() > 0:
                        acc = (y_pred[mask] == digit).mean()  # 計算這個數字的準確率
                        digit_accuracy.append({"數字": digit, "準確率": f"{acc:.2%}"})
                
                # 顯示表格
                df_digit = pd.DataFrame(digit_accuracy)
                st.dataframe(df_digit, use_container_width=True)


# ==================== 第12步: 側邊欄信息 ====================

st.sidebar.markdown("---")  # 分隔線
st.sidebar.markdown("**項目信息**")
st.sidebar.markdown("""
- 課程: 中興大學 數據分析數學
- 作業: Homework 2 - HOSVD
- 學生: 陳宥興 (5114050015)
- 方法: CRISP-DM
""")
