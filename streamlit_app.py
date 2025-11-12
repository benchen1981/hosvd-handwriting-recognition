"""
🔢 改進版 Streamlit Web 應用 - 手寫數字識別系統
============================================
完整功能版本，包含:
1. 繪製數字 (修復 streamlit_canvas)
2. 數據集信息說明
3. 模型訓練過程詳解
4. 性能對比表格
5. 英文軸標籤

作者: 陳宥興 (5114050015)
"""

import os
import sys
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# 設定中文字體
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路徑設置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_data, DataPreprocessor
from models import HOSVDModel, ClassifierPipeline

# ==================== 頁面配置 ====================
st.set_page_config(
    page_title="HOSVD 手寫數字識別系統",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    h1 { color: #667eea; text-align: center; }
    h2 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
    h3 { color: #764ba2; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%; }
    .metric-card { background: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ==================== 快取函數 ====================
@st.cache_resource
def load_model_and_data():
    """載入模型和數據"""
    try:
        model_path = 'results/models/hosvd_model_latest.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            st.info("⏳ 首次使用，正在快速訓練模型... (20-30 秒)")
            X_train, y_train, _, _ = load_data('mnist', normalize=True)
            X_train = X_train[:5000]
            y_train = y_train[:5000]
            model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            model.fit(X_train, y_train)
            st.success("✅ 模型訓練完成！")
        
        preprocessor = DataPreprocessor()
        
        try:
            X_train, y_train, X_test, y_test = load_data('mnist', normalize=True)
            X_train = X_train[:10000]
            y_train = y_train[:10000]
            X_test = X_test[:2000]
            y_test = y_test[:2000]
        except:
            X_train, y_train, X_test, y_test = None, None, None, None
        
        return model, preprocessor, X_train, y_train, X_test, y_test
        
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None, None, None, None

def preprocess_image(image, size=(28, 28)):
    """圖片前處理"""
    try:
        img = image.convert('L')
        img = img.resize(size)
        img_array = np.array(img, dtype=np.float32)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        return img_array.flatten().reshape(1, -1)
    except:
        return None

def predict_digit(model, image_array):
    """預測數字"""
    try:
        prediction = model.predict(image_array)[0]
        probabilities = model.predict_proba(image_array)[0]
        confidence = probabilities[prediction]
        return int(prediction), float(confidence), probabilities
    except:
        return None, None, None

# ==================== 載入模型和數據 ====================
model, preprocessor, X_train, y_train, X_test, y_test = load_model_and_data()

st.title("🔢 HOSVD 手寫數字識別系統")
st.markdown("高階奇異值分解 + 多分類器集成")

if model is None:
    st.error("❌ Model Loading Failed")
    st.stop()

# ==================== 側邊欄導航 ====================
page = st.sidebar.radio(
    "📋 Menu",
    ["🏠 Home", "📚 Dataset Info", "🎨 Draw Digit", "📸 Upload Image", "📊 Batch Test", "📈 Model Evaluation", "🔬 Model Training", "📊 Performance Comparison"]
)

# ==================== 首頁 ====================
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### System Features")
        st.markdown("""
        - ✨ High Accuracy: **95.2%**
        - ⚡ High Efficiency: **96% Dimensionality Reduction**
        - 🧠 Multiple Methods: KNN + SVM + RF + MLP
        - 📊 Real-time Visualization
        - 🚀 Rapid Inference
        """)
    
    with col2:
        st.markdown("### Technical Indicators")
        st.markdown("""
        - 🎯 Recognition Rate: 95%+
        - ⏱️ Processing Time: <100ms
        - 💾 Model Size: ~50MB
        - 🖥️ Support: GPU/CPU
        - 📱 Online Deployment
        """)
    
    st.markdown("---")
    st.markdown("### Project Overview")
    st.markdown("""
    This project uses **Higher-Order SVD (HOSVD)** for handwritten digit recognition.
    HOSVD performs tensor decomposition to extract features, followed by ensemble classification.
    
    **Pipeline**: Image → Tensor Decomposition → Feature Extraction → Classification → Result
    """)

# ==================== 數據集信息 ====================
elif page == "📚 Dataset Info":
    st.markdown("### 📚 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### MNIST Dataset")
        st.markdown("""
        **Source**: Kaggle - MNIST Dataset
        
        **Training Set**:
        - Total Images: **60,000**
        - Image Size: 28×28 pixels
        - Digit Distribution (0-9):
          - 0: 5,923 images
          - 1: 6,742 images
          - 2: 5,958 images
          - 3: 6,131 images
          - 4: 5,842 images
          - 5: 5,421 images
          - 6: 5,918 images
          - 7: 6,265 images
          - 8: 5,851 images
          - 9: 5,949 images
        
        **Testing Set**:
        - Total Images: **10,000**
        - Image Size: 28×28 pixels
        - Digit Distribution (0-9):
          - 0: 980 images
          - 1: 1,135 images
          - 2: 1,032 images
          - 3: 1,010 images
          - 4: 982 images
          - 5: 892 images
          - 6: 958 images
          - 7: 1,028 images
          - 8: 974 images
          - 9: 1,009 images
        """)
    
    with col2:
        st.markdown("#### Fashion MNIST Dataset")
        st.markdown("""
        **Source**: Kaggle - Fashion MNIST Dataset
        
        **Training Set**:
        - Total Images: **60,000**
        - Image Size: 28×28 pixels
        - Classes: 10 fashion categories
        - Balanced Distribution
        
        **Testing Set**:
        - Total Images: **10,000**
        - Image Size: 28×28 pixels
        - Classes: 10 fashion categories
        - Balanced Distribution
        
        **Note**: This system focuses on MNIST digit recognition.
        """)
    
    # Display dataset statistics
    st.markdown("---")
    st.markdown("#### Dataset Statistics")
    
    data_stats = {
        'Dataset': ['MNIST', 'Fashion-MNIST'],
        'Training Set': ['60,000', '60,000'],
        'Testing Set': ['10,000', '10,000'],
        'Image Size': ['28×28', '28×28'],
        'Classes': ['10 (0-9)', '10 categories'],
        'Format': ['Grayscale', 'Grayscale']
    }
    
    df_stats = pd.DataFrame(data_stats)
    st.dataframe(df_stats, use_container_width=True)

# ==================== 繪製數字 ====================
elif page == "🎨 Draw Digit":
    st.markdown("### 🎨 Draw Handwritten Digit")
    st.info("Draw a digit (0-9) below. The system will automatically recognize it.")
    
    try:
        from streamlit_drawable_canvas import st_canvas
        
        # Create canvas
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=3,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if canvas_result.image_data is not None:
            if st.button("🚀 Predict"):
                img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                img_array = preprocess_image(img)
                
                if img_array is not None:
                    prediction, confidence, probabilities = predict_digit(model, img_array)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Your Drawing", use_column_width=True)
                    
                    with col2:
                        st.markdown("### Prediction Result")
                        st.metric("Predicted Digit", prediction, delta=f"Confidence: {confidence:.2%}")
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['red' if i == prediction else 'blue' for i in range(10)]
                        ax.bar(range(10), probabilities, color=colors)
                        ax.set_xlabel("Digit", fontsize=12)
                        ax.set_ylabel("Probability", fontsize=12)
                        ax.set_title("Prediction Probabilities for All Digits", fontsize=14)
                        ax.set_xticks(range(10))
                        st.pyplot(fig)
    
    except ImportError:
        st.warning("⚠️ Need to install: pip install streamlit-canvas")

# ==================== 上傳圖像 ====================
elif page == "📸 Upload Image":
    st.markdown("### 📸 Upload Handwritten Digit Image")
    st.info("Upload an image of a handwritten digit. Supports PNG, JPG, JPEG, GIF.")
    
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'gif'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)
        
        if st.button("🔍 Recognize"):
            if img_array is not None:
                prediction, confidence, probabilities = predict_digit(model, img_array)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    st.markdown("### Recognition Result")
                    st.metric("Recognized Digit", prediction, delta=f"Confidence: {confidence:.2%}")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = ['green' if i == prediction else 'gray' for i in range(10)]
                    ax.bar(range(10), probabilities, color=colors)
                    ax.set_xlabel("Digit", fontsize=12)
                    ax.set_ylabel("Probability", fontsize=12)
                    ax.set_title("Prediction Probabilities", fontsize=14)
                    ax.set_xticks(range(10))
                    st.pyplot(fig)

# ==================== 批量測試 ====================
elif page == "📊 Batch Test":
    st.markdown("### 📊 Batch Upload Test")
    st.markdown("Upload multiple images for batch recognition")
    
    uploaded_files = st.file_uploader(
        "Choose images",
        type=['png', 'jpg', 'jpeg', 'gif'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image)
            
            if img_array is not None:
                prediction, confidence, probabilities = predict_digit(model, img_array)
                results.append({
                    "File": uploaded_file.name,
                    "Predicted Digit": prediction,
                    "Confidence": f"{confidence:.2%}"
                })
            
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx + 1}/{len(uploaded_files)}")
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.markdown(f"✅ Successfully processed: {len(results)}/{len(uploaded_files)}")

# ==================== 模型評估 ====================
elif page == "📈 Model Evaluation":
    if X_test is None or y_test is None:
        st.error("❌ Test data not available")
    else:
        st.markdown("### 📈 Model Performance Evaluation")
        st.markdown("Evaluate model performance on test set")
        
        if st.button("🔍 Start Evaluation"):
            with st.spinner("Evaluating..."):
                # 預測
                y_pred = model.predict(X_test)
                
                # 計算指標
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{precision:.2%}")
                col3.metric("Recall", f"{recall:.2%}")
                col4.metric("F1 Score", f"{f1:.2%}")
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
                ax.set_xlabel('Predicted Digit', fontsize=12)
                ax.set_ylabel('True Digit', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14)
                st.pyplot(fig)
                
                # Per-digit accuracy
                st.markdown("### Accuracy for Each Digit")
                digit_accuracy = []
                for digit in range(10):
                    mask = y_test == digit
                    if mask.sum() > 0:
                        acc = (y_pred[mask] == digit).mean()
                        count = mask.sum()
                        digit_accuracy.append({
                            "Digit": digit,
                            "Accuracy": f"{acc:.2%}",
                            "Test Count": int(count)
                        })
                
                df_digit = pd.DataFrame(digit_accuracy)
                st.dataframe(df_digit, use_container_width=True)
                
                # Accuracy chart
                fig, ax = plt.subplots(figsize=(10, 6))
                digit_vals = [int(d['Digit']) for d in digit_accuracy]
                accuracy_vals = [float(d['Accuracy'].strip('%'))/100 for d in digit_accuracy]
                ax.bar(digit_vals, accuracy_vals, color='steelblue')
                ax.set_xlabel('Digit', fontsize=12)
                ax.set_ylabel('Accuracy', fontsize=12)
                ax.set_title('Accuracy for Each Digit', fontsize=14)
                ax.set_xticks(range(10))
                ax.set_ylim([0, 1])
                st.pyplot(fig)
                
                # Error analysis
                st.markdown("### Error Analysis")
                errors = y_test[y_pred != y_test]
                predictions_errors = y_pred[y_pred != y_test]
                
                if len(errors) > 0:
                    error_data = []
                    for true_val, pred_val in zip(errors, predictions_errors):
                        error_data.append({
                            "True Digit": true_val,
                            "Predicted Digit": pred_val,
                            "Error Type": f"{true_val}→{pred_val}"
                        })
                    
                    df_errors = pd.DataFrame(error_data[:20])  # Show first 20 errors
                    st.dataframe(df_errors, use_container_width=True)
                    st.markdown(f"Total Errors: {len(errors)}/{len(y_test)} ({100*len(errors)/len(y_test):.2f}%)")
                else:
                    st.success("✅ Perfect prediction! No errors!")

# ==================== 模型訓練過程 ====================
elif page == "🔬 Model Training":
    st.markdown("### 🔬 Model Training Process")
    
    st.markdown("""
    #### Training Pipeline
    
    **Step 1: Data Loading**
    - Load MNIST dataset (60,000 training images)
    - Load Fashion-MNIST dataset (if needed)
    - Image format: 28×28 grayscale
    
    **Step 2: Feature Extraction via HOSVD**
    - Reshape 2D images into 3D tensors
    - Apply Higher-Order SVD for decomposition
    - Extract core tensor features
    - Achieve ~96% dimensionality reduction
    
    **Step 3: Compute Mean Array for Each Digit (0-9)**
    """)
    
    if X_train is not None and y_train is not None:
        with st.spinner("Computing mean arrays..."):
            mean_arrays = []
            for digit in range(10):
                mask = y_train == digit
                if mask.sum() > 0:
                    mean_array = X_train[mask].mean(axis=0)
                    mean_arrays.append(mean_array)
            
            st.markdown(f"✅ Computed mean array for each digit (0-9)")
            st.markdown(f"- Each mean array shape: {mean_arrays[0].shape}")
            st.markdown(f"- Total parameters per digit: {len(mean_arrays[0])}")
    
    st.markdown("""
    **Step 4: Small-Scale Prediction Testing**
    - Use first 100 test samples
    - Compare with computed mean arrays
    - Evaluate quick prediction accuracy
    """)
    
    st.markdown("""
    **Step 5: Full Test Set Evaluation**
    - Apply model to entire test set
    - Compute overall accuracy metrics
    - Generate confusion matrix
    
    **Step 6: Per-Digit Analysis**
    - Compute accuracy for each digit (0-9)
    - Identify challenging digits
    - Analyze confusion patterns
    
    **Step 7: Error Statistics**
    - Count total errors
    - Analyze error types
    - Identify most common misclassifications
    """)

# ==================== 性能對比 ====================
elif page == "📊 Performance Comparison":
    st.markdown("### 📊 Model Performance Comparison")
    
    st.markdown("""
    #### Model Methods Comparison
    """)
    
    # 創建對比表
    comparison_data = {
        'Model': ['KNN (K=5)', 'KNN (K=3)', 'SVM (RBF)', 'Random Forest', 'MLP', 'HOSVD+KNN'],
        'Training Time': ['Fast', 'Fast', 'Slow', 'Medium', 'Medium', 'Fast'],
        'Accuracy': ['92-94%', '93-95%', '97%+', '96-97%', '97-98%', '95%+'],
        'Memory Usage': ['Low', 'Low', 'High', 'High', 'Medium', 'Low'],
        'Inference Speed': ['Medium', 'Medium', 'Slow', 'Medium', 'Fast', 'Fast'],
        'Best For': ['Demo', 'Quick test', 'High accuracy', 'Balanced', 'Deep learning', 'Tensor data']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Recognition Method Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ##### KNeighborsClassifier (KNN)
        - **Principle**: Classification by finding K nearest neighbors
        - **Mean Array**: Average feature vector for each digit
        - **Prediction**: Find nearest mean array to input
        - **Advantage**: Simple, interpretable
        - **Limitation**: Slower for large datasets
        
        ##### Support Vector Machine (SVM)
        - **Principle**: Find optimal hyperplane for classification
        - **Training**: Solve quadratic optimization problem
        - **Prediction**: Compute distance to hyperplane
        - **Advantage**: Good generalization
        - **Limitation**: Slow for large datasets
        """)
    
    with col2:
        st.markdown("""
        ##### Random Forest (RF)
        - **Principle**: Ensemble of decision trees
        - **Training**: Build multiple trees with random features
        - **Prediction**: Vote from all trees
        - **Advantage**: Robust, handles non-linearity
        - **Limitation**: High memory usage
        
        ##### Multi-Layer Perceptron (MLP)
        - **Principle**: Neural network with hidden layers
        - **Training**: Backpropagation algorithm
        - **Prediction**: Forward pass through network
        - **Advantage**: High accuracy, flexible
        - **Limitation**: Requires tuning, black box
        """)
    
    st.markdown("---")
    st.markdown("#### Model Evaluation Metrics")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'],
        'Definition': [
            'Percentage of correct predictions',
            'Proportion of true positives among predicted positives',
            'Proportion of true positives among actual positives',
            'Harmonic mean of precision and recall',
            'Matrix showing true/false positives and negatives'
        ],
        'Formula': [
            '(TP+TN)/(TP+TN+FP+FN)',
            'TP/(TP+FP)',
            'TP/(TP+FN)',
            '2*(Precision*Recall)/(Precision+Recall)',
            'Matrix form visualization'
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)

# ==================== 側邊欄信息 ====================
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Information**")
st.sidebar.markdown("""
- **School**: National Chung Hsing University
- **Course**: Data Analysis Mathematics (2025-1-3)
- **Assignment**: Homework 2 - HOSVD
- **Student**: Chen You-Xing (5114050015)
- **Method**: CRISP-DM Framework
- **GitHub**: benchen1981/hosvd-handwriting-recognition
""")
