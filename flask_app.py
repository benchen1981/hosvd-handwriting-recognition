"""
🔢 Flask Web 應用 - 手寫數字識別系統
============================================
想像這個程式是一個數字識別機器:
• 用戶把寫好的數字照片給它 (上傳圖片)
• 機器分析這個照片 (模型預測)
• 機器告訴用戶這是幾 (返回結果)

作者: 陳宥興 (5114050015)
"""

# ==================== 第1步: 準備工具 ====================
# 就像做菜前要準備各種廚具和食材

# 這些是系統工具,用來處理文件、時間、路徑
import os, sys, json, pickle, base64, io, traceback
from datetime import datetime
from pathlib import Path

# 這些是數學和圖片處理工具
import numpy as np              # 用來做數學計算 (像計算機一樣)
import matplotlib.pyplot as plt # 用來畫圖
import seaborn as sns           # 用來美化圖表
from PIL import Image           # 用來處理和改變圖片

# 這些是 Web 伺服器和機器學習的工具
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 讓程式能夠找到我們自己寫的代碼
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 導入我們自己寫的機器學習模組
from data import load_data, DataPreprocessor  # 載入和準備數據
from models import HOSVDModel, ClassifierPipeline  # 機器學習模型


# ==================== 第2步: 建立 Web 伺服器 ====================
# 就像開一家餐廳，需要決定位置、規則、容量

app = Flask(__name__, template_folder='templates')  # 建立 Web 伺服器

# 設定伺服器的規則:
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上傳檔案最大 16MB (就像容量限制)
app.config['UPLOAD_FOLDER'] = 'uploads'  # 上傳的圖片存放在 uploads 文件夾

# 建立存放文件的文件夾 (如果不存在的話)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ==================== 第3步: 建立全域變數 ====================
# 這些變數像是餐廳的"狀態",整個程式都能看到

model = None                # 儲存機器學習模型 (最重要的工具)
preprocessor = None         # 儲存圖片預處理工具
X_test = None              # 儲存測試用的圖片數據
y_test = None              # 儲存測試用的正確答案
model_ready = False        # 記錄模型是否已經準備好


# ==================== 第4步: 載入模型的函數 ====================
# 這個函數就像"開門營業"前的準備工作

def load_model_files():
    """
    載入已訓練好的機器學習模型。
    
    想像過程:
    1. 檢查模型文件是否存在
    2. 如果存在,就讀取模型
    3. 準備好圖片預處理工具
    4. 載入測試數據
    """
    global model, preprocessor, X_test, y_test, model_ready  # 使用全域變數
    
    try:
        model_path = 'results/models/hosvd_model_latest.pkl'
        
        # 檢查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"找不到模型,位置: {model_path}")
            model_ready = False
            return False
        
        # 從文件讀取模型 (就像從冰箱拿出已做好的菜)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ 模型載入成功: {model_path}")
        
        # 初始化圖片預處理工具
        preprocessor = DataPreprocessor()
        print("✓ 圖片預處理工具準備好了")
        
        # 載入測試用的圖片和正確答案
        try:
            _, _, X_test_temp, y_test_temp = load_data('mnist', normalize=True)
            X_test = X_test_temp[:1000]  # 只取前1000張 (為了速度)
            y_test = y_test_temp[:1000]
            print(f"✓ 測試數據已載入: {X_test.shape[0]} 張圖片")
        except Exception as e:
            print(f"⚠️ 警告: 測試數據載入失敗: {e}")
        
        model_ready = True  # 模型準備好了！
        return True
        
    except Exception as e:
        print(f"✗ 模型載入失敗: {e}")
        traceback.print_exc()
        model_ready = False
        return False

def preprocess_image(image_array, size=(28, 28)):
    """
    準備圖片讓機器能讀懂。
    
    想像過程:
    1. 把彩色圖片變成黑白 (模型只看懂黑白)
    2. 改變圖片大小成 28×28 (統一規格)
    3. 把像素值改到 0~1 之間 (正規化)
    4. 把 2D 圖片變成 1D 列表 (給模型用)
    """
    try:
        # 把圖片轉成 PIL Image 格式
        if isinstance(image_array, np.ndarray):
            img = Image.fromarray(image_array.astype('uint8'))
        else:
            img = image_array
        
        # 把圖片變成黑白 (灰度)
        img = img.convert('L')
        
        # 改變圖片大小成 28×28 像素
        img = img.resize(size)
        
        # 把圖片轉成數字列表
        img_array = np.array(img, dtype=np.float32)
        
        # 反轉顏色 (白色數字變黑色背景)
        img_array = 255 - img_array
        
        # 正規化: 把像素值從 0~255 變成 0~1
        img_array = img_array / 255.0
        
        # 展平成 1D 列表 (784 個數字)
        img_array = img_array.flatten().reshape(1, -1)
        
        return img_array
    except Exception as e:
        print(f"圖片預處理失敗: {e}")
        return None

def image_to_display(image_array, size=(28, 28)):
    """
    把圖片轉成網頁能顯示的格式 (Base64)。
    
    想像: 就像把圖片編碼成一長串密碼,網頁再把密碼解碼回圖片
    """
    try:
        # 確保圖片大小正確
        if isinstance(image_array, np.ndarray):
            img_array = image_array.reshape(size) if image_array.ndim == 1 else image_array
        else:
            img_array = np.array(image_array).reshape(size)
        
        # 把像素值從 0~1 變回 0~255
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        
        # 建立圖片物件
        img = Image.fromarray(img_array, mode='L')
        
        # 把圖片保存到記憶體 (不是硬碟)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # 把圖片編碼成長文字
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # 回傳可以在網頁上顯示的格式
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"圖片轉換失敗: {e}")
        return None

def predict_digit(image_array):
    """
    用機器學習模型預測這是什麼數字。
    
    想像過程:
    1. 把圖片給模型
    2. 模型輸出 10 個數字 (0~9 的概率)
    3. 找出最高的概率
    4. 那個就是預測的數字
    """
    try:
        if model is None:
            return None, None, None  # 如果沒有模型,回傳 None
        
        # 預測 (模型分析圖片)
        prediction = model.predict(image_array)[0]  # [0] 是取第一個結果
        
        # 得到每個數字的概率 (0有 95% 可能, 1有 2% 可能...)
        probabilities = model.predict_proba(image_array)[0]
        
        # 取最高的概率 (信心度)
        confidence = probabilities[prediction]
        
        # 回傳: 預測的數字, 信心度, 所有概率
        return int(prediction), float(confidence), probabilities.tolist()
    except Exception as e:
        print(f"預測失敗: {e}")
        traceback.print_exc()
        return None, None, None


# ==================== 第5步: 定義網頁路由 ====================
# 路由就像餐廳的"菜單",告訴用戶可以做什麼

@app.route('/')
def index():
    """顯示主頁面"""
    return render_template('index.html', model_ready=model_ready)

@app.route('/api/status')
def api_status():
    """檢查系統是否準備好"""
    return jsonify({
        'status': 'ready' if model_ready else 'error',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    主要功能: 預測上傳的圖片是什麼數字
    
    工作流程:
    1. 接收用戶上傳的圖片
    2. 準備圖片
    3. 預測
    4. 回傳結果
    """
    try:
        # 檢查模型是否準備好
        if not model_ready:
            return jsonify({'error': '模型未準備好'}), 500
        
        # 檢查用戶是否上傳了圖片
        if 'image' not in request.files:
            return jsonify({'error': '沒有上傳圖片'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': '沒有選擇文件'}), 400
        
        # 讀取圖片
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': f'無效的圖片格式: {str(e)}'}), 400
        
        # 準備圖片 (預處理)
        img_array = preprocess_image(image)
        if img_array is None:
            return jsonify({'error': '圖片預處理失敗'}), 400
        
        # 把圖片轉成網頁可顯示的格式
        display_img = image_to_display(img_array.flatten())
        
        # 預測
        prediction, confidence, probabilities = predict_digit(img_array)
        
        if prediction is None:
            return jsonify({'error': '預測失敗'}), 500
        
        # 回傳結果給用戶
        return jsonify({
            'success': True,
            'prediction': prediction,       # 預測的數字 (0~9)
            'confidence': confidence,       # 信心度 (0~1)
            'probabilities': probabilities, # 每個數字的概率
            'display_image': display_img   # 可以顯示的圖片
        })
    
    except Exception as e:
        print(f"預測端點出錯: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """
    同時預測多個圖片 (批量預測)
    
    想像: 一次給廚師 100 張菜譜,廚師挨個識別
    """
    try:
        if not model_ready:
            return jsonify({'error': '模型未準備好'}), 500
        
        # 檢查是否上傳了圖片
        if 'images' not in request.files:
            return jsonify({'error': '沒有上傳圖片'}), 400
        
        files = request.files.getlist('images')  # 取得所有上傳的文件
        
        if not files:
            return jsonify({'error': '沒有文件'}), 400
        
        results = []        # 用來存放所有結果
        successful = 0      # 計數成功的預測
        
        # 逐個處理每張圖片
        for file in files:
            try:
                image = Image.open(io.BytesIO(file.read()))
                img_array = preprocess_image(image)
                
                if img_array is None:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': '預處理失敗'
                    })
                    continue
                
                prediction, confidence, probabilities = predict_digit(img_array)
                
                if prediction is None:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': '預測失敗'
                    })
                    continue
                
                # 成功! 加入結果列表
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(files),          # 總共幾張
            'successful': successful,     # 成功幾張
            'results': results            # 每張的結果
        })
    
    except Exception as e:
        print(f"批量預測失敗: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['GET'])
def api_evaluate():
    """
    評估模型的準確率 (在測試集上)
    
    想像: 拿 10000 道已知答案的題目考模型
    """
    try:
        # 檢查是否有測試數據
        if not model_ready or X_test is None or y_test is None:
            return jsonify({'error': '沒有測試數據'}), 500
        
        # 用模型預測所有測試圖片
        y_pred = model.predict(X_test)
        
        # 計算各種評估指標
        accuracy = accuracy_score(y_test, y_pred)          # 正確率
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  # 精確度
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)        # 召回率
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)                # F1 分數
        
        return jsonify({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sample_count': len(y_test)
        })
    
    except Exception as e:
        print(f"評估失敗: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/confusion-matrix', methods=['GET'])
def api_confusion_matrix():
    """
    計算混淆矩陣 (顯示模型哪些地方容易出錯)
    
    想像: 看看模型把 3 誤認成 8 有幾次
    """
    try:
        if not model_ready or X_test is None or y_test is None:
            return jsonify({'error': '沒有測試數據'}), 500
        
        # 預測所有測試圖片
        y_pred = model.predict(X_test)
        
        # 計算混淆矩陣 (10x10 的表格)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
        
        return jsonify({
            'matrix': cm.tolist(),  # 轉成列表
            'shape': cm.shape       # 形狀 (10, 10)
        })
    
    except Exception as e:
        print(f"混淆矩陣計算失敗: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 處理錯誤的路由
@app.errorhandler(404)
def not_found(error):
    """如果用戶訪問不存在的頁面"""
    return jsonify({'error': '找不到該頁面'}), 404

@app.errorhandler(500)
def server_error(error):
    """如果伺服器出錯"""
    return jsonify({'error': '伺服器錯誤'}), 500


# ==================== 第6步: 啟動伺服器 ====================

if __name__ == '__main__':
    # 程式啟動時的準備工作
    print("\n" + "="*80)
    print("🔢 HOSVD 手寫數字識別 - Flask Web 應用")
    print("="*80)
    
    print("\n📦 正在載入模型...")
    load_model_files()  # 載入模型
    
    if model_ready:
        print("\n✓ 系統準備好了!")
        print("📱 啟動 Web 伺服器: http://localhost:5000")
        print("   (按 Ctrl+C 停止伺服器)")
        
        # 啟動伺服器
        # debug=True: 有錯誤時顯示詳細信息
        # host='0.0.0.0': 允許從任何電腦訪問
        # port=5000: 使用 5000 埠
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ 模型載入失敗,無法啟動伺服器")
        print("請檢查:")
        print("  1. results/models/hosvd_model_latest.pkl 是否存在")
        print("  2. 所有依賴包是否已安裝")
        sys.exit(1)
