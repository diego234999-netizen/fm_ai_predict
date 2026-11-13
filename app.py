import streamlit as st
import requests
import google.generativeai as genai
from PIL import Image
import json
from datetime import datetime

# ==========================================
# 1. 核心功能函式定義
# ==========================================

def get_xitun_forecast(cwa_api_key):
    """取得台中市西屯區未來12小時氣溫與降雨機率"""
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-073?Authorization={cwa_api_key}&LocationName=西屯區"
    try:
        response = requests.get(url)
        data = response.json()
        
        weather_elements = data['records']['locations'][0]['location'][0]['weatherElement']
        # 提取隔日的平均溫度 (T) 與 12小時降雨機率 (PoP12h)
        temp = next(item for item in weather_elements if item["elementName"] == "T")['time'][1]['elementValue'][0]['value']
        rain_prob = next(item for item in weather_elements if item["elementName"] == "PoP12h")['time'][1]['elementValue'][0]['value']
        
        return float(temp), float(rain_prob)
    except Exception as e:
        st.error(f"氣象資料獲取失敗: {e}")
        return None, None

def analyze_waste_receipt(image, gemini_api_key):
    """使用 Gemini 分析報廢明細並分類"""
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    這是一張超商的報廢明細照片。請自動辨識照片中的商品名稱與數量，並將其分類加總為以下五大類：
    1. 主食正餐
    2. 輕食點心
    3. 小吃滷味
    4. 湯品小菜
    5. 烘焙甜點
    
    請嚴格只回傳 JSON 格式，不要有任何其他文字說明，格式範例如下：
    {"主食正餐": 2, "輕食點心": 0, "小吃滷味": 1, "湯品小菜": 0, "烘焙甜點": 3}
    """
    response = model.generate_content([prompt, image])
    
    # 清理 Markdown 標記並轉為 Python Dictionary
    clean_json = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(clean_json)

# ==========================================
# 2. Streamlit 前端網頁介面
# ==========================================

st.set_page_config(page_title="全家 AI 預測系統 - 報廢登錄", layout="wide")
st.title("🏪 全家報廢明細自動登錄系統")
st.write(f"📍 目前店鋪位置：**台中市西屯區** | 📅 登錄日期：{datetime.now().strftime('%Y-%m-%d')}")

# 左側邊欄：API 設定
with st.sidebar:
    st.header("⚙️ 系統設定")
    cwa_key = st.text_input("中央氣象署 API Key", type="password")
    gem_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown("---")
    st.info("提示：為確保系統資安，實務上可將 Key 寫入 `.streamlit/secrets.toml` 檔案中。")

# 主畫面：照片上傳區
uploaded_file = st.file_uploader("📸 上傳本日報廢小白單或明細照片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 顯示上傳的照片
    col_img, col_data = st.columns([1, 2])
    
    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上傳報廢明細", use_column_width=True)
    
    with col_data:
        if st.button("🚀 開始智能分析 (讀取天氣與明細)", use_container_width=True):
            if not cwa_key or not gem_key:
                st.warning("請先於左側欄輸入 API Key！")
            else:
                with st.spinner("系統運算中...正在連結氣象局與 AI 視覺模型..."):
                    
                    # 步驟 A：抓取西屯區天氣
                    temp, rain_prob = get_xitun_forecast(cwa_key)
                    
                    # 步驟 B：辨識明細分類
                    try:
                        waste_data = analyze_waste_receipt(image, gem_key)
                        
                        # 呈現結果
                        st.success("✅ 分析完成！準備寫入資料庫")
                        
                        st.subheader("🌤️ 預估明日環境變數 (Feature X)")
                        weather_col1, weather_col2 = st.columns(2)
                        weather_col1.metric("明日平均溫度", f"{temp} °C")
                        weather_col2.metric("明日降雨機率", f"{rain_prob} %")
                        
                        st.subheader("🍙 報廢數量辨識結果 (Target Y)")
                        w_cols = st.columns(5)
                        w_cols[0].metric("主食正餐", waste_data.get("主食正餐", 0))
                        w_cols[1].metric("輕食點心", waste_data.get("輕食點心", 0))
                        w_cols[2].metric("小吃滷味", waste_data.get("小吃滷味", 0))
                        w_cols[3].metric("湯品小菜", waste_data.get("湯品小菜", 0))
                        w_cols[4].metric("烘焙甜點", waste_data.get("烘焙甜點", 0))
                        
                        # ==========================================
                        # 3. 銜接隨機森林預測模型與資料庫 (概念展示)
                        # ==========================================
                        st.markdown("---")
                        st.write("🔄 **後台系統動作確認：**")
                        st.code(f"""
# 1. 寫入歷史資料庫 (供日後模型 Retrain)
db.insert({{
    "date": "{datetime.now().strftime('%Y-%m-%d')}",
    "主食": {waste_data.get("主食正餐", 0)}, 
    "天氣": {temp}, 
    "降雨": {rain_prob}
}})

# 2. 呼叫隨機森林模型，以今日天氣變數推算明日最佳訂貨量
rf_model.predict([[is_weekend, {temp}, {rain_prob}]])
                        """, language="python")
                        
                    except Exception as e:
                        st.error(f"辨識過程發生錯誤，請確認照片清晰度。錯誤細節：{e}")
