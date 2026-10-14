import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

# ==========================================
# ⚙️ 系統與資料庫設定
# ==========================================
st.set_page_config(page_title="全家 AI 預測系統", page_icon="🏪", layout="wide")
import time # 最上面記得確認有沒有 import time，沒有的話順便補上

import time 

# ==========================================
# 🎮 全家美食開場動畫 (快速俐落版)
# ==========================================
if 'first_open' not in st.session_state:
    splash = st.empty()
    
    with splash.container():
        st.markdown("<h1 style='text-align: center; margin-top: 15vh; font-size: 50px;'>報廢剩食Ai預測</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: gray;'>逢甲運物二乙期末展示</h3>", unsafe_allow_html=True)
        st.divider()
        
        with st.spinner("啟動 AI 預測"):
            # 縮短等待時間，讓節奏更明快
            time.sleep(1.2) 
            
    splash.empty()
    
    # 🌟 快速俐落版：全家美食噴發特效
    food_emojis = ["🍞", "🍎", "🍱", "🍙", "🧁", "🍉", "🍰", "🍌", "🥤", "🥪", "🥗", "🍨"]
    emoji_style = ""
    # 將數量減少一點，避免畫面太雜亂，並讓它們幾乎同時噴發
    for idx, emoji in enumerate(food_emojis * 2): 
        left = (idx * 4) % 95 
        # 大幅縮短延遲時間，讓食物幾乎是瞬間一起衝出來
        delay = (idx * 0.03) 
        # 大幅加快飛行速度，只需 1.5 到 2.5 秒就會飛出畫面
        duration = 1.5 + (idx % 2) 
        size = 28 + (idx % 3) * 10 
        
        emoji_style += f"""
        @keyframes fly_{idx} {{
            0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
            10% {{ opacity: 0.9; }}
            90% {{ opacity: 0.9; }}
            100% {{ transform: translateY(-110vh) rotate({180 + idx * 30}deg); opacity: 0; }}
        }}
        .food_{idx} {{
            position: fixed;
            bottom: -50px;
            left: {left}vw;
            font-size: {size}px;
            /* 將動畫延遲跟持續時間調短 */
            animation: fly_{idx} {duration}s ease-in {delay}s forwards; 
            pointer-events: none;
            z-index: 99999;
        }}
        """
        
    html_content = f"<style>{emoji_style}</style>"
    for idx, emoji in enumerate(food_emojis * 2):
        html_content += f"<div class='food_{idx}'>{emoji}</div>"
        
    st.markdown(html_content, unsafe_allow_html=True)
    
    st.session_state['first_open'] = True
# 預設歷史資料 (如果雲端沒有 CSV 檔案，就自動建立)
default_history = [
    ['2026-09-02',0,0,0,5,3,0,2,4], ['2026-09-03',0,0,0,8,4,0,1,7], ['2026-09-04',0,0,0,2,10,1,1,7],
    ['2026-09-05',1,0,0,6,12,5,1,8], ['2026-09-07',0,0,0,2,5,0,0,11], ['2026-09-08',0,0,0,4,5,0,2,7],
    ['2026-09-14',0,0,0,3,0,0,0,3], ['2026-09-15',0,0,0,6,3,0,2,3], ['2026-09-16',0,0,0,2,8,0,4,5],
    ['2026-09-20',1,0,0,6,5,1,9,3], ['2026-09-25',0,1,1,7,15,1,3,5], ['2026-09-27',1,1,0,9,5,1,4,9],
    ['2026-09-28',0,1,0,5,4,1,5,5], ['2026-09-29',0,0,0,0,0,1,0,2], ['2026-09-30',0,0,0,7,2,0,1,2],
    ['2026-10-01',0,0,0,1,1,1,0,1], ['2026-10-02',0,0,1,2,8,2,0,6], ['2026-10-04',1,0,0,5,8,0,1,12],
    ['2026-10-05',0,0,0,3,2,0,3,5], ['2026-10-06',0,0,0,5,3,0,1,18], ['2026-10-07',0,0,0,2,4,0,2,5],
    ['2026-10-08',0,0,1,4,4,0,1,4], ['2026-10-10',1,1,0,7,6,3,2,8], ['2026-10-11',1,1,0,5,5,1,3,4]
]

default_tracking = [
    ['2026-10-02', 18, 18], ['2026-10-04', 26, 24], ['2026-10-05', 13, 11],
    ['2026-10-06', 27, 11], ['2026-10-07', 13, 12], ['2026-10-08', 13, 17],
    ['2026-10-10', 26, 26], ['2026-10-11', 18, 24]
]

# 讀取共用資料 (Streamlit 雲端共享資料夾)
def load_data():
    if not os.path.exists('history.csv'):
        pd.DataFrame(default_history, columns=['Date', 'Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'MainMeal', 'LightMeal', 'HotDeli', 'SoupFruit', 'Dessert']).to_csv('history.csv', index=False)
    if not os.path.exists('tracking.csv'):
        pd.DataFrame(default_tracking, columns=['Date', 'Actual_Waste', 'Recorded_AI_Prediction']).to_csv('tracking.csv', index=False)
    
    return pd.read_csv('history.csv'), pd.read_csv('tracking.csv')

df_history, df_tracking = load_data()

# ==========================================
# 🧠 訓練真正的隨機森林 AI 模型
# ==========================================
X = df_history[['Is_Weekend', 'Is_Holiday', 'Is_PreHoliday']]
targets = ['MainMeal', 'LightMeal', 'HotDeli', 'SoupFruit', 'Dessert']
target_names_zh = {'MainMeal': '主食正餐', 'LightMeal': '輕食點心', 'HotDeli': '小吃滷味', 'SoupFruit': '湯品水果', 'Dessert': '烘焙甜點'}
models = {}

for target in targets:
    y = df_history[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[target] = model

# ==========================================
# 🎨 網頁 UI 設計 (主畫面)
# ==========================================
st.title("Ai報廢剩食預測")
st.markdown("**運物二乙 生產與配銷 第八組期末**")
st.divider()

col1, col2 = st.columns([1, 1])

# --- 左半邊：AI 預測引擎 ---
with col1:
    st.header("明日報廢量預測")
    st.info("請輸入明日特徵進行預測")
    
    with st.form("predict_form"):
        p_wk = st.selectbox("是否為週末?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        p_hol = st.selectbox("是否為連假?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        p_pre = st.selectbox("是否為連假前夕?", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        submit_predict = st.form_submit_button("執行預測")
        
    if submit_predict:
        future_X = pd.DataFrame([[p_wk, p_hol, p_pre]], columns=['Is_Weekend', 'Is_Holiday', 'Is_PreHoliday'])
        pred_results = {t: int(models[t].predict(future_X)[0]) for t in targets}
        total_pred = sum(pred_results.values())
        
        st.success(f"系統預估明日總報廢量： **{total_pred} 件**")
        cols = st.columns(5)
        for idx, t in enumerate(targets):
            cols[idx].metric(label=target_names_zh[t], value=pred_results[t])

# --- 右半邊：真實 MAE 戰績 ---
with col2:
    st.header("預測誤差MAE")
    if not df_tracking.empty:
        true_mae = mean_absolute_error(df_tracking['Actual_Waste'], df_tracking['Recorded_AI_Prediction'])
        st.metric(label="目前系統實戰平均誤差 (MAE)", value=f"{true_mae:.2f} 件")
    
    st.dataframe(df_tracking, use_container_width=True, hide_index=True)

st.divider()

# ==========================================
# 📥 資料寫入區 (側邊欄)
# ==========================================
st.sidebar.header("資料更新中心")
st.sidebar.markdown("各店長請於打烊後，在此同步最新數據。")

# 1. 新增歷史資料
with st.sidebar.expander("新增今日分類報廢"):
    with st.form("history_form"):
        h_date = st.date_input("日期")
        h_wk = st.selectbox("週末", [0,1], key='h1', format_func=lambda x: "是" if x == 1 else "否")
        h_hol = st.selectbox("連假", [0,1], key='h2', format_func=lambda x: "是" if x == 1 else "否")
        h_pre = st.selectbox("連假前夕", [0,1], key='h3', format_func=lambda x: "是" if x == 1 else "否")
        h_main = st.number_input("主食", min_value=0, value=0)
        h_light = st.number_input("輕食", min_value=0, value=0)
        h_hot = st.number_input("小吃", min_value=0, value=0)
        h_soup = st.number_input("湯品水果", min_value=0, value=0)
        h_dessert = st.number_input("甜點麵包", min_value=0, value=0)
        
        if st.form_submit_button("寫入雲端資料庫"):
            new_row = [str(h_date), h_wk, h_hol, h_pre, h_main, h_light, h_hot, h_soup, h_dessert]
            df_history.loc[len(df_history)] = new_row
            df_history.to_csv('history.csv', index=False)
            st.success("✅ 歷史資料已同步！模型已自動重新訓練。")
            st.rerun()
            
    if st.button("刪除最後一筆歷史資料"):
        if not df_history.empty:
            df_history = df_history.iloc[:-1]
            df_history.to_csv('history.csv', index=False)
            st.success("已成功刪除最後一筆歷史資料！")
            st.rerun()

# 2. 新增 MAE 戰績
with st.sidebar.expander("新增 MAE "):
    with st.form("tracking_form"):
        t_date = st.date_input("戰績日期")
        t_actual = st.number_input("實際報廢總數", min_value=0, value=0)
        t_predict = st.number_input("AI 事前預測總數", min_value=0, value=0)
        
        if st.form_submit_button("寫入戰績"):
            df_tracking.loc[len(df_tracking)] = [str(t_date), t_actual, t_predict]
            df_tracking.to_csv('tracking.csv', index=False)
            st.success("✅ 戰績已同步更新！")
            st.rerun()
            
    if st.button("刪除最後一筆戰績資料"):
        if not df_tracking.empty:
            df_tracking = df_tracking.iloc[:-1]
            df_tracking.to_csv('tracking.csv', index=False)
            st.success("已成功刪除最後一筆戰績資料！")
            st.rerun()

# ==========================================
# 📂 老闆專用後台：資料庫查看與備份
# ==========================================
st.divider()
st.header("📂 資料庫管理與備份 (店長專用)")
with st.expander("點我展開：查看並下載完整的雲端歷史資料庫"):
    st.markdown("⚠️ **注意：** 免費伺服器休眠時可能會清除暫存資料，請記得定期點擊下方按鈕將資料下載備份到電腦中。")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("歷史訓練庫 (history.csv)")
        st.dataframe(df_history, use_container_width=True)
        st.download_button(
            label="下載歷史資料",
            data=df_history.to_csv(index=False).encode('utf-8-sig'),
            file_name='FamilyMart_History.csv',
            mime='text/csv'
        )
        
    with col_b:
        st.subheader("MAE 戰績庫 (tracking.csv)")
        st.dataframe(df_tracking, use_container_width=True)
        st.download_button(
            label="下載戰績資料",
            data=df_tracking.to_csv(index=False).encode('utf-8-sig'),
            file_name='FamilyMart_Tracking.csv',
            mime='text/csv'
        )
