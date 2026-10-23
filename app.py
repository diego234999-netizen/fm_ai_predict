import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import time

# ==========================================
# 系統頁面設定 (企業專業風格)
# ==========================================
st.set_page_config(
    page_title="全家便利商店 報廢預測系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全家企業標準色 CSS 視覺設計 (藍 #0080C6 / 綠 #009B4C)
st.markdown("""
<style>
    .fm-top-bar {
        height: 10px;
        background: linear-gradient(90deg, #0080C6 0%, #0080C6 50%, #009B4C 50%, #009B4C 100%);
        margin-bottom: 20px;
        border-radius: 2px;
    }
    .fm-header-box {
        background-color: #ffffff;
        padding: 18px 24px;
        border-left: 6px solid #0080C6;
        border-bottom: 2px solid #009B4C;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 24px;
    }
    .fm-title {
        color: #005A8B;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .fm-subtitle {
        color: #555555;
        font-size: 14px;
        font-weight: 500;
        margin-top: 6px;
    }
    .section-title {
        color: #0080C6;
        font-size: 20px;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 開場載入畫面 (無表情符號、俐落企業風)
# ==========================================
if 'first_open' not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.markdown("<div class='fm-top-bar'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center; margin-top: 22vh;'>"
            "<h1 style='color: #0080C6; font-size: 32px; font-weight: 700;'>Ai報廢預測系統</h1>"
            "<p style='color: #009B4C; font-size: 16px; font-weight: 600;'>逢甲大學運物二乙第八組 生產與配銷期末</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()
        with st.spinner("系統正在初始化 AI 預測引擎與雲端資料庫，請稍候..."):
            time.sleep(1.5)
    splash.empty()
    st.session_state['first_open'] = True

# ==========================================
# 資料庫初始化與自我修復機制 (更新至 10/22 完整資料)
# ==========================================
HISTORY_FILE = 'history.csv'
TRACKING_FILE = 'tracking.csv'

expected_history_cols = ['Date', 'Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'MainMeal', 'LightMeal', 'HotDeli', 'SoupFruit', 'Dessert']
expected_tracking_cols = ['Date', 'Actual_Waste', 'Recorded_AI_Prediction']

# 完整 33 筆歷史分類報廢資料 (更新至 10/22)
default_history_data = [
    ['2026-09-02', 0, 0, 0, 5, 3, 0, 2, 4],
    ['2026-09-03', 0, 0, 0, 8, 4, 0, 1, 7],
    ['2026-09-04', 0, 0, 0, 2, 10, 1, 1, 7],
    ['2026-09-05', 1, 0, 0, 6, 12, 5, 1, 8],
    ['2026-09-07', 0, 0, 0, 2, 5, 0, 0, 11],
    ['2026-09-08', 0, 0, 0, 4, 5, 0, 2, 7],
    ['2026-09-14', 0, 0, 0, 3, 0, 0, 0, 3],
    ['2026-09-15', 0, 0, 0, 6, 3, 0, 2, 3],
    ['2026-09-16', 0, 0, 0, 2, 8, 0, 4, 5],
    ['2026-09-20', 1, 0, 0, 6, 5, 1, 9, 3],
    ['2026-09-25', 0, 1, 1, 7, 15, 1, 3, 5],
    ['2026-09-27', 1, 1, 0, 9, 5, 1, 4, 9],
    ['2026-09-28', 0, 1, 0, 5, 4, 1, 5, 5],
    ['2026-09-29', 0, 0, 0, 0, 0, 1, 0, 2],
    ['2026-09-30', 0, 0, 0, 7, 2, 0, 1, 2],
    ['2026-10-01', 0, 0, 0, 1, 1, 1, 0, 1],
    ['2026-10-02', 0, 0, 1, 2, 8, 2, 0, 6],
    ['2026-10-04', 1, 0, 0, 5, 8, 0, 1, 12],
    ['2026-10-05', 0, 0, 0, 3, 2, 0, 3, 5],
    ['2026-10-06', 0, 0, 0, 5, 3, 0, 1, 18],
    ['2026-10-07', 0, 0, 0, 2, 4, 0, 2, 5],
    ['2026-10-08', 0, 0, 1, 4, 4, 0, 1, 4],
    ['2026-10-10', 1, 1, 0, 7, 6, 3, 2, 8],
    ['2026-10-11', 1, 1, 0, 5, 5, 1, 3, 4],
    ['2026-10-12', 0, 0, 0, 1, 1, 0, 0, 4],
    ['2026-10-14', 0, 0, 0, 3, 0, 1, 0, 6],
    ['2026-10-15', 0, 0, 0, 1, 6, 1, 1, 8],
    ['2026-10-16', 0, 0, 1, 2, 17, 0, 3, 5], # 新增資料
    ['2026-10-17', 1, 0, 0, 4, 6, 0, 2, 7],  # 新增資料
    ['2026-10-18', 1, 0, 0, 3, 1, 0, 0, 6],  # 新增資料
    ['2026-10-19', 0, 0, 0, 1, 3, 0, 1, 3],  # 新增資料
    ['2026-10-20', 0, 0, 0, 2, 7, 0, 1, 4],  # 新增資料
    ['2026-10-22', 0, 0, 0, 2, 7, 0, 2, 2]   # 新增資料
]

# 完整 17 筆 MAE 追蹤資料 (更新至 10/22)
default_tracking_data = [
    ['2026-10-02', 18, 18],
    ['2026-10-04', 26, 24],
    ['2026-10-05', 13, 11],
    ['2026-10-06', 27, 11],
    ['2026-10-07', 13, 12],
    ['2026-10-08', 13, 17],
    ['2026-10-10', 26, 26],
    ['2026-10-11', 18, 24],
    ['2026-10-12', 6, 12],
    ['2026-10-14', 10, 12],
    ['2026-10-15', 17, 12],
    ['2026-10-16', 27, 17], # 新增資料
    ['2026-10-17', 28, 19], # 新增資料
    ['2026-10-18', 10, 24], # 新增資料
    ['2026-10-19', 8, 13],  # 新增資料
    ['2026-10-20', 14, 13], # 新增資料
    ['2026-10-22', 13, 14]  # 新增資料
]

def load_data():
    # 檢查並修復 history.csv
    need_reset_hist = True
    if os.path.exists(HISTORY_FILE):
        try:
            temp_df = pd.read_csv(HISTORY_FILE)
            if 'Unnamed: 0' in temp_df.columns:
                temp_df = temp_df.drop(columns=['Unnamed: 0'])
            if all(col in temp_df.columns for col in expected_history_cols) and len(temp_df) >= len(default_history_data):
                df_hist = temp_df[expected_history_cols]
                need_reset_hist = False
        except Exception:
            need_reset_hist = True
            
    if need_reset_hist:
        df_hist = pd.DataFrame(default_history_data, columns=expected_history_cols)
        df_hist.to_csv(HISTORY_FILE, index=False)

    # 檢查並修復 tracking.csv
    need_reset_track = True
    if os.path.exists(TRACKING_FILE):
        try:
            temp_track = pd.read_csv(TRACKING_FILE)
            if 'Unnamed: 0' in temp_track.columns:
                temp_track = temp_track.drop(columns=['Unnamed: 0'])
            if all(col in temp_track.columns for col in expected_tracking_cols) and len(temp_track) >= len(default_tracking_data):
                df_track = temp_track[expected_tracking_cols]
                need_reset_track = False
        except Exception:
            need_reset_track = True

    if need_reset_track:
        df_track = pd.DataFrame(default_tracking_data, columns=expected_tracking_cols)
        df_track.to_csv(TRACKING_FILE, index=False)

    return df_hist, df_track

df_history, df_tracking = load_data()

# ==========================================
# AI 隨機森林預測引擎訓練
# ==========================================
features = ['Is_Weekend', 'Is_Holiday', 'Is_PreHoliday']
targets = ['MainMeal', 'LightMeal', 'HotDeli', 'SoupFruit', 'Dessert']
target_names_zh = {
    'MainMeal': '主食便當類',
    'LightMeal': '輕食飯糰三明治',
    'HotDeli': '小吃熟食類',
    'SoupFruit': '湯品與截切水果',
    'Dessert': '甜點與烘焙麵包'
}

models = {}
X_train = df_history[features]
for target in targets:
    y_train = df_history[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model

# ==========================================
# 網頁主畫面設計 (企業正式風格)
# ==========================================
st.markdown("<div class='fm-top-bar'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='fm-header-box'>
    <div class='fm-title'>
        生產與配銷期末<br>
        運物二乙第八組<br>
        Ai報廢預測系統
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# --- 左半邊：AI 預測引擎 ---
with col1:
    st.markdown("<div class='section-title'>明日門市報廢量預測</div>", unsafe_allow_html=True)
    st.caption("請設定明日之日曆與營運特徵，系統將透過模型估算各品項預期報廢量。")
    
    with st.form("predict_form"):
        p_wk = st.selectbox("是否為週末 (週六、週日)", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        p_hol = st.selectbox("是否為國定連假期間", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        p_pre = st.selectbox("是否為連假前夕(含星期五)", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        submit_predict = st.form_submit_button("執行智能預測運算", use_container_width=True)
        
    if submit_predict:
        future_X = pd.DataFrame([[p_wk, p_hol, p_pre]], columns=features)
        pred_results = {t: int(round(models[t].predict(future_X)[0])) for t in targets}
        total_pred = sum(pred_results.values())
        
        st.info(f"系統預估明日總報廢量：**{total_pred} 件**")
        cols = st.columns(5)
        for idx, t in enumerate(targets):
            cols[idx].metric(label=target_names_zh[t], value=f"{pred_results[t]} 件")

# --- 右半邊：真實 MAE 驗證指標 ---
with col2:
    st.markdown("<div class='section-title'>模型預測準確度驗證 (MAE 追蹤)</div>", unsafe_allow_html=True)
    
    if not df_tracking.empty:
        # 計算絕對誤差與當前平均絕對誤差 (MAE)
        df_tracking['Abs_Error'] = (df_tracking['Actual_Waste'] - df_tracking['Recorded_AI_Prediction']).abs()
        current_mae = df_tracking['Abs_Error'].mean()
        
        # 建立大型指標展示區
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="目前平均絕對誤差 (MAE)", value=f"{current_mae:.2f} 件")
        m_col2.metric(label="已追蹤驗證天數", value=f"{len(df_tracking)} 天")
        
        # 繪製趨勢對比圖
        st.markdown("<div style='margin-top: 10px; font-weight: 500; font-size: 14px; color: #555;'>真實總報廢量 vs AI 模型預測值 趨勢對比</div>", unsafe_allow_html=True)
        chart_data = df_tracking.set_index('Date')[['Actual_Waste', 'Recorded_AI_Prediction']]
        chart_data.columns = ['真實總報廢量', 'AI 預測總量']
        st.line_chart(chart_data, height=220)
        
        # 呈現追蹤歷史數據表
        st.markdown("<div style='margin-top: 15px; font-weight: 500; font-size: 14px; color: #555;'>歷史驗證數據明細</div>", unsafe_allow_html=True)
        st.dataframe(
            df_tracking[['Date', 'Actual_Waste', 'Recorded_AI_Prediction', 'Abs_Error']].rename(
                columns={
                    'Date': '日期',
                    'Actual_Waste': '實際總報廢量 (件)',
                    'Recorded_AI_Prediction': 'AI 預測總量 (件)',
                    'Abs_Error': '絕對誤差 (件)'
                }
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("目前暫無追蹤驗證數據。")
