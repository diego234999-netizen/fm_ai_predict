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
    page_title="全家便利商店 | 智能配銷與報廢預測系統",
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
            "<div style='text-align: center; margin-top: 18vh;'>"
            "<h1 style='color: #0080C6; font-size: 42px; font-weight: 700; letter-spacing: 2px;'>全家便利商店 智能配銷中樞</h1>"
            "<p style='color: #009B4C; font-size: 18px; font-weight: 600;'>逢甲大學運輸與物流學系 | 生產與配銷期末專題 (SDGs 12)</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.divider()
        with st.spinner("系統正在初始化 AI 預測引擎與雲端資料庫，請稍候..."):
            time.sleep(1.5)
    splash.empty()
    st.session_state['first_open'] = True

# ==========================================
# 資料庫初始化與自我修復機制 (更新至 10/15 完整資料)
# ==========================================
HISTORY_FILE = 'history.csv'
TRACKING_FILE = 'tracking.csv'

expected_history_cols = ['Date', 'Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'MainMeal', 'LightMeal', 'HotDeli', 'SoupFruit', 'Dessert']
expected_tracking_cols = ['Date', 'Actual_Waste', 'Recorded_AI_Prediction']

# 完整 27 筆歷史分類報廢資料 (已按日期排序整理)
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
    ['2026-10-15', 0, 0, 0, 1, 6, 1, 1, 8]
]

# 完整 11 筆 MAE 追蹤資料 (已按日期排序整理)
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
    ['2026-10-15', 17, 12]
]

def load_data():
    # 檢查並修復 history.csv
    need_reset_hist = True
    if os.path.exists(HISTORY_FILE):
        try:
            temp_df = pd.read_csv(HISTORY_FILE)
            if 'Unnamed: 0' in temp_df.columns:
                temp_df = temp_df.drop(columns=['Unnamed: 0'])
            # 確保所有必要欄位都存在，且資料筆數不小於預設的 27 筆
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
    <div class='fm-title'>全家便利商店 | 智能配銷與報廢預測系統</div>
    <div class='fm-subtitle'>逢甲大學運輸與物流學系 二乙 | 生產與配銷 第八組期末專案 (SDGs 12 負責任的消費與生產)</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# --- 左半邊：AI 預測引擎 ---
with col1:
    st.markdown("<div class='section-title'>明日門市報廢量預測模組</div>", unsafe_allow_html=True)
    st.caption("請設定明日之日曆與營運特徵，系統將透過隨機森林模型估算各品項預期報廢量，作為訂貨減量之決策依據。")
    
    with st.form("predict_form"):
        p_wk = st.selectbox("是否為週末 (週六、週日)", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        p_hol = st.selectbox("是否為國定連假期間", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        p_pre = st.selectbox("是否為連假前夕", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
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
        df_tracking['Abs_Error'] = (df_tracking['Actual_Waste'] - df_tracking['Recorded_AI_Prediction']).abs()
        current_mae = df_tracking['Abs_Error'].mean()
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="目前累積平均絕對誤差 (MAE)", value=f"{current_mae:.2f} 件")
        m_col2.metric(label="累計實測驗證天數", value=f"{len(df_tracking)} 天")
        
        chart_data = df_tracking.set_index('Date')[['Actual_Waste', 'Recorded_AI_Prediction']].rename(
            columns={'Actual_Waste': '門市實際報廢量', 'Recorded_AI_Prediction': 'AI 系統預測量'}
        )
        # 採用全家企業藍與永續綠繪製折線圖
        st.line_chart(chart_data, color=["#0080C6", "#009B4C"])
    else:
        st.warning("目前尚無驗證資料。")

st.divider()

# ==========================================
# 側邊欄：門市數據登錄中心
# ==========================================
st.sidebar.markdown("### 門市數據登錄中心")
st.sidebar.caption("請於每日晚班打烊結算後，於此處登錄當日實際報廢數據以利模型持續重新訓練。")

# 1. 新增歷史分類報廢
with st.sidebar.expander("新增當日分類報廢紀錄"):
    with st.form("history_form"):
        h_date = st.date_input("結算日期")
        h_wk = st.selectbox("是否為週末", [0, 1], key='h1', format_func=lambda x: "是" if x == 1 else "否")
        h_hol = st.selectbox("是否為連假", [0, 1], key='h2', format_func=lambda x: "是" if x == 1 else "否")
        h_pre = st.selectbox("是否為連假前夕", [0, 1], key='h3', format_func=lambda x: "是" if x == 1 else "否")
        h_main = st.number_input("主食便當類 (件)", min_value=0, value=0)
        h_light = st.number_input("輕食飯糰三明治 (件)", min_value=0, value=0)
        h_hot = st.number_input("小吃熟食類 (件)", min_value=0, value=0)
        h_soup = st.number_input("湯品與截切水果 (件)", min_value=0, value=0)
        h_dessert = st.number_input("甜點與烘焙麵包 (件)", min_value=0, value=0)
        
        if st.form_submit_button("寫入資料庫並重新訓練模型", use_container_width=True):
            new_row = [str(h_date), h_wk, h_hol, h_pre, h_main, h_light, h_hot, h_soup, h_dessert]
            df_history.loc[len(df_history)] = new_row
            df_history.to_csv(HISTORY_FILE, index=False)
            st.success("歷史分類資料已同步更新。")
            time.sleep(0.5)
            st.rerun()
            
    if st.button("刪除最後一筆歷史紀錄", use_container_width=True):
        if len(df_history) > 1:
            df_history = df_history.iloc[:-1]
            df_history.to_csv(HISTORY_FILE, index=False)
            st.success("已刪除最後一筆歷史紀錄。")
            time.sleep(0.5)
            st.rerun()

# 2. 新增 MAE 對決戰績
with st.sidebar.expander("新增每日預測誤差驗證紀錄"):
    with st.form("tracking_form"):
        t_date = st.date_input("驗證日期")
        t_actual = st.number_input("當日實際總報廢量 (件)", min_value=0, value=0)
        t_pred = st.number_input("系統預估總報廢量 (件)", min_value=0, value=0)
        
        if st.form_submit_button("寫入驗證紀錄", use_container_width=True):
            new_track = [str(t_date), t_actual, t_pred]
            df_tracking = df_tracking[expected_tracking_cols]
            df_tracking.loc[len(df_tracking)] = new_track
            df_tracking.to_csv(TRACKING_FILE, index=False)
            st.success("驗證紀錄已同步更新。")
            time.sleep(0.5)
            st.rerun()

    if st.button("刪除最後一筆驗證紀錄", use_container_width=True):
        if len(df_tracking) > 1:
            df_tracking = df_tracking[expected_tracking_cols].iloc[:-1]
            df_tracking.to_csv(TRACKING_FILE, index=False)
            st.success("已刪除最後一筆驗證紀錄。")
            time.sleep(0.5)
            st.rerun()

# ==========================================
# 底部區塊：資料庫檢視與備份匯出
# ==========================================
st.markdown("<div class='section-title'>雲端資料庫明細與備份管理</div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["歷史分類報廢資料庫 (模型訓練集)", "預測誤差驗證資料庫 (MAE 追蹤集)"])

with tab1:
    display_hist = df_history.rename(columns={
        'Date': '日期', 'Is_Weekend': '週末', 'Is_Holiday': '連假', 'Is_PreHoliday': '連假前夕',
        'MainMeal': '主食便當類', 'LightMeal': '輕食飯糰三明治', 'HotDeli': '小吃熟食類',
        'SoupFruit': '湯品與截切水果', 'Dessert': '甜點與烘焙麵包'
    })
    st.dataframe(display_hist, use_container_width=True)
    csv_hist = df_history.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="下載歷史分類報廢資料 (CSV)",
        data=csv_hist,
        file_name="familymart_history_export.csv",
        mime="text/csv"
    )

with tab2:
    display_track = df_tracking[expected_tracking_cols].copy()
    display_track['絕對誤差 (件)'] = (display_track['Actual_Waste'] - display_track['Recorded_AI_Prediction']).abs()
    display_track = display_track.rename(columns={
        'Date': '日期', 'Actual_Waste': '實際總報廢量', 'Recorded_AI_Prediction': 'AI 預估總報廢量'
    })
    st.dataframe(display_track, use_container_width=True)
    csv_track = df_tracking[expected_tracking_cols].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="下載預測誤差驗證資料 (CSV)",
        data=csv_track,
        file_name="familymart_mae_export.csv",
        mime="text/csv"
    )
