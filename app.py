import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import time

# ==========================================
# 企業級頁面設定與全家配色 CSS 視覺設計
# ==========================================
st.set_page_config(page_title="全家便利商店｜智能配銷與報廢預測系統", layout="wide")

# 注入全家便利商店品牌色系 (FamilyMart Blue #0080C6 / Green #009B4C)
st.markdown("""
<style>
    /* 頂部全家招牌雙色橫條 */
    .fm-top-bar {
        background: linear-gradient(90deg, #0080C6 0%, #0080C6 50%, #009B4C 50%, #009B4C 100%);
        height: 8px;
        width: 100%;
        border-radius: 4px 4px 0 0;
        margin-bottom: 0px;
    }
    /* 主標題區塊設計 */
    .fm-header-box {
        background-color: rgba(0, 128, 198, 0.06);
        border: 1px solid rgba(0, 128, 198, 0.2);
        border-left: 6px solid #009B4C;
        padding: 20px 24px;
        border-radius: 0 0 8px 8px;
        margin-bottom: 24px;
    }
    .fm-title {
        color: #0080C6;
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: 1px;
    }
    .fm-subtitle {
        color: #6c757d;
        font-size: 15px;
        font-weight: 500;
        margin: 0;
    }
    /* 區塊小標題 */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #009B4C;
        border-bottom: 2px solid rgba(0, 155, 76, 0.3);
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    /* 數據指標卡優化 */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 128, 198, 0.25);
        border-top: 3px solid #0080C6;
        padding: 12px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 系統啟動開場畫面 (專業正式版，僅於初次開啟時執行)
# ==========================================
if 'first_open' not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.markdown("<div style='margin-top: 20vh;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='fm-top-bar' style='max-width: 600px; margin: 0 auto;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px; border: 1px solid rgba(0,128,198,0.3); max-width: 600px; margin: 0 auto; border-radius: 0 0 8px 8px; background-color: rgba(0,128,198,0.03);'>
            <h1 style='color: #0080C6; font-size: 36px; font-weight: 700; margin-bottom: 10px; letter-spacing: 1.5px;'>全家智能配銷中樞</h1>
            <h4 style='color: #009B4C; font-weight: 500; margin-bottom: 24px;'>逢甲大學運輸與物流學系｜SDGs 12 產學專題系統</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            with st.spinner("正在載入歷史報廢資料庫與隨機森林預測模型..."):
                time.sleep(1.8)
                
    splash.empty()
    st.session_state['first_open'] = True

# ==========================================
# 雲端資料庫初始化 (已更新至 2026-10-15 最新實戰數據)
# ==========================================
default_history = [
    ["2026-09-28", 0, 1, 0, 2, 5, 4, 2, 10],
    ["2026-09-29", 0, 0, 0, 0, 2, 2, 3, 9],
    ["2026-10-02", 0, 0, 0, 3, 4, 0, 6, 5],
    ["2026-10-04", 1, 0, 0, 5, 4, 3, 2, 12],
    ["2026-10-05", 0, 0, 0, 3, 2, 2, 1, 5],
    ["2026-10-06", 0, 0, 0, 3, 8, 3, 6, 7],
    ["2026-10-07", 0, 0, 0, 4, 1, 0, 4, 4],
    ["2026-10-08", 0, 0, 1, 3, 3, 1, 3, 3],
    ["2026-10-10", 1, 1, 0, 9, 4, 3, 2, 8],
    ["2026-10-11", 1, 1, 0, 3, 3, 4, 2, 6],
    ["2026-10-12", 0, 0, 0, 1, 1, 1, 3, 0],
    ["2026-10-13", 0, 0, 0, 4, 0, 2, 5, 2],
    ["2026-10-14", 0, 0, 0, 1, 2, 1, 4, 2],
    ["2026-10-15", 0, 0, 0, 5, 4, 1, 3, 2]
]

default_tracking = [
    ["2026-10-02", 18, 22],
    ["2026-10-04", 26, 20],
    ["2026-10-05", 13, 20],
    ["2026-10-06", 27, 17],
    ["2026-10-07", 13, 21],
    ["2026-10-08", 13, 19],
    ["2026-10-10", 26, 22],
    ["2026-10-11", 18, 26],
    ["2026-10-12", 6, 17],
    ["2026-10-13", 13, 14],
    ["2026-10-14", 10, 14],
    ["2026-10-15", 15, 13]
]

history_cols = ['Date', 'Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'Main_Dish', 'Light_Meal', 'Hot_Snack', 'Soup_Fruit', 'Dessert_Bread']
tracking_cols = ['Date', 'Actual_Waste', 'Recorded_AI_Pred']

if not os.path.exists('history.csv'):
    pd.DataFrame(default_history, columns=history_cols).to_csv('history.csv', index=False)
if not os.path.exists('tracking.csv'):
    pd.DataFrame(default_tracking, columns=tracking_cols).to_csv('tracking.csv', index=False)

df_history = pd.read_csv('history.csv')
df_tracking = pd.read_csv('tracking.csv')

# 自動檢查：若暫存檔筆數少於內建最新筆數，自動升級為最新 10/15 資料庫
if len(df_history) < len(default_history):
    df_history = pd.DataFrame(default_history, columns=history_cols)
    df_history.to_csv('history.csv', index=False)

if len(df_tracking) < len(default_tracking):
    df_tracking = pd.DataFrame(default_tracking, columns=tracking_cols)
    df_tracking.to_csv('tracking.csv', index=False)

# ==========================================
# 機器學習模型訓練 (Random Forest)
# ==========================================
features = ['Is_Weekend', 'Is_Holiday', 'Is_PreHoliday']
targets = ['Main_Dish', 'Light_Meal', 'Hot_Snack', 'Soup_Fruit', 'Dessert_Bread']
target_names_zh = {
    'Main_Dish': '主食便當類',
    'Light_Meal': '輕食飯糰手卷',
    'Hot_Snack': '熟食小吃類',
    'Soup_Fruit': '湯品與截切水果',
    'Dessert_Bread': '甜點與烘焙麵包'
}

models = {}
X_train = df_history[features]
for target in targets:
    y_train = df_history[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model

# ==========================================
# 網頁主畫面設計 (全家企業專業風格)
# ==========================================
st.markdown("<div class='fm-top-bar'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="fm-header-box">
    <div class="fm-title">全家便利商店｜智能配銷與鮮食報廢預測系統</div>
    <div class="fm-subtitle">逢甲大學運輸與物流學系｜生產與配銷管理 第八組期末專題 (SDGs 12 負責任的消費與生產)</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# --- 左半邊：AI 預測引擎 ---
with col1:
    st.markdown("<div class='section-title'>明日鮮食報廢量預測引擎</div>", unsafe_allow_html=True)
    st.caption("請設定明日之營運日曆特徵，系統將透過隨機森林模型推估各品類可能之報廢數量，作為訂貨減量之決策依據。")
    
    with st.form("predict_form"):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            p_wk = st.selectbox("是否為週末", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        with c_p2:
            p_hol = st.selectbox("是否為連假期間", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
        with c_p3:
            p_pre = st.selectbox("是否為連假前夕", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
            
        submit_predict = st.form_submit_button("執行智能預測運算", use_container_width=True)
        
    if submit_predict:
        future_X = pd.DataFrame([[p_wk, p_hol, p_pre]], columns=features)
        pred_results = {t: int(round(models[t].predict(future_X)[0])) for t in targets}
        total_pred = sum(pred_results.values())
        
        st.markdown(f"""
        <div style='background-color: rgba(0, 155, 76, 0.1); border-left: 4px solid #009B4C; padding: 12px 16px; border-radius: 4px; margin: 16px 0;'>
            <span style='font-size: 16px; font-weight: 600; color: #009B4C;'>系統預估明日總報廢量：</span>
            <span style='font-size: 22px; font-weight: 700;'>{total_pred} 件</span>
        </div>
        """, unsafe_allow_html=True)
        
        m_cols = st.columns(3)
        for idx, t in enumerate(targets):
            with m_cols[idx % 3]:
                st.metric(label=target_names_zh[t], value=f"{pred_results[t]} 件")

# --- 右半邊：真實 MAE 戰績 ---
with col2:
    st.markdown("<div class='section-title'>模型預測準確度驗證 (MAE 追蹤)</div>", unsafe_allow_html=True)
    
    if not df_tracking.empty:
        df_eval = df_tracking.copy()
        # 計算歷史移動平均作為傳統基準 (Baseline)
        df_eval['Baseline_Pred'] = df_eval['Actual_Waste'].shift(1).expanding().mean().fillna(df_eval['Actual_Waste'].iloc[0]).round()
        
        mae_ai = np.mean(np.abs(df_eval['Actual_Waste'] - df_eval['Recorded_AI_Pred']))
        mae_base = np.mean(np.abs(df_eval['Actual_Waste'] - df_eval['Baseline_Pred']))
        improvement = ((mae_base - mae_ai) / mae_base) * 100 if mae_base != 0 else 0
        
        k_col1, k_col2, k_col3 = st.columns(3)
        k_col1.metric("AI 模型平均絕對誤差 (MAE)", f"{mae_ai:.2f} 件")
        k_col2.metric("傳統經驗平均誤差", f"{mae_base:.2f} 件")
        k_col3.metric("預測準確度提升幅度", f"{improvement:.1f}%")
        
        st.markdown("<div style='margin-top: 16px; font-weight: 600; font-size: 15px;'>實際報廢量 vs. AI 預測值趨勢對比圖</div>", unsafe_allow_html=True)
        chart_data = df_eval.set_index('Date')[['Actual_Waste', 'Recorded_AI_Pred']].rename(
            columns={'Actual_Waste': '實際總報廢量', 'Recorded_AI_Pred': 'AI 預估報廢量'}
        )
        st.line_chart(chart_data, color=["#0080C6", "#009B4C"])
    else:
        st.info("目前尚無驗證戰績資料。")

st.divider()

# ==========================================
# 資料庫檢視與備份管理區
# ==========================================
st.markdown("<div class='section-title'>門市資料庫管理與本地端備份</div>", unsafe_allow_html=True)
db_col1, db_col2 = st.columns([1, 1], gap="large")

with db_col1:
    st.markdown("**歷史分類報廢資料庫 (history.csv)**")
    st.dataframe(df_history, use_container_width=True, height=280)
    csv_history = df_history.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="下載歷史分類報廢資料 (CSV)",
        data=csv_history,
        file_name="history_export.csv",
        mime="text/csv"
    )

with db_col2:
    st.markdown("**MAE 預測準確度驗證紀錄庫 (tracking.csv)**")
    st.dataframe(df_tracking, use_container_width=True, height=280)
    csv_tracking = df_tracking.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="下載 MAE 驗證紀錄資料 (CSV)",
        data=csv_tracking,
        file_name="mae_tracking_export.csv",
        mime="text/csv"
    )

# ==========================================
# 側邊欄：門市資料登錄中心
# ==========================================
st.sidebar.markdown("<div class='fm-top-bar'></div>", unsafe_allow_html=True)
st.sidebar.header("門市數據登錄中心")
st.sidebar.caption("值班人員請於每日打烊結算後，於此處登錄當日實際報廢數據以利模型持續迭代訓練。")

# 1. 新增歷史資料
with st.sidebar.expander("新增當日分類報廢紀錄"):
    with st.form("history_form"):
        h_date = st.date_input("結算日期")
        h_wk = st.selectbox("週末", [0, 1], key='h1', format_func=lambda x: "是" if x == 1 else "否")
        h_hol = st.selectbox("連假期間", [0, 1], key='h2', format_func=lambda x: "是" if x == 1 else "否")
        h_pre = st.selectbox("連假前夕", [0, 1], key='h3', format_func=lambda x: "是" if x == 1 else "否")
        h_main = st.number_input("主食便當類", min_value=0, value=0)
        h_light = st.number_input("輕食飯糰手卷", min_value=0, value=0)
        h_hot = st.number_input("熟食小吃類", min_value=0, value=0)
        h_soup = st.number_input("湯品與截切水果", min_value=0, value=0)
        h_dessert = st.number_input("甜點與烘焙麵包", min_value=0, value=0)
        
        if st.form_submit_button("寫入分類報廢資料庫", use_container_width=True):
            new_row = [str(h_date), h_wk, h_hol, h_pre, h_main, h_light, h_hot, h_soup, h_dessert]
            df_history.loc[len(df_history)] = new_row
            df_history.to_csv('history.csv', index=False)
            st.success("分類報廢資料已同步更新，模型已完成重新訓練。")
            st.rerun()
            
    if st.button("刪除最後一筆分類紀錄", use_container_width=True):
        if not df_history.empty:
            df_history = df_history.iloc[:-1]
            df_history.to_csv('history.csv', index=False)
            st.success("已刪除最後一筆分類報廢資料。")
            st.rerun()

# 2. 新增 MAE 戰績資料
with st.sidebar.expander("新增 MAE 驗證對比紀錄"):
    with st.form("tracking_form"):
        t_date = st.date_input("驗證日期", key='t_date')
        t_actual = st.number_input("當日實際總報廢量", min_value=0, value=0)
        t_pred = st.number_input("系統昨日預估總量", min_value=0, value=0)
        
        if st.form_submit_button("寫入 MAE 驗證紀錄庫", use_container_width=True):
            new_track = [str(t_date), t_actual, t_pred]
            df_tracking.loc[len(df_tracking)] = new_track
            df_tracking.to_csv('tracking.csv', index=False)
            st.success("驗證紀錄已更新。")
            st.rerun()
            
    if st.button("刪除最後一筆驗證紀錄", use_container_width=True):
        if not df_tracking.empty:
            df_tracking = df_tracking.iloc[:-1]
            df_tracking.to_csv('tracking.csv', index=False)
            st.success("已刪除最後一筆驗證紀錄。")
            st.rerun()
