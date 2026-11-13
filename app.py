import io
import os
import time
import json
import requests
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 系統頁面設定 (企業專業風格)
# ==========================================
st.set_page_config(
    page_title="全家便利商店 報廢預測系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# 開場載入畫面
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
        with st.spinner("系統正在初始化 AI 預測引擎、氣象連線與資料庫，請稍候..."):
            time.sleep(1.2)
    splash.empty()
    st.session_state['first_open'] = True

# ==========================================
# 常數與基本設定 (鎖定：台中市西屯區至善里)
# ==========================================
STORE_LOCATION_NAME = "台中市西屯區至善里"
STORE_LAT = 24.1818
STORE_LON = 120.6406

HISTORY_FILE = 'history.csv'
TRACKING_FILE = 'tracking.csv'
PREDLOG_FILE = 'prediction_log.csv'

TW_TZ = timezone(timedelta(hours=8))  # 伺服器多為 UTC，一律以台灣時間判斷「今天／明天」
TODAY_TW = datetime.now(TW_TZ).date()

targets = ['MainMeal', 'LightMeal', 'HotDeli', 'SoupFruit', 'Dessert']
target_names_zh = {
    'MainMeal': '主食便當類',
    'LightMeal': '輕食飯糰三明治',
    'HotDeli': '小吃熟食類',
    'SoupFruit': '湯品與截切水果',
    'Dessert': '甜點與烘焙麵包'
}
WEEKDAY_ZH = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']

expected_history_cols = ['Date', 'Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'Temp', 'Rain_Prob'] + targets
expected_tracking_cols = ['Date', 'Actual_Waste', 'Recorded_AI_Prediction']
expected_predlog_cols = ['Date'] + targets + ['Total']
FEATURES = ['Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'Weekday', 'Temp', 'Rain_Prob']

# 國定假日／連假清單
HOLIDAYS = {pd.Timestamp(d).date() for d in [
    '2026-09-25', '2026-09-26', '2026-09-27', '2026-09-28',
    '2026-10-09', '2026-10-10', '2026-10-11',
    '2026-10-25', '2026-12-25', '2027-01-01',
]}

# 品項平均進價(NT$/件)
DEFAULT_COST = {'MainMeal': 45, 'LightMeal': 25, 'HotDeli': 30, 'SoupFruit': 25, 'Dessert': 22}

# 預設歷史資料 (含西屯區歷史氣溫 Temp 與降雨機率 Rain_Prob)
# 欄位順序: Date, Is_Weekend, Is_Holiday, Is_PreHoliday, Temp, Rain_Prob, MainMeal, LightMeal, HotDeli, SoupFruit, Dessert
default_history_data = [
    ['2026-09-02', 0, 0, 0, 29, 20, 5, 3, 0, 2, 4],
    ['2026-09-03', 0, 0, 0, 28, 40, 8, 4, 0, 1, 7],
    ['2026-09-04', 0, 0, 0, 29, 30, 2, 10, 1, 1, 7],
    ['2026-09-05', 1, 0, 0, 27, 70, 6, 12, 5, 1, 8],
    ['2026-09-07', 0, 0, 0, 28, 30, 2, 5, 0, 0, 11],
    ['2026-09-08', 0, 0, 0, 29, 20, 4, 5, 0, 2, 7],
    ['2026-09-14', 0, 0, 0, 30, 10, 3, 0, 0, 0, 3],
    ['2026-09-15', 0, 0, 0, 29, 20, 6, 3, 0, 2, 3],
    ['2026-09-16', 0, 0, 0, 28, 30, 2, 8, 0, 4, 5],
    ['2026-09-20', 1, 0, 0, 27, 60, 6, 5, 1, 9, 3],
    ['2026-09-25', 0, 1, 1, 26, 70, 7, 15, 1, 3, 5],
    ['2026-09-27', 1, 1, 0, 26, 80, 9, 5, 1, 4, 9],
    ['2026-09-28', 0, 1, 0, 27, 50, 5, 4, 1, 5, 5],
    ['2026-09-29', 0, 0, 0, 29, 10, 0, 0, 1, 0, 2],
    ['2026-09-30', 0, 0, 0, 28, 20, 7, 2, 0, 1, 2],
    ['2026-10-01', 0, 0, 0, 28, 10, 1, 1, 1, 0, 1],
    ['2026-10-02', 0, 0, 1, 27, 30, 2, 8, 2, 0, 6],
    ['2026-10-04', 1, 0, 0, 25, 60, 5, 8, 0, 1, 12],
    ['2026-10-05', 0, 0, 0, 26, 20, 3, 2, 0, 3, 5],
    ['2026-10-06', 0, 0, 0, 25, 70, 5, 3, 0, 1, 18],
    ['2026-10-07', 0, 0, 0, 26, 20, 2, 4, 0, 2, 5],
    ['2026-10-08', 0, 0, 1, 26, 20, 4, 4, 0, 1, 4],
    ['2026-10-10', 1, 1, 0, 24, 60, 7, 6, 3, 2, 8],
    ['2026-10-11', 1, 1, 0, 25, 40, 5, 5, 1, 3, 4],
    ['2026-10-12', 0, 0, 0, 27, 10, 1, 1, 0, 0, 4],
    ['2026-10-14', 0, 0, 0, 26, 15, 3, 0, 1, 0, 6],
    ['2026-10-15', 0, 0, 0, 25, 30, 1, 6, 1, 1, 8],
    ['2026-10-16', 0, 0, 1, 24, 75, 2, 17, 0, 3, 5],
    ['2026-10-17', 1, 0, 0, 24, 50, 4, 6, 0, 2, 7],
    ['2026-10-18', 1, 0, 0, 25, 20, 3, 1, 0, 0, 6],
    ['2026-10-19', 0, 0, 0, 26, 10, 1, 3, 0, 1, 3],
    ['2026-10-20', 0, 0, 0, 25, 25, 2, 7, 0, 1, 4],
    ['2026-10-22', 0, 0, 0, 24, 20, 2, 7, 0, 2, 2]
]

default_tracking_data = [
    ['2026-10-02', 18, 18], ['2026-10-04', 26, 24], ['2026-10-05', 13, 11],
    ['2026-10-06', 27, 11], ['2026-10-07', 13, 12], ['2026-10-08', 13, 17],
    ['2026-10-10', 26, 26], ['2026-10-11', 18, 24], ['2026-10-12', 6, 12],
    ['2026-10-14', 10, 12], ['2026-10-15', 17, 12], ['2026-10-16', 27, 17],
    ['2026-10-17', 28, 19], ['2026-10-18', 10, 24], ['2026-10-19', 8, 13],
    ['2026-10-20', 14, 13], ['2026-10-22', 13, 14]
]


# ==========================================
# 自動氣象偵測 (台中市西屯區至善里) 與 AI 照片辨識
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_zhishan_weather_map():
    """
    透過 Open-Meteo 免費氣象 API 自動抓取台中市西屯區至善里 (24.1818, 120.6406)
    過去 60 天至未來 16 天的每日平均氣溫 (°C) 與最大降雨機率 (%)，免 API Key。
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={STORE_LAT}&longitude={STORE_LON}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        f"&timezone=Asia%2FTaipei&past_days=60&forecast_days=16"
    )
    try:
        resp = requests.get(url, timeout=6)
        data = resp.json().get('daily', {})
        dates = data.get('time', [])
        t_mean = data.get('temperature_2m_mean', [])
        t_max = data.get('temperature_2m_max', [])
        t_min = data.get('temperature_2m_min', [])
        pops = data.get('precipitation_probability_max', [])
        w_map = {}
        for i, d_str in enumerate(dates):
            # 若 mean 為 None 則取 max 與 min 平均
            temp_val = t_mean[i] if i < len(t_mean) and t_mean[i] is not None else (
                (t_max[i] + t_min[i]) / 2 if i < len(t_max) and t_max[i] is not None and t_min[i] is not None else 25
            )
            pop_val = pops[i] if i < len(pops) and pops[i] is not None else 20
            w_map[d_str] = (int(round(temp_val)), int(round(pop_val)))
        return w_map
    except Exception:
        return {}


def get_weather_for_date(target_date):
    """取得指定日期的西屯區至善里氣溫與降雨機率，若超出預報範圍則回傳季節預設值。"""
    w_map = fetch_zhishan_weather_map()
    d_str = str(target_date)
    if d_str in w_map:
        return w_map[d_str], True
    return (24, 20), False


def analyze_waste_photo_with_gemini(image, gemini_api_key):
    """使用 Google Gemini 視覺模型自動辨識報廢明細照片並歸類為五大類別"""
   
    prompt = """
    這是一張全家便利商店的報廢明細（小白單、螢幕畫面或手寫紀錄）或剩食照片。
    請自動辨識裡面的商品名稱與數量，並將其分類加總為以下五大類：
    1. MainMeal (主食便當類：便當、燴飯、炒飯、義大利麵、涼麵、炒麵、粥等主食)
    2. LightMeal (輕食飯糰三明治：飯糰、手卷、三明治、漢堡、吐司夾心、捲餅、沙拉等)
    3. HotDeli (小吃熟食類：滷味、雞胸肉、炸物、烤物、茶葉蛋、熱狗、包子、關東煮等小吃)
    4. SoupFruit (湯品與截切水果：湯品、茶碗蒸、小菜、截切水果盒等)
    5. Dessert (甜點與烘焙麵包：甜點、布丁、泡芙、蛋糕、袋裝烘焙麵包、土司等)

    請嚴格只回傳 JSON 格式（數值必須為整數），不要有任何額外說明文字，格式如下：
    {
      "MainMeal": 0,
      "LightMeal": 0,
      "HotDeli": 0,
      "SoupFruit": 0,
      "Dessert": 0,
      "Items_Summary": "簡短列出辨識到的商品與數量"
    }
    """
    try:
         client = genai.Client(api_key=gemini_api_key.strip())
         response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[ prompt,image]
         )
         clean_json = response.text.strip().replace('```json', '').replace('```', '').strip()
         return json.loads(clean_json)
    except Exception as e:
        raise RuntimeError(f"AI 模型呼叫失敗：{e}")


# ==========================================
# 資料存取工具 (支援舊版 CSV 自動補齊天氣欄位)
# ==========================================
def clean_by_date(df, cols):
    """統一日期格式、數值型別，依日期去重(保留最後一筆)並排序。"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).round().astype(int)
    df = df.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)
    return df[cols]


def upgrade_legacy_history_df(df):
    """若舊版 CSV 缺少 Temp 或 Rain_Prob 欄位，自動補上預設或歷史氣象值，確保舊資料不遺失。"""
    default_weather_lookup = {row[0]: (row[4], row[5]) for row in default_history_data}
    if 'Temp' not in df.columns:
        df['Temp'] = df['Date'].map(lambda d: default_weather_lookup.get(str(d), (25, 20))[0])
    if 'Rain_Prob' not in df.columns:
        df['Rain_Prob'] = df['Date'].map(lambda d: default_weather_lookup.get(str(d), (25, 20))[1])
    return df


def load_or_seed(path, cols, default_rows):
    """檔案不存在或欄位損毀才以預設資料重建；若為舊版歷史資料則自動升級補齊天氣欄位。"""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed')])
            if path == HISTORY_FILE:
                df = upgrade_legacy_history_df(df)
            if all(c in df.columns for c in cols):
                return clean_by_date(df[cols], cols)
        except Exception:
            pass
    df = clean_by_date(pd.DataFrame(default_rows, columns=cols), cols) if default_rows else pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)
    return df


def upsert_by_date(df, row, cols):
    df = df[df['Date'] != row['Date']]
    df = pd.concat([df, pd.DataFrame([row])[cols]], ignore_index=True)
    return clean_by_date(df, cols)


def save(df, path):
    df.to_csv(path, index=False)


df_history = load_or_seed(HISTORY_FILE, expected_history_cols, default_history_data)
df_tracking = load_or_seed(TRACKING_FILE, expected_tracking_cols, default_tracking_data)
df_predlog = load_or_seed(PREDLOG_FILE, expected_predlog_cols, [])

if len(df_history) < 5:
    st.error("歷史資料少於 5 筆，無法訓練模型，請先於側邊欄匯入或登錄資料。")
    st.stop()


# ==========================================
# 日曆特徵：依日期自動判斷（可手動覆寫）
# ==========================================
def auto_calendar(d):
    wk = int(d.weekday() >= 5)
    hol = int(d in HOLIDAYS)
    pre = int(d.weekday() == 4 or (d + timedelta(days=1)) in HOLIDAYS)
    return wk, hol, pre


# ==========================================
# AI 隨機森林預測引擎 (含日曆特徵 + 氣溫 + 降雨機率)
# ==========================================
def make_X(df):
    X = df[['Is_Weekend', 'Is_Holiday', 'Is_PreHoliday', 'Temp', 'Rain_Prob']].copy()
    X['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
    return X[FEATURES]


def build_models(df, n_estimators=200, n_jobs=-1):
    X = make_X(df)
    return {
        t: RandomForestRegressor(
            n_estimators=n_estimators, criterion='absolute_error',
            min_samples_leaf=3, random_state=42, n_jobs=n_jobs
        ).fit(X, df[t])
        for t in targets
    }


@st.cache_resource(show_spinner=False)
def get_models(csv_text):
    return build_models(pd.read_csv(io.StringIO(csv_text)))


def predict_day(models, wk, hol, pre, d, temp, rain_prob):
    X = pd.DataFrame([[wk, hol, pre, d.weekday(), temp, rain_prob]], columns=FEATURES)
    X_np = X.to_numpy(dtype=np.float32)
    point, tree_sum = {}, 0
    for t in targets:
        m = models[t]
        point[t] = max(0, int(round(m.predict(X)[0])))
        tree_sum = tree_sum + np.array([e.predict(X_np)[0] for e in m.estimators_])
    lo, hi = np.percentile(tree_sum, [10, 90])
    return point, (int(round(lo)), int(round(hi)))


@st.cache_data(show_spinner=False)
def run_backtest(csv_text, start=10):
    """滾動式回測：每天只用「當天以前」的資料訓練，再預測當天，與簡單基準比較。"""
    df = pd.read_csv(io.StringIO(csv_text))
    df['Total'] = df[targets].sum(axis=1)
    rows = []
    for i in range(start, len(df)):
        tr, te = df.iloc[:i], df.iloc[[i]]
        ms = build_models(tr, n_estimators=60, n_jobs=None)
        X = make_X(te)
        rows.append({
            'Date': te['Date'].iloc[0],
            'Actual': te['Total'].iloc[0],
            'Model': sum(m.predict(X)[0] for m in ms.values()),
            'Mean': tr['Total'].mean(),
            'MA7': tr['Total'].tail(7).mean(),
        })
    return pd.DataFrame(rows)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


models = get_models(df_history.to_csv(index=False))

# ==========================================
# 網頁主畫面
# ==========================================
st.markdown("<div class='fm-top-bar'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='fm-header-box'>
    <div class='fm-title'>
        生產與配銷期末｜運物二乙第八組｜Ai報廢預測系統
    </div>
    <div style='color: #009B4C; font-size: 15px; font-weight: 600; margin-top: 6px;'>
        📍 門市定位：{STORE_LOCATION_NAME}（已啟用自動氣象偵測與 AI 視覺明細辨識）
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# --- 左半邊：AI 預測引擎 ---
with col1:
    st.markdown("<div class='section-title'>門市報廢量預測</div>", unsafe_allow_html=True)
    st.caption(f"選擇預測日期後，系統自動抓取「{STORE_LOCATION_NAME}」氣象預報與台灣日曆特徵（皆可手動微調）。")

    p_date = st.date_input("預測日期", value=TODAY_TW + timedelta(days=1), key="p_date")
    a_wk, a_hol, a_pre = auto_calendar(p_date)
    (auto_temp, auto_rain), is_live_weather = get_weather_for_date(p_date)

    weather_status = "即時預報連線成功" if is_live_weather else "超出預報範圍(採用歷史均值)"
    st.caption(
        f"{p_date} {WEEKDAY_ZH[p_date.weekday()]}｜日曆自動判斷：週末={'是' if a_wk else '否'}、"
        f"連假={'是' if a_hol else '否'}、連假前夕={'是' if a_pre else '否'}｜"
        f"氣象({weather_status})：{auto_temp}°C、降雨機率 {auto_rain}%"
    )

    c1, c2, c3 = st.columns(3)
    p_wk = c1.checkbox("週末", value=bool(a_wk), key=f"p_wk_{p_date}")
    p_hol = c2.checkbox("國定連假", value=bool(a_hol), key=f"p_hol_{p_date}")
    p_pre = c3.checkbox("連假前夕(含週五)", value=bool(a_pre), key=f"p_pre_{p_date}")

    wc1, wc2 = st.columns(2)
    p_temp = wc1.slider("預估當日平均氣溫 (°C)", min_value=5, max_value=40, value=int(auto_temp), key=f"p_temp_{p_date}")
    p_rain = wc2.slider("預估當日降雨機率 (%)", min_value=0, max_value=100, value=int(auto_rain), step=5, key=f"p_rain_{p_date}")

    if st.button("執行智能預測運算", use_container_width=True, type="primary"):
        point, (lo, hi) = predict_day(models, int(p_wk), int(p_hol), int(p_pre), p_date, int(p_temp), int(p_rain))
        st.session_state['last_pred'] = {
            'date': str(p_date), 'point': point, 'range': (lo, hi),
            'temp': int(p_temp), 'rain': int(p_rain)
        }
        # 自動記錄「事前預測」：僅限尚未有實際資料的日期，避免事後預測污染驗證
        if str(p_date) not in set(df_history['Date']):
            row = {'Date': str(p_date), **point, 'Total': sum(point.values())}
            df_predlog = upsert_by_date(df_predlog, row, expected_predlog_cols)
            save(df_predlog, PREDLOG_FILE)
            st.session_state['logged'] = True
        else:
            st.session_state['logged'] = False

    lp = st.session_state.get('last_pred')
    if lp:
        point, (lo, hi) = lp['point'], lp['range']
        total_pred = sum(point.values())
        cost_map = {t: st.session_state.get(f'cost_{t}', DEFAULT_COST[t]) for t in targets}
        est_cost = sum(point[t] * cost_map[t] for t in targets)

        st.info(
            f"📅 **{lp['date']}**（🌡️ {lp.get('temp', 25)}°C / ☔ 降雨 {lp.get('rain', 20)}%）"
            f"預估總報廢量：**{total_pred} 件**（參考區間 {lo}–{hi} 件）｜預估報廢成本約 **NT$ {est_cost:,}**"
        )
        if st.session_state.get('logged'):
            st.caption("已自動記錄此預測；待當日實際報廢登錄後，將自動納入準確度驗證。")
        elif lp['date'] in set(df_history['Date']):
            st.caption("該日期已有實際資料，此次預測不列入驗證紀錄。")

        cols = st.columns(5)
        for idx, t in enumerate(targets):
            cols[idx].metric(label=target_names_zh[t], value=f"{point[t]} 件")

        # 各品項風險與建議行動
        risk_rows = []
        for t in targets:
            hist_mean = float(df_history[t].mean())
            ratio = point[t] / max(hist_mean, 0.5)
            if point[t] >= 3 and ratio >= 1.3:
                level, act = "偏高", "建議下修訂貨量；晚班提前啟動即期品折扣(如友善食光)或評估捐贈"
            elif ratio <= 0.7:
                level, act = "偏低", "報廢風險低，留意是否反而有缺貨之機會損失"
            else:
                level, act = "正常", "維持現行訂貨量，持續觀察"
            risk_rows.append({
                '品項': target_names_zh[t], '預測(件)': point[t],
                '歷史平均(件)': round(hist_mean, 1), '風險': level, '建議行動': act
            })
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    with st.expander("品項平均進價設定（用於估算報廢成本）"):
        st.caption("預設值僅為示意，請改為門市實際進貨成本。")
        for t in targets:
            st.number_input(f"{target_names_zh[t]} (NT$/件)", min_value=0, value=DEFAULT_COST[t],
                            key=f"cost_{t}")

# --- 右半邊：模型準確度驗證 ---
with col2:
    st.markdown("<div class='section-title'>模型預測準確度驗證</div>", unsafe_allow_html=True)

    if not df_tracking.empty:
        trk = df_tracking.copy()
        err = trk['Recorded_AI_Prediction'] - trk['Actual_Waste']
        m_mae = err.abs().mean()
        m_bias = err.mean()
        m_wmape = err.abs().sum() / max(trk['Actual_Waste'].sum(), 1) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("平均絕對誤差 MAE", f"{m_mae:.2f} 件")
        m2.metric("平均偏差", f"{m_bias:+.2f} 件", help="正值＝系統平均高估；負值＝平均低估")
        m3.metric("WMAPE", f"{m_wmape:.1f}%", help="總絕對誤差 ÷ 總實際報廢量，不受單日極端值放大")
        m4.metric("驗證天數", f"{len(trk)} 天")

        chart_data = trk.assign(Date=pd.to_datetime(trk['Date'])).set_index('Date')[
            ['Actual_Waste', 'Recorded_AI_Prediction']
        ].rename(columns={'Actual_Waste': '門市實際報廢量', 'Recorded_AI_Prediction': 'AI 系統預測量'})
        st.line_chart(chart_data, color=["#0080C6", "#009B4C"])
    else:
        st.warning("目前尚無驗證資料。")

    with st.expander("回測：模型是否真的比簡單平均更準？"):
        st.caption("滾動式回測：每一天都只用「當天以前」的資料訓練後預測當天，包含平假日與西屯區氣候特徵。")
        if st.checkbox("執行回測（約數秒）", key="run_bt"):
            if len(df_history) < 14:
                st.info("資料不足 14 筆，暫無法回測。")
            else:
                bt = run_backtest(df_history.to_csv(index=False))
                b1, b2, b3 = st.columns(3)
                mm, mb, m7 = mae(bt['Actual'], bt['Model']), mae(bt['Actual'], bt['Mean']), mae(bt['Actual'], bt['MA7'])
                b1.metric("本模型 MAE", f"{mm:.2f} 件", delta=f"{mm - mb:+.2f} vs 歷史平均", delta_color="inverse")
                b2.metric("基準：歷史平均", f"{mb:.2f} 件")
                b3.metric("基準：近7筆平均", f"{m7:.2f} 件")
                st.caption(f"回測樣本 {len(bt)} 天（特徵含週末、連假、星期、西屯區氣溫與降雨機率）。")
                st.line_chart(
                    bt.assign(Date=pd.to_datetime(bt['Date'])).set_index('Date')[['Actual', 'Model', 'Mean']]
                    .rename(columns={'Actual': '實際', 'Model': '模型回測預測', 'Mean': '歷史平均基準'})
                )

st.divider()

# ==========================================
# 側邊欄：門市數據登錄中心 & AI 照片辨識
# ==========================================
st.sidebar.markdown("### 門市數據登錄中心")
st.sidebar.caption(f"📍 門市：{STORE_LOCATION_NAME}\n請於每日晚班打烊結算後，拍照上傳或手動登錄當日實際報廢數據。")

# 0. AI 報廢明細照片自動分類
with st.sidebar.expander("📸 報廢明細照片 AI 自動分類", expanded=True):
    default_gem_key = ""
    try:
        default_gem_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_gem_key = ""

    gem_key = st.text_input(
        "Google Gemini API Key",
        value=default_gem_key,
        type="password",
        help="請前往 Google AI Studio 免費取得 API Key，用於辨識報廢明細照片。"
    )
    uploaded_photo = st.file_uploader("上傳報廢明細或剩食照片", type=["jpg", "jpeg", "png"], key="waste_photo")

    if uploaded_photo is not None:
        img = Image.open(uploaded_photo)
        st.image(img, caption="待辨識明細照片", use_container_width=True)
        if st.button("🔍 AI 自動辨識並帶入數量", use_container_width=True, type="primary"):
            if not gem_key:
                st.warning("請先輸入上方 Google Gemini API Key！")
            else:
                with st.spinner("AI 正在辨識照片中的品項並自動歸類..."):
                    try:
                        parsed = analyze_waste_photo_with_gemini(img, gem_key)
                        st.session_state['ai_parsed_waste'] = {
                            t: int(parsed.get(t, 0)) for t in targets
                        }
                        # 同步更新下方登錄表單的數值
                        cur_h_date = str(st.session_state.get("h_date", TODAY_TW))
                        for t in targets:
                            st.session_state[f"h_{t}_{cur_h_date}"] = int(parsed.get(t, 0))
                        summary_txt = parsed.get("Items_Summary", "")
                        st.success("辨識完成！已自動填入下方「新增／更新當日分類報廢紀錄」。")
                        if summary_txt:
                            st.caption(f"📋 辨識明細：{summary_txt}")
                    except Exception as e:
                        st.error(f"辨識失敗，請確認 API Key 或照片清晰度：{e}")

# 1. 新增／更新當日分類報廢
with st.sidebar.expander("新增／更新當日分類報廢紀錄", expanded=True):
    h_date = st.date_input("結算日期", value=TODAY_TW, key="h_date")
    existing = df_history[df_history['Date'] == str(h_date)]
    (h_auto_temp, h_auto_rain), _ = get_weather_for_date(h_date)

    if not existing.empty:
        st.caption("此日期已有紀錄，儲存後將覆蓋。")
        e = existing.iloc[0]
        d_wk, d_hol, d_pre = int(e['Is_Weekend']), int(e['Is_Holiday']), int(e['Is_PreHoliday'])
        d_temp, d_rain = int(e['Temp']), int(e['Rain_Prob'])
        d_vals = {t: int(e[t]) for t in targets}
    else:
        d_wk, d_hol, d_pre = auto_calendar(h_date)
        d_temp, d_rain = int(h_auto_temp), int(h_auto_rain)
        ai_prefill = st.session_state.get('ai_parsed_waste', {})
        d_vals = {t: int(ai_prefill.get(t, 0)) for t in targets}

    k = str(h_date)  # 以日期作為 key，換日期時預設值才會跟著更新
    h_wk = st.checkbox("週末", value=bool(d_wk), key=f"h_wk_{k}")
    h_hol = st.checkbox("國定連假", value=bool(d_hol), key=f"h_hol_{k}")
    h_pre = st.checkbox("連假前夕(含週五)", value=bool(d_pre), key=f"h_pre_{k}")

    st.caption(f"🌤️ 系統自動偵測「{STORE_LOCATION_NAME}」當日天氣：")
    hw1, hw2 = st.columns(2)
    h_temp = hw1.number_input("氣溫 (°C)", min_value=0, max_value=45, value=d_temp, key=f"h_temp_{k}")
    h_rain = hw2.number_input("降雨機率 (%)", min_value=0, max_value=100, value=d_rain, step=5, key=f"h_rain_{k}")

    h_vals = {
        t: st.number_input(f"{target_names_zh[t]} (件)", min_value=0, value=d_vals[t], key=f"h_{t}_{k}")
        for t in targets
    }

    h_total = sum(h_vals.values())
    hist_totals = df_history[targets].sum(axis=1)
    if h_total > hist_totals.max():
        st.warning(f"本日合計 {h_total} 件，高於歷史最高 {int(hist_totals.max())} 件；"
                   "請確認是否為單次異常(如商品下架、配送異常、檔期結束)，並建議另行註記。")

    if st.button("寫入資料庫並重新訓練模型", use_container_width=True, key="btn_hist"):
        row = {
            'Date': str(h_date),
            'Is_Weekend': int(h_wk),
            'Is_Holiday': int(h_hol),
            'Is_PreHoliday': int(h_pre),
            'Temp': int(h_temp),
            'Rain_Prob': int(h_rain),
            **h_vals
        }
        df_history = upsert_by_date(df_history, row, expected_history_cols)
        save(df_history, HISTORY_FILE)

        msg = "歷史分類資料（含氣候變數）已同步更新。"
        logged = df_predlog[df_predlog['Date'] == str(h_date)]
        if not logged.empty:  # 有事前預測紀錄 → 自動進入驗證
            trow = {'Date': str(h_date), 'Actual_Waste': h_total,
                    'Recorded_AI_Prediction': int(logged.iloc[0]['Total'])}
            df_tracking = upsert_by_date(df_tracking, trow, expected_tracking_cols)
            save(df_tracking, TRACKING_FILE)
            msg += " 已自動比對事前預測並寫入驗證紀錄。"
        st.success(msg)
        time.sleep(0.8)
        st.rerun()

# 2. 手動補登驗證紀錄
with st.sidebar.expander("手動補登預測誤差驗證紀錄"):
    st.caption("一般情況下系統會自動產生；此處僅供補登舊資料。")
    t_date = st.date_input("驗證日期", value=TODAY_TW, key="t_date")
    t_actual = st.number_input("當日實際總報廢量 (件)", min_value=0, value=0, key="t_actual")
    t_pred = st.number_input("系統預估總報廢量 (件)", min_value=0, value=0, key="t_pred")
    if st.button("寫入驗證紀錄", use_container_width=True, key="btn_trk"):
        df_tracking = upsert_by_date(
            df_tracking, {'Date': str(t_date), 'Actual_Waste': t_actual, 'Recorded_AI_Prediction': t_pred},
            expected_tracking_cols)
        save(df_tracking, TRACKING_FILE)
        st.success("驗證紀錄已同步更新。")
        time.sleep(0.8)
        st.rerun()

# 3. 刪除指定日期
with st.sidebar.expander("刪除指定日期的紀錄"):
    which = st.radio("資料類型", ["歷史分類報廢", "預測誤差驗證"], key="del_which")
    src = df_history if which == "歷史分類報廢" else df_tracking
    if len(src) > 0:
        del_date = st.selectbox("選擇日期", sorted(src['Date'], reverse=True), key="del_date")
        sure = st.checkbox("我確認要刪除此日期的紀錄", key="del_sure")
        if st.button("刪除", use_container_width=True, key="btn_del"):
            if not sure:
                st.warning("請先勾選確認。")
            elif which == "歷史分類報廢" and len(df_history) <= 5:
                st.warning("歷史資料至少需保留 5 筆。")
            else:
                if which == "歷史分類報廢":
                    df_history = df_history[df_history['Date'] != del_date].reset_index(drop=True)
                    save(df_history, HISTORY_FILE)
                else:
                    df_tracking = df_tracking[df_tracking['Date'] != del_date].reset_index(drop=True)
                    save(df_tracking, TRACKING_FILE)
                st.success("已刪除。")
                time.sleep(0.8)
                st.rerun()

# 4. 匯入還原
with st.sidebar.expander("匯入備份 CSV（還原／合併資料）"):
    st.caption("伺服器重啟後本機 CSV 可能還原為內建資料，請定期下載備份，需要時由此匯入。同日期以匯入檔為準。")
    up_h = st.file_uploader("歷史分類報廢資料 CSV", type="csv", key="up_h")
    up_t = st.file_uploader("預測誤差驗證資料 CSV", type="csv", key="up_t")
    if st.button("匯入並合併", use_container_width=True, key="btn_imp"):
        done = []
        try:
            if up_h is not None:
                new_h = pd.read_csv(up_h)
                new_h = new_h.drop(columns=[c for c in new_h.columns if str(c).startswith('Unnamed')])
                new_h = upgrade_legacy_history_df(new_h)
                if not all(c in new_h.columns for c in expected_history_cols):
                    raise ValueError("歷史資料欄位不符")
                df_history = clean_by_date(pd.concat([df_history, new_h[expected_history_cols]]), expected_history_cols)
                save(df_history, HISTORY_FILE)
                done.append(f"歷史資料 {len(new_h)} 筆")
            if up_t is not None:
                new_t = pd.read_csv(up_t)
                new_t = new_t.drop(columns=[c for c in new_t.columns if str(c).startswith('Unnamed')])
                if not all(c in new_t.columns for c in expected_tracking_cols):
                    raise ValueError("驗證資料欄位不符")
                df_tracking = clean_by_date(pd.concat([df_tracking, new_t[expected_tracking_cols]]), expected_tracking_cols)
                save(df_tracking, TRACKING_FILE)
                done.append(f"驗證資料 {len(new_t)} 筆")
            if done:
                st.success("已匯入：" + "、".join(done))
                time.sleep(0.8)
                st.rerun()
            else:
                st.info("尚未選擇檔案。")
        except Exception as ex:
            st.error(f"匯入失敗：{ex}")

# ==========================================
# 底部區塊：資料庫檢視與備份匯出
# ==========================================
st.markdown("<div class='section-title'>資料庫明細與備份管理</div>", unsafe_allow_html=True)
st.caption("提醒：部署於 Streamlit Community Cloud 時，本機 CSV 在重啟後會還原；請定期下載備份，"
           "長期使用建議改接 Google Sheets 或雲端資料庫。")
tab1, tab2, tab3 = st.tabs(["歷史分類報廢資料庫 (模型訓練集)", "預測誤差驗證資料庫", "事前預測紀錄"])

with tab1:
    display_hist = df_history.copy()
    display_hist.insert(1, '星期', pd.to_datetime(display_hist['Date']).dt.weekday.map(lambda i: WEEKDAY_ZH[i]))
    display_hist['合計'] = display_hist[targets].sum(axis=1)
    display_hist = display_hist.rename(columns={
        'Date': '日期', 'Is_Weekend': '週末', 'Is_Holiday': '連假', 'Is_PreHoliday': '連假前夕',
        'Temp': '氣溫(°C)', 'Rain_Prob': '降雨機率(%)',
        **target_names_zh
    })
    st.dataframe(display_hist, use_container_width=True, hide_index=True)
    st.download_button("下載歷史分類報廢資料 (CSV)", df_history.to_csv(index=False).encode('utf-8-sig'),
                       file_name="familymart_history_export.csv", mime="text/csv")

with tab2:
    display_track = df_tracking.copy()
    display_track['誤差 (預測-實際)'] = display_track['Recorded_AI_Prediction'] - display_track['Actual_Waste']
    display_track['絕對誤差 (件)'] = display_track['誤差 (預測-實際)'].abs()
    display_track = display_track.rename(columns={
        'Date': '日期', 'Actual_Waste': '實際總報廢量', 'Recorded_AI_Prediction': 'AI 預估總報廢量'})
    st.dataframe(display_track, use_container_width=True, hide_index=True)
    st.download_button("下載預測誤差驗證資料 (CSV)", df_tracking.to_csv(index=False).encode('utf-8-sig'),
                       file_name="familymart_mae_export.csv", mime="text/csv")

with tab3:
    display_log = df_predlog.rename(columns={'Date': '預測日期', 'Total': '預測合計', **target_names_zh})
    st.dataframe(display_log, use_container_width=True, hide_index=True)
    st.download_button("下載事前預測紀錄 (CSV)", df_predlog.to_csv(index=False).encode('utf-8-sig'),
                       file_name="familymart_prediction_log.csv", mime="text/csv")
