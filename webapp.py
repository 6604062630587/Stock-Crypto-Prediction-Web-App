import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def fetch_binance_data(symbol="BNBUSDT", interval="1d", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                      "quote_asset_volume", "trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume", "trades"]] = df[[
        "open", "high", "low", "close", "volume", "trades"]].astype(float)
    return df


def add_indicators(df):
    df["MA_7"] = df["close"].rolling(window=7).mean()
    df["MA_14"] = df["close"].rolling(window=14).mean()
    df["EMA_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["EMA_14"] = df["close"].ewm(span=14, adjust=False).mean()
    df["RSI_14"] = 100 - (100 / (1 + (df["close"].diff(1).clip(lower=0).rolling(
        14).mean() / df["close"].diff(1).clip(upper=0).abs().rolling(14).mean())))
    df["Daily_Return"] = df["close"].pct_change() * 100
    df.dropna(inplace=True)
    return df


def train_and_save_model(df, symbol):
    features = ["MA_7", "MA_14", "EMA_7", "EMA_14", "RSI_14", "Daily_Return"]
    target = "close"
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}  # เก็บผลลัพธ์ของแต่ละโมเดล
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "R² Score": r2}

    best_model_name = max(results, key=lambda x: results[x]["R² Score"])
    best_model = models[best_model_name]
    model_filename = f"{symbol.lower()}_best_model.pkl"

    joblib.dump(best_model, model_filename)  # ✅ บันทึกโมเดล
    joblib.dump(scaler, f"{symbol.lower()}_scaler.pkl")  # ✅ บันทึก scaler

    return model_filename, results


st.set_page_config(page_title="Stock & Crypto Prediction Web App", page_icon="📈", layout="wide")


with st.sidebar:
    selected = option_menu("Main Menu", ["🏠 Home", "📊 Cryptocurrency Price Prediction(ML model)", "📈US Stock Price Prediction (NN model)", "📖 Machine Learning Model Development", "📖 Neural Network Model Development"],
                           icons=["house", "graph-up", "graph-up", "book","book"], menu_icon="menu-hamburger", default_index=0)


if selected == "🏠 Home":
    st.title("📈 Stock & Crypto Prediction Web App")
    st.write("""
    ยินดีต้อนรับสู่ **Stock & Cryptocurrency Prediction Web App**  
    เว็บนี้ช่วยให้คุณสามารถ **พยากรณ์ราคาหุ้นและเหรียญคริปโต** ได้โดยใช้ **Machine Learning และ Neural Networks**  
    """)

    # 🔹 **1. ฟีเจอร์หลักของเว็บ**
    st.header("🔹 ฟีเจอร์หลักของเว็บ")
    st.write("""
    - 📊 **พยากรณ์ราคาคริปโต** → ใช้ **Linear Regression, Random Forest และ XGBoost**  
    - 📈 **พยากรณ์ราคาหุ้น** → ใช้ **LSTM และ CNN**  
    - 📖 **เรียนรู้เกี่ยวกับโมเดล Machine Learning & Neural Networks**  
    - 🚀 **ทดลองใช้งานโมเดลแบบเรียลไทม์**  
    """)

    # 🔹 **2. คำแนะนำการใช้งาน**
    st.header("🔹 วิธีการใช้งาน")
    st.write("""
    1️⃣ **ไปที่หน้า "Cryptocurrency Price Prediction"**  
       - ป้อนชื่อเหรียญที่ต้องการวิเคราะห์ เช่น **BNB, DOGE, ETH**  
       - ระบบจะดึงข้อมูลจาก Binance API และทำนายราคาตามโมเดล Machine Learning  

    2️⃣ **ไปที่หน้า "US Stock Price Prediction"**  
       - ป้อนสัญลักษณ์หุ้น เช่น **AAPL, TSLA, MSFT**  
       - เลือกโมเดล **LSTM หรือ CNN** และกดปุ่ม "🔍 ทำนาย"  

    3️⃣ **ศึกษาข้อมูลการพัฒนาโมเดล**  
       - ไปที่หน้า 📖 **Machine Learning Model Development** หรือ **Neural Network Model Development**  
       - ศึกษาวิธีการเตรียมข้อมูล, การพัฒนาโมเดล และผลลัพธ์การทดสอบ  
    """)




elif selected == "📊 Cryptocurrency Price Prediction(ML model)":
    st.title("📊 Cryptocurrency Price Prediction")
    crypto_symbol = st.sidebar.text_input(
        "🔠 ใส่ชื่อย่อเหรียญ (เช่น BNB, DOGE)", value="BNB").upper()

    st.write(f"🔄 กำลังดึงข้อมูล {crypto_symbol} จาก Binance API...")
    df = fetch_binance_data(symbol=f"{crypto_symbol}USDT")
    df = add_indicators(df)

    model_filename = f"{crypto_symbol.lower()}_best_model.pkl"
    if not os.path.exists(model_filename):
        st.write("🚀 กำลังเทรนโมเดลใหม่...")
        model_filename, results = train_and_save_model(df, crypto_symbol)
        model = joblib.load(model_filename)

    model = joblib.load(model_filename)
    st.write("✅ โมเดลโหลดสำเร็จ!")

    st.sidebar.header("🔢 ใส่ค่า Indicator")
    default_values = {"MA_7": df["MA_7"].iloc[-1], "MA_14": df["MA_14"].iloc[-1], "EMA_7": df["EMA_7"].iloc[-1],
                      "EMA_14": df["EMA_14"].iloc[-1], "RSI_14": df["RSI_14"].iloc[-1], "Daily_Return": df["Daily_Return"].iloc[-1]}
    features = {key: st.sidebar.number_input(f"{key}", value=float(
        value), step=0.1) for key, value in default_values.items()}

    if st.button(f"🔮 Predict {crypto_symbol} Price"):
        scaler = joblib.load(f"{crypto_symbol.lower()}_scaler.pkl")
        model = joblib.load(f"{crypto_symbol.lower()}_best_model.pkl")

        X_input_scaled = scaler.transform([list(features.values())])
        predicted_price = model.predict(X_input_scaled)[0]

        st.success(
            f"💰 ราคาที่คาดการณ์ของ {crypto_symbol}: ${predicted_price:.2f}")

    st.subheader(f"📊 กราฟราคาของ {crypto_symbol}")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["close"], label="Actual Price", color="blue")

    # กำหนดรูปแบบของวันที่ให้เหมาะสม
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)  # หมุนวันที่ให้ดูง่ายขึ้น

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # เรียกใช้ฟังก์ชันเทรนโมเดลและรับค่าผลลัพธ์
    model_filename, results = train_and_save_model(df, crypto_symbol)

    # แสดงผล Model Performance เป็นตาราง
    st.subheader("📊 Model Performance Comparison")

    # สร้าง DataFrame สำหรับผลลัพธ์ของโมเดล
    results_df = pd.DataFrame(results).T  # Transpose เพื่อให้ชื่อโมเดลเป็นแถว
    st.dataframe(results_df)

    # แสดงผล Model Performance เป็นกราฟ
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # กราฟ MAE
    ax[0].bar(results.keys(), [res["MAE"]
                               for res in results.values()], color=['red', 'blue', 'green'])
    ax[0].set_title("Mean Absolute Error (MAE)")
    ax[0].set_ylabel("MAE")

    # กราฟ R² Score
    ax[1].bar(results.keys(), [res["R² Score"]
                               for res in results.values()], color=['red', 'blue', 'green'])
    ax[1].set_title("R² Score")
    ax[1].set_ylabel("Score")

    st.pyplot(fig)

elif selected == "📖 Machine Learning Model Development":
    st.title("📖 การพัฒนาโมเดล Machine Learning สำหรับพยากรณ์ราคา BNB")
    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.write("""
การเตรียมข้อมูลเป็นขั้นตอนสำคัญที่ช่วยให้โมเดล Machine Learning สามารถเรียนรู้และพยากรณ์ได้อย่างแม่นยำ โดยในกรณีนี้ เราใช้ข้อมูลราคาของ BNB จาก Binance API และทำการปรับแต่งข้อมูลให้เหมาะสมกับการใช้งาน

### ขั้นตอนหลัก:
1. **ดึงข้อมูลจาก Binance API:** ใช้คำขอ HTTP ผ่าน `requests` เพื่อรับข้อมูลราคาของ BNB/USDT
""")
    st.code("""
def fetch_binance_data(symbol="BNBUSDT", interval="1d", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                      "quote_asset_volume", "trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume", "trades"]] = df[[
        "open", "high", "low", "close", "volume", "trades"]].astype(float)
    return df
""", language="python")
    st.write("""
2. **ทำความสะอาดข้อมูล:** ตรวจสอบค่าที่ขาดหาย (`NaN`), ค่าผิดปกติ (Outliers) และจัดรูปแบบ Timestamp
3. **สร้างฟีเจอร์เพิ่มเติม (Feature Engineering):** คำนวณตัวชี้วัดทางเทคนิค เช่น
   - **Moving Average (MA):** ค่าเฉลี่ยของราคาย้อนหลังเพื่อหาความเคลื่อนไหวของแนวโน้มตลาด
   - **Exponential Moving Average (EMA):** คล้าย MA แต่ให้น้ำหนักมากขึ้นกับค่าล่าสุด
   - **Relative Strength Index (RSI):** วัดว่าหุ้นถูกซื้อมากเกินไป (Overbought) หรือขายมากเกินไป (Oversold)
""")
    st.code("""
def add_indicators(df):
    df["MA_7"] = df["close"].rolling(window=7).mean()
    df["MA_14"] = df["close"].rolling(window=14).mean()
    df["EMA_7"] = df["close"].ewm(span=7, adjust=False).mean()
    df["EMA_14"] = df["close"].ewm(span=14, adjust=False).mean()
    df["RSI_14"] = 100 - (100 / (1 + (df["close"].diff(1).clip(lower=0).rolling(
        14).mean() / df["close"].diff(1).clip(upper=0).abs().rolling(14).mean())))
    df["Daily_Return"] = df["close"].pct_change() * 100
    df.dropna(inplace=True)
    return df
""", language="python")
    st.write("""
4. **การจัดการค่าที่หายไป:** ใช้เทคนิคการเติมค่า (`imputation`) หรือการลบข้อมูลที่ไม่จำเป็น
5. **การ Normalize ข้อมูล:** ใช้ StandardScaler หรือ MinMaxScaler เพื่อทำให้ข้อมูลอยู่ในช่วงที่เหมาะสมสำหรับโมเดล
""")
    st.code("""
def train_and_save_model(df, symbol):
    features = ["MA_7", "MA_14", "EMA_7", "EMA_14", "RSI_14", "Daily_Return"]
    target = "close"
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}  # เก็บผลลัพธ์ของแต่ละโมเดล
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "R² Score": r2}

    best_model_name = max(results, key=lambda x: results[x]["R² Score"])
    best_model = models[best_model_name]
    model_filename = f"{symbol.lower()}_best_model.pkl"

    joblib.dump(best_model, model_filename)  # ✅ บันทึกโมเดล
    joblib.dump(scaler, f"{symbol.lower()}_scaler.pkl")  # ✅ บันทึก scaler

    return model_filename, results
""", language="python")

    st.header("2. ทฤษฎีของอัลกอริทึม (Machine Learning Algorithms)")

    st.subheader("📈 Linear Regression")
    st.write("""
**Linear Regression** ใช้สมการเชิงเส้นเพื่อพยากรณ์ราคาปิดของ BNB โดยอาศัยความสัมพันธ์เชิงเส้นระหว่างฟีเจอร์ (ตัวชี้วัดทางเทคนิค) และราคาปิด

### การประยุกต์ใช้กับข้อมูล BNB:
- เราใช้ค่า **Moving Average (MA), Exponential Moving Average (EMA), และ RSI** เป็นตัวแปรอิสระ (Independent Variables)
- ใช้ **ราคาปิดของ BNB (Close Price)** เป็นตัวแปรตาม (Dependent Variable)
- โมเดลจะเรียนรู้แนวโน้มราคาจากข้อมูลย้อนหลัง และทำนายราคาปิดของวันถัดไป

ข้อจำกัด: Linear Regression เหมาะกับข้อมูลที่มีแนวโน้มเชิงเส้นชัดเจน แต่ตลาดคริปโตมักมีความซับซ้อนและความผันผวนสูง อาจทำให้โมเดลพยากรณ์ได้ไม่แม่นยำเมื่อแนวโน้มไม่เป็นเส้นตรง
""")

    st.subheader("🌲 Random Forest")
    st.write("""
**Random Forest** เป็นอัลกอริทึมที่ใช้การรวมผลของหลาย ๆ Decision Tree เพื่อลดความผิดพลาดและเพิ่มความแม่นยำ

### การประยุกต์ใช้กับข้อมูล BNB:
- ใช้ตัวชี้วัดทางเทคนิค เช่น **MA, EMA, RSI และ Daily Return** เป็นฟีเจอร์
- โมเดลจะสร้างหลาย Decision Tree และรวมผลลัพธ์เพื่อลด Overfitting
- สามารถจับความสัมพันธ์ที่ไม่เป็นเชิงเส้นระหว่างฟีเจอร์และราคาปิดได้

ข้อดีของ Random Forest คือความสามารถในการจัดการกับข้อมูลที่มี Noise และความซับซ้อนได้ดี ทำให้เหมาะกับตลาดคริปโตที่มีความผันผวนสูง
""")

    st.subheader("⚡ XGBoost")
    st.write("""
**XGBoost (Extreme Gradient Boosting)** เป็นอัลกอริทึมที่มีประสิทธิภาพสูงในการพยากรณ์ข้อมูลที่มีความซับซ้อนและมีแนวโน้มไม่เป็นเส้นตรง

### การประยุกต์ใช้กับข้อมูล BNB:
- ใช้ตัวชี้วัดทางเทคนิค เช่น MA, EMA, RSI, และ Daily Return เป็นฟีเจอร์
- ใช้เทคนิค Gradient Boosting เพื่อค่อย ๆ ปรับปรุงผลลัพธ์ของต้นไม้แต่ละต้น ทำให้สามารถจับแนวโน้มที่ซับซ้อนของตลาดคริปโตได้
- มีความสามารถในการจัดการกับค่าที่ขาดหายไป และให้ผลลัพธ์ที่แม่นยำกว่าโมเดลอื่น ๆ

ข้อดี: XGBoost สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนของตลาดและลดข้อผิดพลาดของการพยากรณ์ได้ดี จึงเป็นตัวเลือกที่เหมาะสมสำหรับการทำนายราคาคริปโต
""")

    st.header("3. ขั้นตอนการพัฒนาโมเดล (Model Development Steps)")
    st.subheader("🔹 แบ่งข้อมูลเป็น Train และ Test")
    st.write("""
เราทำการแบ่งข้อมูลออกเป็น 80% สำหรับการฝึกโมเดล (Training Set) และ 20% สำหรับการทดสอบ (Test Set) เพื่อให้สามารถประเมินความสามารถของโมเดลในการพยากรณ์ข้อมูลใหม่ได้
""")
    st.code("""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language="python")

    st.subheader("🔹 การปรับแต่ง Hyperparameter")

    st.write("""
Hyperparameter คือค่าต่าง ๆ ที่เราสามารถกำหนดเองได้เพื่อเพิ่มประสิทธิภาพของโมเดล ตัวอย่างเช่น:
- **Linear Regression:**
  - `fit_intercept`: กำหนดว่าจะคำนวณค่าตัดแกน (Intercept) หรือไม่
  - `normalize`: ระบุว่าข้อมูลควรถูกปรับขนาดก่อนการคำนวณหรือไม่
""")
    st.code("""
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression(fit_intercept=True, normalize=True)
lr_model.fit(X_train, y_train)
""", language="python")
    st.write("""
- **Random Forest:**
  - `n_estimators`: จำนวนต้นไม้ในป่า
  - `max_depth`: ความลึกของต้นไม้แต่ละต้น
  - `min_samples_split`: จำนวนตัวอย่างขั้นต่ำที่ต้องมีเพื่อแบ่งโหนด
""")
    st.code("""
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)
""", language="python")
    st.write("""
- **XGBoost:**
  - `learning_rate`: อัตราการเรียนรู้
  - `n_estimators`: จำนวนรอบของการเรียนรู้
  - `max_depth`: กำหนดความซับซ้อนของโมเดล
""")

    st.code("""
from xgboost import XGBRegressor
xgb_model = XGBRegressor(learning_rate=0.1, n_estimators=200, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
""", language="python")

    st.header("4. การนำโมเดลไปใช้งาน (Deployment & Prediction)")
    st.write("""
เมื่อนำโมเดลไปใช้งาน เราจะ:
- โหลดโมเดลที่ดีที่สุดจากไฟล์ที่บันทึกไว้ (`joblib.load`)
- รับค่าตัวแปรอิสระจากผู้ใช้ผ่าน Streamlit
- ทำการพยากรณ์ราคา BNB โดยใช้โมเดลที่โหลดมา
- แสดงผลลัพธ์เป็นค่าที่พยากรณ์ออกมา พร้อมกราฟเปรียบเทียบกับราคาจริง
""")


elif selected == "📈US Stock Price Prediction (NN model)":
    st.title("📈๊US Stock Price Prediction")
    # เลือกโมเดล
    model_option = st.sidebar.radio("เลือกโมเดลที่ใช้พยากรณ์ราคา:", ["CNN", "LSTM"])
    stock_symbol = st.sidebar.text_input("ป้อนสัญลักษณ์หุ้น (เช่น AAPL, TSLA, MSFT):", value="AAPL").upper()
    
    if st.sidebar.button(f"🔍 ทำนาย {stock_symbol}"):
        with st.spinner(f"กำลังประมวลผลข้อมูลของ {stock_symbol}..."):
            from datetime import datetime
            current_date = datetime.today().strftime('%Y-%m-%d')
            df = yf.download(stock_symbol, start='2010-01-01', end=current_date)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(df)
            
            def create_sequences(data, time_steps=30):
                X, y = [], []
                for i in range(len(data) - time_steps):
                    X.append(data[i:i + time_steps])
                    y.append(data[i + time_steps, 3])  
                return np.array(X), np.array(y)
            
            X, y = create_sequences(scaled_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            if model_option == "CNN":
                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                    BatchNormalization(),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(50, activation='relu'),
                    Dense(1)
                ])
            else:
                model = Sequential([
                    LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', unroll=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', unroll=True),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])
            
            predicted_prices = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted_prices.shape[0], 3)), predicted_prices, np.zeros((predicted_prices.shape[0], 1))), axis=1))[:, 3]
            
            st.success(f"💰 ราคาที่คาดการณ์ของ {stock_symbol}: ${predicted_prices[-1]:.2f}")
            
            st.subheader(f"📊 Actual vs Predicted Stock Prices {stock_symbol}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[-90:], df['Close'].values[-90:], label="Actual Price", color="blue")
            ax.plot(df.index[-90:], predicted_prices[-90:], label="Predicted Price", color="red", linestyle="dashed")
            ax.set_xlabel("Date")
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            ax.annotate(f'Latest: {latest_date}', xy=(df.index[-1], df['Close'].values[-1]), xytext=(-50,10), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, color='black')
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            plt.xticks(rotation=45)
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("📉 Training & Validation Loss")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history.history['loss'], label='Training Loss', color='blue')
            ax.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
            st.pyplot(fig)

elif selected == "📖 Neural Network Model Development":
    st.title("📖 Neural Network Model Development")
    st.write("""
    ในหน้านี้เราจะอธิบาย **แนวคิด, ทฤษฎี, และขั้นตอนการพัฒนาโมเดล Neural Network**  
    สำหรับพยากรณ์ราคาหุ้น AAPL โดยใช้ **LSTM และ CNN**  
    """)

    # 🔹 **1. การเตรียมข้อมูล (Data Preparation)**
    st.header("🔹 1. การเตรียมข้อมูล (Data Preparation)")
    st.write("""
    - ดึงข้อมูลหุ้น AAPL จาก **Yahoo Finance API**  
    - ทำความสะอาดข้อมูล และปรับขนาดข้อมูล (Normalization)  
    """)

    st.code("""
    import yfinance as yf
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # ดึงข้อมูลหุ้น AAPL
    df = yf.download('AAPL', start='2010-01-01', end='2025-03-16')

    # เลือกคอลัมน์ที่สำคัญ
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].interpolate(method='linear')

    # ปรับขนาดข้อมูล
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    """, language="python")

    # 🔹 **2. ทฤษฎีของอัลกอริทึม (LSTM & CNN)**
    st.header("🔹 2. ทฤษฎีของอัลกอริทึม (LSTM & CNN)")

    ## 🔸 LSTM (Long Short-Term Memory)
    st.subheader("🔸 LSTM (Long Short-Term Memory)")
    st.write("""
    **LSTM** เป็น Neural Network ที่ออกแบบมาเพื่อเรียนรู้ข้อมูลแบบลำดับ (Sequential Data)  
    ซึ่งเหมาะสำหรับ Time Series เช่น ราคาหุ้นที่เปลี่ยนแปลงตามเวลา  

    **การประยุกต์ใช้กับข้อมูล AAPL:**  
    - ใช้ข้อมูลย้อนหลัง (เช่น 30 วัน) เพื่อทำนายราคาหุ้นในวันถัดไป  
    - สามารถจับ **แนวโน้มราคาหุ้นในระยะยาว** ได้  
    """)

    ## 🔸 CNN (Convolutional Neural Network)
    st.subheader("🔸 CNN (Convolutional Neural Network)")
    st.write("""
    **CNN** ใช้ Convolutional Layers ในการเรียนรู้ **ลักษณะเฉพาะของข้อมูล**  
    แม้ CNN มักใช้กับข้อมูลรูปภาพ แต่สามารถใช้กับ **Time Series** ได้โดยใช้ **Conv1D**  

    **การประยุกต์ใช้กับข้อมูล AAPL:**  
    - ใช้ CNN วิเคราะห์รูปแบบของการเคลื่อนไหวของราคา  
    - ดึงลักษณะเด่นของข้อมูลหุ้น เช่น **แนวโน้มราคา และปริมาณการซื้อขาย**  
    """)
    
    # 🔹 **3. ขั้นตอนการพัฒนาโมเดล (Model Development Steps)**
    st.header("🔹 3. ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    ในส่วนนี้ เราจะอธิบายรายละเอียดเกี่ยวกับการพัฒนาโมเดล Neural Network  
    ตั้งแต่การเตรียมข้อมูล จนถึงการเทรน และประเมินผลโมเดล  
    """)



    ## 🔸 3.1 สร้างชุดข้อมูล Time Series
    st.subheader("🔹 3.1 สร้างชุดข้อมูล Time Series")
    st.write("""
    - ข้อมูลหุ้นเป็นข้อมูลแบบ **Time Series** (ข้อมูลที่เปลี่ยนแปลงตามเวลา)  
    - โมเดล LSTM และ CNN ต้องการ **ข้อมูลย้อนหลัง** เพื่อใช้เป็นอินพุต  
    - เราจะใช้ **30 วันย้อนหลัง** เพื่อช่วยพยากรณ์ราคาหุ้นในอนาคต  
    """)

    st.code("""
    import numpy as np

    def create_sequences(data, time_steps=30):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps, 3])  # ราคาปิด
        return np.array(X), np.array(y)

    # สร้างชุดข้อมูล
    X, y = create_sequences(scaled_data, time_steps=30)
    """, language="python")

    ## 🔸 3.2 แบ่งข้อมูลเป็น Train & Test
    st.subheader("🔹 3.2 แบ่งข้อมูลเป็น Train & Test")
    st.write("""
    - ข้อมูลทั้งหมดจะถูกแบ่งเป็น **2 ส่วน:**  
      1. **Training Set (80%)** → ใช้ในการฝึกโมเดล  
      2. **Testing Set (20%)** → ใช้ในการทดสอบความสามารถของโมเดล  
    - เราใช้ `shuffle=False` เพื่อรักษาลำดับของข้อมูล Time Series  
    """)

    st.code("""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    """, language="python")

    ## 🔸 3.3 สร้างและคอมไพล์โมเดล LSTM
    st.subheader("🔹 3.3 สร้างและคอมไพล์โมเดล LSTM")
    st.write("""
    - **LSTM (Long Short-Term Memory)** เป็นโมเดลที่สามารถจดจำข้อมูลในระยะยาวได้ดี  
    - เราใช้ **2 LSTM Layers** และเพิ่ม **Dropout 20%** เพื่อลด Overfitting  
    - ใช้ **Mean Squared Error (MSE)** เป็น Loss Function  
    """)

    st.code("""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, 5)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    """, language="python")

    ## 🔸 3.4 สร้างและคอมไพล์โมเดล CNN
    st.subheader("🔹 3.4 สร้างและคอมไพล์โมเดล CNN")
    st.write("""
    - **CNN (Convolutional Neural Network)** ช่วยดึงรูปแบบของข้อมูลหุ้น  
    - ใช้ **Conv1D Layer** เพื่อตรวจจับแนวโน้มของราคา  
    - ใช้ **MaxPooling1D** เพื่อลดขนาดข้อมูลก่อนส่งเข้า Fully Connected Layer  
    """)

    st.code("""
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

    cnn_model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 5)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    cnn_model.compile(optimizer='adam', loss='mean_squared_error')
    """, language="python")

    ## 🔸 3.5 เทรนโมเดล
    st.subheader("🔹 3.5 เทรนโมเดล")
    st.write("""
    - ใช้ **Early Stopping** เพื่อลด Overfitting โดยหยุดเทรนหาก `val_loss` ไม่ลดลง  
    - เทรนโมเดล **LSTM และ CNN** โดยใช้ **50 epochs**  
    """)

    st.code("""
    from tensorflow.keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # เทรนโมเดล LSTM
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # เทรนโมเดล CNN
    cnn_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])
    """, language="python")

    ## 🔸 3.6 ประเมินผลโมเดล
    st.subheader("🔹 3.6 ประเมินผลโมเดล")
    st.write("""
    - ใช้ **MAE (Mean Absolute Error)** และ **R² Score** วัดความแม่นยำของโมเดล  
    - ค่า **MAE ต่ำ** และ **R² ใกล้ 1** แสดงว่าโมเดลทำงานได้ดี  
    """)

    st.code("""
    from sklearn.metrics import mean_absolute_error, r2_score

    y_pred = lstm_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"LSTM - MAE: {mae:.4f}, R²: {r2:.4f}")
    """, language="python")
    
    # 🔹 **4. การนำโมเดลไปใช้งาน (Deployment & Prediction)**
    st.header("🔹 4. การนำโมเดลไปใช้งาน (Deployment & Prediction)")
    st.write("""
    เมื่อนำโมเดลไปใช้งาน เราจะ:
    - **รับค่าตัวแปรอิสระจากผู้ใช้ผ่าน Streamlit**
    - **ทำการพยากรณ์ราคาหุ้นโดยใช้โมเดลที่ผู้ใช้เลือกมา**
    - **แสดงผลลัพธ์เป็นค่าที่พยากรณ์ออกมา พร้อมกราฟเปรียบเทียบกับราคาจริง**
    """)
