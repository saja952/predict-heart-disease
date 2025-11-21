import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


st.title("🔍 نموذج التنبؤ بخطر الإصابة بأمراض القلب")

@st.cache_data
def load_data():
    df = pd.read_csv("framingham.csv")
    num_cols = ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    cols_to_check = ['totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        df[col] = df[col].clip(lower, upper)
    return df

df = load_data()

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = dict(zip(np.unique(y_train), class_weights))

model = LogisticRegression(random_state=42, max_iter=1000, class_weight=weights)
model.fit(X_train_scaled, y_train)

st.subheader("📝 أدخل بياناتك")

gender = st.selectbox("الجنس (0 = أنثى ، 1 = ذكر)", [0, 1])
age = st.number_input("العمر", min_value=1, max_value=120)
education = st.selectbox(
    "المستوى التعليمي (1–4) - 1 = ابتدائي / 2 = ثانوي / 3 = جامعي / 4 = أعلى من ذلك",
    [1, 2, 3, 4]
)
smoker = st.selectbox("هل أنت مدخن حاليًا؟ (0 = لا ، 1 = نعم)", [0,1])
cigs_per_day = st.number_input("عدد السجائر يوميًا", min_value=0, max_value=80)
bp_meds = st.selectbox("هل تتناول أدوية لضغط الدم؟ (0 = لا ، 1 = نعم)", [0,1])
stroke = st.selectbox("هل أصبت بسكتة دماغية سابقًا؟ (0 = لا ، 1 = نعم)", [0,1])
hypertension = st.selectbox("هل لديك ارتفاع ضغط الدم؟ (0 = لا ، 1 = نعم)", [0,1])
diabetes = st.selectbox("هل لديك مرض السكري؟ (0 = لا ، 1 = نعم)", [0,1])
totChol = st.number_input("الكولسترول الإجمالي", min_value=100, max_value=600)
sysBP = st.number_input("الضغط الانقباضي (SYS)", min_value=60, max_value=250)
diaBP = st.number_input("الضغط الانبساطي (DIA)", min_value=40, max_value=150)
BMI = st.number_input("مؤشر كتلة الجسم BMI", min_value=10.0, max_value=60.0)
heartRate = st.number_input("معدل ضربات القلب", min_value=30, max_value=200)
glucose = st.number_input("مستوى الجلوكوز", min_value=40, max_value=500)


if st.button("🔮 تنبأ بخطر الإصابة"):
    user_data = np.array([[gender, age, education, smoker, cigs_per_day,
                           bp_meds, stroke, hypertension, diabetes,
                           totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    user_scaled = scaler.transform(user_data)

    threshold = 0.30
    probability = model.predict_proba(user_scaled)[0][1]
    prediction = int(probability >= threshold)

    st.subheader("نتيجة التنبؤ")
    if prediction == 1:
        st.error(f"⚠ احتمال الإصابة: {probability*100:.2f}% — هناك خطر أعلى من الطبيعي.")
    else:
        st.success(f"✔ احتمال الإصابة: {probability*100:.2f}% — لا يوجد خطر كبير.")

    st.write("**تنبيه:** هذا التنبؤ مبني على بيانات إحصائية ولا يغني عن الفحص الطبي.")
