import streamlit as st
import pandas as pd

st.write('''
         # AI-Based System for :orange[Motorcycle Rider Helmet-Wearing Detection] :blue[[อธิบาย]]''')
st.divider()

st.write('''## คือ :orange[Model] อะไร ?''')
l,m,r = st.columns(3)
m.image("images/roboflow1-non.png", caption="จะใส่ .gif", width=400)
st.write('''### :orange[Motorcycle Rider Helmet-Wearing Detector] ถูกพัฒนามาเพื่อ:green[ตรวจจับ]ผู้ที่ :red[\"ไม่สวมใส่หมวกนิรภัย\"]''')

q,w,e = st.columns(3)
st.divider()
urlRobodflow = "https://app.roboflow.com"
st.write('''
         ## about :blue[Dataset]
         1. นำเข้า Datasets ไปที่ [Roboflow](%s) เพื่อเตรียมทำการ label'''% urlRobodflow)
a1,a2,a3 = st.columns(3)
a1.image("images/roboflow1-non.png", caption="non-labeled dataset [1]",)
a2.image("images/roboflow2-non.png", caption="non-labeled dataset [2]",)
a3.image("images/roboflow3-non.png", caption="non-labeled dataset [3]",)
st.write("2. label รูปภาพทั้งหมด โดนจะมี class object :blue[3 ประเภท] คือ")
c1,c2,c3 = st.columns(3)
c1.metric("คนขับมอเตอร์ไซค์ ที่:green[สวมใส่หมวกนิรภัย]", "HELMET", border=True, delta_color="off")
c2.metric("คนขับมอเตอร์ไซค์ ที่:red[ไม่สวมใส่หมวกนิรภัย]", "NO_HELMET", border=True)
c3.metric("มอเตอร์ไซค์", "MOTOCYCLE", border=True)
b1,b2,b3 = st.columns(3)
b1.image("images/roboflow1.png", caption="labeled dataset [1]",)
b2.image("images/roboflow2.png", caption="labeled dataset [2]",)
b3.image("images/roboflow3.png", caption="labeled dataset [3]",)

st.write('''
         3. เมื่อทำการ label ผลเฉลยของ datasets ทั้งหมดแล้ว ผมจะแบ่ง dataset ออกเป็น :blue[3] part ก็คือ
         - Training     :gray[ส่วนของ data ที่ให้ model เอาไว้เรียนรู้เหมือนกับการ \"ทำแบบฝึกหัด\"]
         - Validation   :gray[\"ฝึกทำข้อสอบ\"]
         - Testing      :gray[\"ทำข้อสอบจริง\"]''')