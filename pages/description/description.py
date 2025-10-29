import streamlit as st
import pandas as pd

st.write('''
         # AI-Based System for :orange[Motorcycle Rider Helmet-Wearing Detection] :blue[[อธิบาย]]''')
st.divider()

st.write('''## คือ :orange[Model] อะไร ?''')
l,m,r = st.columns(3)
m.image("images/demo.gif", caption="Sample Demo")
st.write('''### :orange[Motorcycle Rider Helmet-Wearing Detector] ถูกพัฒนามาเพื่อ:green[ตรวจจับ]ผู้ที่ :red[\"ไม่สวมใส่หมวกนิรภัย\"]''')

q,w,e = st.columns(3)
st.divider()
urlRobodflow = "https://app.roboflow.com"
st.write('''
         ## About :blue[Dataset]
         1. นำเข้า Datasets ไปที่ [Roboflow](%s) เพื่อเตรียมทำการ label'''% urlRobodflow)

with st.container(border=True):
    a1,a2,a3 = st.columns(3)
    a1.image("images/roboflow1-non.png", caption="non-labeled dataset [1]",)
    a2.image("images/roboflow2-non.png", caption="non-labeled dataset [2]",)
    a3.image("images/roboflow3-non.png", caption="non-labeled dataset [3]",)

st.write("2. label รูปภาพทั้งหมด โดนจะมี class object :blue[2 ประเภท] คือ")
with st.container(border=True):
    c1,c2 = st.columns(2)
    c1.metric("คนขับมอเตอร์ไซค์ ที่:green[สวมใส่หมวกนิรภัย]", "HELMET", border=True, delta_color="off")
    c2.metric("คนขับมอเตอร์ไซค์ ที่:red[ไม่สวมใส่หมวกนิรภัย]", "NO_HELMET", border=True)
    with st.container(border=True):
        b1,b2,b3 = st.columns(3)
        b1.image("images/roboflow1.png", caption="labeled dataset [1]",)
        b2.image("images/roboflow2.png", caption="labeled dataset [2]",)
        b3.image("images/roboflow3.png", caption="labeled dataset [3]",)

st.write('''
         3. เมื่อทำการ label ผลเฉลยของ datasets ทั้งหมดแล้ว ผมจะแบ่ง dataset ออกเป็น :blue[3] part ก็คือ
         - Training     :gray[ส่วนของ data ที่ให้ model เอาไว้เรียนรู้เหมือนกับการ \"ทำแบบฝึกหัด\"]
         - Validation   :gray[\"ฝึกทำข้อสอบ\"]
         - Testing      :gray[\"ทำข้อสอบจริง\"]''')

d1,d2,d3,d4 = st.columns(4)
d1.metric("จำนวนข้อมูลทั้งหมด", "696 รูป", border=True, delta_color="off")
d2.metric("Training", "636 รูป", border=True)
d3.metric("Validation ", "30 รูป", border=True)
d4.metric("Testing ", "30 รูป", border=True)

st.divider()

# MODEL
urlYOLO= "https://docs.ultralytics.com"

st.write("## :blue[Training] Model :orange[Transfer Learning (YOLO)]")

st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            โดยการพัฒนาเพื่อนผมเลยเลือกที่จะใช้เทคนิคที่ได้เรียนมาในห้องเรียน ซึ่งก็คือการเรียนรู้แบบ ***:orange[Transfer Learning]*** โดยเอา pre-trained model ของ [YOLO](%s) มาต่อยอดในการพัฒนาซึ่งมันมีการเรียนรู้ของการมองเห็นแบบพื้นฐานไปก่อนหน้าแล้ว เราแค่เอา datasets รูปภาพคนใส่หมวกกันน็อคยัดใส่ให้มัน แล้วหลังจากนั้น model จะ fine-tune เอง มันจะเรียนรู้ว่า'''% urlYOLO)

a,b,c = st.columns([1,3,1])
b.code('''ไอ้วัตถุที่มีลักษณะกลมๆแข็งๆอยู่บนหัวคน จากที่เคยมาก่อนว่าเห็นเป็น'หัวคน' ---[Transfer Learning]---> ตอนนี้จะต้องเรียกว่า'Helmet' แล้ว''', language="python")

st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            มันจะค่อยๆปรับ internal weights เองทำให้ตรวจจับให้เก่งมากขึ้นและจะปรับโดยการเทียบ loss จากความแตกต่างของ output เมื่อเทียบกับ label ที่เป็นผลเฉลยที่เราตีกรอบไว้ ซึ่ง project นี้จะนำเสนอโดยใช้ pre-trained model 2 ตัวคือ''')    

st.image("images/yolov8.jpg", caption="source : ***https://github.com/ultralytics/ultralytics/issues/189***",)

st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            YOLOv8s และ YOLOv8m มีโครงสร้างพื้นฐานที่:green[เหมือนกัน]ใช้ส่วนประกอบหลักเดียวกัน :orange[(Backbone, Neck, Head)]''')

st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            1.) ***Backbone*** : ทำหน้าที่สกัด features ที่เด่นๆออกจากภาพ โดยจะลดขนาดของภาพไปเรื่อยๆ แล้วจะได้ feature map ที่มีความละเอียดต่างกัน''')
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            2.) ***Neck*** : ทำหน้าที่นำ feature maps จาก Backbone ที่มีความละเอียดต่างกัน (เช่น features ที่จับภาพรวมกว้างๆ กับ features ที่จับรายละเอียดเล็กๆ) มาผสมผสานกัน''')
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            3.) ***Head*** : ทำหน้าที่รับ feature ที่ผสมแล้วมาทำนายผลลัพธ์ (Bounding Boxes, Class, และ Confidence Score)''')


st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
        :red[แตกต่างกันที่] ก็คือ :orange[(Scale)]''')
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            1.) ความกว้าง คือ จำนวน filters หรือ channels ในแต่ละชั้น convolutional''')
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            2.) ความลึก คือ จำนวน layers หรือ blocks ในแต่ละส่วนของ network''')

a,v8s,m,v8m,z = st.columns([2,16,1,16,2])
with v8s:
    with st.container(border=True):
        st.write("### YOLOv8s(Small)")
        st.write("- ความเร็วกว่า แต่ความแม่นยำน้อยกว่า")
        st.write("- ใช้จำนวน Channels/Filters น้อยกว่า  ")
        st.write("- จำนวน  Parameters(11.2 ล้าน) น้อยกว่า")
        st.write("- ใช้จำนวน Channels/Filters มากกว่า")
        st.write("- เหมาะกับการจับหมวกกันน็อคแบบ Real-time บนอุปกรณ์ขนาดเล็ก (Edge devices, Mobile)")

with v8m:
    with st.container(border=True):
        st.write("### YOLOv8m(Medium)")
        st.write("- เร็วน้อยกว่า แต่ความแม่นยำมากกว่า")
        st.write("- ใช้จำนวน Channels/Filters มากกว่า")
        st.write("- จำนวน Parameters(25.9) มากกว่า")
        st.write("- ใช้จำนวน Channels/Filters มากกว่า")
        st.write("- เหมาะกับงานที่ต้องการความแม่นยำสูงขึ้น แต่ยังคงความเร็วไว้ได้ (GPU, Server)")
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
        *** ไม่สามารถแยก model 's' และ 'm' โดยดูแค่โครงสร้างเพราะมันเหมือนกัน จะแยกได้ก็ต่อเมื่อดู "จำนวน" ของบล็อกพวกนั้ันนั้น และ "ขนาด" ของแต่ละบล็อก''')

st.divider()

st.write("## :blue[Testing] & :orange[Evaluation]")

st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            หลังจาก train ทั้ง 2 model จะเอาไปวัดผลว่า model ทำงานได้ดีแค่ไหนกับข้อมูลที่มันไม่เคยเห็นมาก่อนด้วย Precision, Recall, Precission-Recall, F1-Score ''')

# Precision
st.write("### :blue[Precision] Confidence Curve")
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            จาก Confidence Curve จะทำให้เห็นว่าค่า Precision ของ model ที่ v8m มีความเสถียรมากกว่า model ที่ v8s ดังนั้น model ที่ :blue[ทำนายได้แม่น]ย่ำกว่า (:green[v8m] > :red[v8s])''')
x,y,z = st.columns([1,8,1])
with y:
    a,b = st.columns(2)
    a.image("images/v8s/BoxP_curve.png", caption="v8s",)
    b.image("images/v8m/BoxP_curve.png", caption="v8m",)
st.divider()

# Recall
st.write("### :blue[Recall] Confidence Curve")
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            จาก Confidence Curve จะทำให้เห็นว่าค่าของช่วง Recall ที่ 0.2 - 0.8 ของ model v8m น้ันมีค่าที่สูงกว่า model v8s อย่างมีนัยยะสำคัญ ดั้งนั้น model v8m มี:blue[ความสารถในการตรวจจับ] object ที่มากกว่า model v8s (:green[v8m] > :red[v8s])''')
x,y,z = st.columns([1,8,1])
with y:
    a,b = st.columns(2)
    a.image("images/v8s/BoxR_curve.png", caption="v8s",)
    b.image("images/v8m/BoxR_curve.png", caption="v8m",)
st.divider()

# Precision-Recall
st.write("### :blue[Precission-Recall] Confidence Curve")
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            จาก Confidence Curve แสดงให้เห็นได้อย่างชัดเจนว่า พื้นที่ใต้กราฟ(MAP)ของ model v8m นั้นมากกว่าอย่างมีนัยยะสำคัญ ดั้งนั้น สรุปได้ว่า model v8m สามารถ:blue[ใช้งานได้ครอบคลุม]มากกว่า (:green[v8m] > :red[v8s])''')
x,y,z = st.columns([1,8,1])
with y:
    a,b = st.columns(2)
    a.image("images/v8s/BoxPR_curve.png", caption="v8s",)
    b.image("images/v8m/BoxPR_curve.png", caption="v8m",)
st.divider()

# F1-Score
st.write("### :blue[F1-Score] Confidence Curve")
st.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            จาก Confidence Curve แสดงให้เห็นถึงช่วง ค่าสูงสุด ของแต่ละ model ดั้งนั้น จากกราฟของ model v8m ณ ที่ค่า F1 มีค่ามากกว่า 0.4 model v8m สามารถทำช่วงที่กว้างกว่า อย่างมาก และ ค่าสูงสุดของ model v8m มีค่าโดยประมาณที่ 0.6 ซึ่ง มากกว่า model v8s มีค่าสูงสุดโดยประมาณ 0.45-0.5 สรุปได้ว่า model v8m สามารถ:blue[นำมาใช้งานจริง]ได้ดีกว่า (:green[v8m] > :red[v8s])''')
x,y,z = st.columns([1,8,1])
with y:
    a,b = st.columns(2)
    a.image("images/v8s/BoxF1_curve.png", caption="v8s",)
    b.image("images/v8m/BoxF1_curve.png", caption="v8m",)