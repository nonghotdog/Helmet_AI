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
st.write("2. label รูปภาพทั้งหมด โดนจะมี class object :blue[2 ประเภท] คือ")
c1,c2 = st.columns(2)
c1.metric("คนขับมอเตอร์ไซค์ ที่:green[สวมใส่หมวกนิรภัย]", "HELMET", border=True, delta_color="off")
c2.metric("คนขับมอเตอร์ไซค์ ที่:red[ไม่สวมใส่หมวกนิรภัย]", "NO_HELMET", border=True)
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

st.write('''3. กำหนด batch training ปรับ pixel ให้อยู่ใน range 0-1 และเพิ่มตัว augmentation เพื่อความหลากหลาย จะได้ไม่ overfit ดีกับ unseen data''')
st.code('''
    rescale=1./255,           # range pixel 0-1
    rotation_range=30,        # range หมุนๆ
    width_shift_range=0.2,    # range y
    height_shift_range=0.2,   # range x
    shear_range=0.2,          # range บิดภาพ
    zoom_range=0.2,           # range zoom
    horizontal_flip=True,     # เปิดกลับ imaghe
    fill_mode='nearest'       # เติม pixel
)
''', language="python")

st.write('''4. เหมือนกับ train_datagen ทำการ norm แต่ไม่มีตัว augmentation เพราะจะเอามา valid และ test ''')
st.code('''
    validation_datagen = ImageDataGenerator(rescale=1./255) # ทำ pre ข้อสอบ
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    validation_generator = validation_datagen.flow_from_directory( # ทำ pre ข้อสอบ
        validation_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = validation_datagen.flow_from_directory( # สอบจริง
        test_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
''', language="python")
st.divider()

st.write('''
         ## :blue[Training] Model :orange[Algoritms]
         ผมให้ model เรียนรู้แบบ Transfer Learning โดยใช้ **VGG16(CNN)** (เป็นmodelที่เคยฝึกมาก่อนแล้ว) เพราะประหยัดเวลาและทรัพยากรในการ train modle ส่วนใหญ่จะเน้นไปที่การออกแบบตัว augment
        
         # AGG16 ''')
# st.image("graphic/pic/vgg16.png", caption="source : ***https://www.cs.toronto.edu/~frossard/post/vgg16/***")
st.write('''
        ตามชื่อเลย AGG16 มี 16 layer ใช้ filter ขนาดเล็ก เช่น 3x3 convolution filters และ stride = 1 เนื่องจากการ ขนาดของ filter ที่เล็ก การทำ padding เลยสำคัญ แต่จำนวน layer มีขนาดลึกเลยเอามาหักล้างหรือทดแทนกันได้ และ max pooling 2x2 และ หลังจากโดน convolution และ pooling ข้อมูลที่ได้จะถูกแปลงเป็น vector ไปยัง fully connected เพื่อ output
        แล้ว AGG16 ก็เหมือนกับ deep learning ตัวอื่นๆ ใช้แนวคิด gradient descent กับ backpropagation 

         ***https://www.youtube.com/watch?v=QW7aygOH22I&ab_channel=Wuttipong%E0%B8%A7%E0%B8%B8%E0%B8%92%E0%B8%B4%E0%B8%9E%E0%B8%87%E0%B8%A9%E0%B9%8CKumwilaisak%E0%B8%84%E0%B9%8D%E0%B8%B2%E0%B8%A7%E0%B8%B4%E0%B8%A5%E0%B8%B1%E0%B8%A2%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%94%E0%B8%B4%E0%B9%8C***''')

st.write('''''')
st.write('''
         1. กำหนดค่า VGG16 ไม่ให้ฝึกใน layer ของตัวเอง''')
st.code('''
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False '''
, language="python")

st.write('''2. ต้องแปลงไปเป็น 1D เพื่อจะได้ไปเข้า dense(Fully Connected) ต่อไปได้''')
st.code('''
    # input
    model = Sequential()
    model.add(base_model)
    model.add(Flatten()) '''
, language="python")

st.write('''3. ออกแบบ layer dense กับ dropout และ กัน vanishing gradient''')
st.code('''
    # hindden
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) '''
, language="python")

st.write('''4. หลังจากนั้นจะใช้ softmax ให้ output ออกมาเป็นค่า prob เพราะมีหลายประเภท''')
st.code('''
    # output
    model.add(Dense(15, activation='softmax')) '''
, language="python")

st.write('''5. ผมเลือกที่จะ step น้อยๆ เพราะไม่อยาก overfit และใช้ adam เป็นตัว optimizer เพราะคิดว่าที่ทำอยู่เหมาะกับการ adaptive learning rate ''')
st.code('''
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy']) '''
, language="python")

st.write('''6. ทำการ callback เพื่อให้ save model ที่ดีที่สุดเก็บไว้ และเผื่อกรณีคอมดับเพราะผม train นานมาก เพราะข้อจำกัดของ hardware กับกลัวไฟไหม้หอ''')
st.code('''
    checkPoint = ModelCheckpoint('savePointAnimals.h5', monitor='val_loss', save_best_only=True) '''
, language="python")

st.write('''7. กำหนด epochs แบบเยอะๆไว้ก่อนถ้าไม่ถึง ก็สามารถใช้ที่ callback ไว้''')
st.code('''
    history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkPoint]
) '''
, language="python")

st.write('''8. mornitor เพื่อดู behavior ของ model''')
st.code('''
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples//test_generator.batch_size)
) '''
, language="python")
st.divider()

st.write("# Timeline Accuracy & Hardware Details")
st.video("https://youtu.be/Qon2JsmcW6g?si=NraX6nMCF9BQDPp7")

border, = st.columns(1)
border.metric("Model Accuracy", "81.25 %", border=True)
border.metric("Test Accuracy", "79.62 %", "-1.63 %", border=True)