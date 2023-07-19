import numpy as np
import pandas as pd
import streamlit as st

st.title('測試首頁')
st.header('1st header')
st.write('good job')

a= 100
st.write(a)
st.write('--------------------\n')
st.write('表格')
df = pd.DataFrame({'f1':[1,2,3,4], 'f2':[18,19,21,22]})
st.write(df)

st.subheader('--------------------\n')
st.subheader('核取方塊')
cb = st.checkbox('選我')
if cb:
    st.info('爽快')

st.subheader('--------------------\n')
st.subheader('選項按鈕')
gender = st.radio('性別:',('男生','女生','畜生'))
st.subheader(gender)
st.success(gender)

st.subheader('--------------------\n')
st.subheader('下拉選單')
option = st.selectbox('性別?',('男生','女生','畜生'))
st.subheader(option)
st.success(option)
'你的回答', option

# st.subheader('--------------------\n')
# st.subheader('進度條')
# import time
# b = st.empty()
# bar = st.progress(0)
# for i in range(100):
#     b.text(f'目前進度:{i+1}%')
#     bar.progress(i+1)
#     time.sleep(0.1)

st.subheader('--------------------\n')
def c():
    st.text('已被確認')

st.subheader('按鈕')
btn = st.button('按我')
if btn:
    st.info('已確認')
    c()

st.subheader('--------------------\n')
st.subheader('滑桿')
num = st.slider('請選擇數量',1,20)
'你選的數量:', num

st.subheader('--------------------\n')
st.subheader('檔案上傳')
loader = st.file_uploader('請選擇csv檔:')
# df1 = pd.read_csv(loader, header=None)
# st.table(df1.iloc[:2])

if loader is not None:
    df2 = pd.read_csv(loader, header=None)
    st.dataframe(df2)
    st.table(df2.iloc[:2])

st.subheader('--------------------\n')
st.subheader('隱藏選單')
hidden = st.expander('按下後展開')
hidden.write('Hello! Python!')

st.subheader('--------------------\n')
st.subheader('圖片上傳+圖片展示')
img = st.file_uploader('請選擇圖檔:',type=['png','jpg','jpeg','bmp'])
if img is not None:
    st.image(img)

st.subheader('--------------------\n')
st.subheader('側邊攔')
side1 = st.sidebar.button('slid me')
side2 = st.sidebar.checkbox('check me')

st.subheader('--------------------\n')
st.subheader('分欄')
lift, right = st.columns(2)
lift.write('左邊')
right.write('右邊')