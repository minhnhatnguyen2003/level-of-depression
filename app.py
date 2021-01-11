import streamlit as st
import pandas as pd
import numpy as np
import statistics
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

import contractions
from cleantext import clean

from sklearn.ensemble import RandomForestClassifier

st.title('Level of Depression Prediction App')
st.sidebar.header('User Input')
d_text1 = st.sidebar.text_input('A. Write about your feelings for the past 2 weeks', '')
#No matter how I try to make myself forget about the way I feel it always resurfaces and I feel alone. I cried at school all day and people kept asking if I’m okay . I said yes because no matter how I try to explain I feel no one will truly understand. It’s not like I haven’t tried to explain but my feelings either get invalidated or tossed to the side. I want to be happy but my environment is restricting me. I always thought “I can’t be depressed when people have it worse than me” Today I realized that’s incorrect. For the past two days I’ve asked God to take me off this Earth like I did two months ago . Here I am. Alone.
st.sidebar.markdown('B. Rate the extent to which the pair of traits applies to you, even if one characteristic applies more strongly than the other')
st.sidebar.markdown('1:Disagree strongly; 5: Agree strongly')


def user_input():
    a1 = st.sidebar.slider('Extraverted, enthusiastic', 1, 5, 2)
    a2 = st.sidebar.slider('Critical, quarrelsome', 1, 5, 5)
    a3 = st.sidebar.slider('Dependable, self-disciplined', 1, 5, 2)
    a4 = st.sidebar.slider('Anxious, easily upset', 1, 5, 2)
    a5 = st.sidebar.slider('Open to new experiences, complex', 1, 5, 2)
    a6 = st.sidebar.slider('Reserved, quiet', 1, 5, 2)
    a7 = st.sidebar.slider('Sympathetic, warm', 1, 5, 2)
    a8 = st.sidebar.slider('Disorganized, careless', 1, 5, 2)
    a9 = st.sidebar.slider('Calm, emotionally stable', 1, 5, 2)
    a10 = st.sidebar.slider('Conventional, uncreative', 1, 5, 2)

    liO = [1,2,3,4,5,6,7]
    liO_reversed = liO[::-1]
    reversed_num2 = liO_reversed[a2-1]
    reversed_num4 = liO_reversed[a4-1]
    reversed_num6 = liO_reversed[a6-1]
    reversed_num8 = liO_reversed[a8-1]
    reversed_num10 = liO_reversed[a10-1]

    grp1 = [int(a1),reversed_num6]
    grp2 = [int(a7),reversed_num2]
    grp3 = [int(a3),reversed_num8]
    grp4 = [int(a9),reversed_num4]
    grp5 = [int(a5),reversed_num10]

    data ={'EX':statistics.mean(grp1),
            'AG' :statistics.mean(grp2),
            'CON':statistics.mean(grp3),
            'ES' :statistics.mean(grp4),
            'OP' :statistics.mean(grp5)          }
    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input()
st.subheader('Your TIPI Score')
st.write(df1)

EX = 'EX: Extraversion'
AG = 'AG: Agreeableness'
CON = 'CON: Conscientiousness'
ES = 'ES: Emotional Stability'
OP = 'OP: Openness to Experiences'
liOS = [EX,AG,CON,ES,OP]

if st.checkbox('Show acronym meaning'):
    liOS

#def user_input_features():
 #   EX = st.sidebar.slider('EX', 1, 5, 2)
  #  AG = st.sidebar.slider('AG', 1, 5, 2)
   # CON = st.sidebar.slider('CON', 1, 5, 2)
    #ES = st.sidebar.slider('ES', 1, 5, 2)
    #OP = st.sidebar.slider('OP', 1, 5, 2)
    #ata = {'EX': int(EX),
     #       'AG': int(AG),
      #      'CON': int(CON),
       #     'ES': int(ES),
        #    'OP': int(OP)}
    #features = pd.DataFrame(data, index=[0])
    #return features

#df1 = user_input_features()
#st.write(df1)

df =pd.read_csv("C:/ISEF/Depression/KHAO_SAT_tipi - Form Responses 1.csv") 
df.drop(['Timestamp','SUM','Unnamed: 24','Hãy ghi ít nhất 3 câu tâm trạng của bạn bây giờ ( ít nhất 50 từ)','Score','ĐỀ MỤC 1','ĐỀ MỤC 2','ĐỀ MỤC 3','ĐỀ MỤC 4','ĐỀ MỤC 5','ĐỀ MỤC 6','ĐỀ MỤC 7','ĐỀ MỤC 8','ĐỀ MỤC 9','ĐỀ MỤC 10','ĐỀ MỤC 11','ĐỀ MỤC 12','ĐỀ MỤC 13','ĐỀ MỤC 14','ĐỀ MỤC 15','ĐỀ MỤC 16','ĐỀ MỤC 17','ĐỀ MỤC 18','ĐỀ MỤC 19','ĐỀ MỤC 20','ĐỀ MỤC 21','REVERSED R2','REVERSED R4','REVERSED R6','REVERSED R8','REVERSED R10','Hướng ngoại, nhiệt tình (Extraverted, enthusiastic.)','Hay phán đoán, gây gổ (Critical, quarrelsome.)','Có thể được trông cậy, có tính kỉ luật (Dependable, self-disciplined.)','Lo lắng, dễ nổi nóng (Anxious, easily upset.)','Sẵn sàng trải nghiệm, phức tạp (Open to new experiences, complex.)','Kín đáo, im lặng (Reserved, quiet.)','Cảm thông, ấm áp (Sympathetic, warm.)','Bừa bộn, vô tư (Disorganized, careless.)','Bình tĩnh, tâm lí ổn định (Calm, emotionally stable.)','Thông thường, không sáng tạo (Conventional, uncreative.)'], axis=1, inplace=True)
X = df[['EX','AG','CON','ES','OP']]
y = df['LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)


prediction = clf.predict(df1)
#prediction_proba = clf.predict_proba(df1)

st.subheader('Prediction based on personality')
st.write(prediction)
#st.write(prediction_proba)

#######text ########################################################

train_df = pd.read_csv('dataset-3.csv')
train_texts = train_df['body'].tolist()
train_texts = [ str(char).lower() for char in train_texts]

t = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
t.fit_on_texts(train_texts)

alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1
    
# Use char_dict to replace the tk.word_index
t.word_index = char_dict.copy() 
# Add 'UNK' to the vocabulary 
t.word_index[t.oov_token] = max(char_dict.values()) + 1


model = tf.keras.models.load_model('C:/Users/ACER/Desktop/level-of-depression/vietnamese_model.h5')

def predict(text):
	text = contractions.fix(text)

	text = clean(text, 
               fix_unicode = True, to_ascii = True, 
               lower = True, no_line_breaks = True, no_urls = True, 
               replace_with_url = ' ',
               no_punct = True,
               lang = 'en')

	instance = t.texts_to_sequences(text)

	flat_list = []
	for sublist in instance:
		for item in sublist:
			flat_list.append(item)

	flat_list = [flat_list]

	instance = pad_sequences(flat_list, padding='post', maxlen=1014)

	prediction = model.predict(instance)
  
	return prediction

st.subheader('Negativity Prediction based on your snippet')
st.write(predict(d_text1)) # số liệu đầu tiên là dự đoán % nguy cơ không trầm cảm
                        # số liệu thứ hai là dự đoán % nguy cơ trầm cảm
st.write("The label '1' represents the prediction of negativity in your snippet.")
#Thật sự không thể ăn được nữa. Mùi thức ăn thật sự rất khó chịu. Biết là cần phải có sức khỏe để cố gắng học hành cho đàng hoàng nhưng mà chỉ có nước không mùi vị mới nuốt được. Ăn giống như hành động phí thời gian. Kể cả khi rất đói, thật sự chẳng muốn thứ gì cả, chỉ muốn ngủ thật sâu, quên đi cái đói. Mày lại một lần nữa vô tình làm hại bản thân. Thật hài hước. Chỉ là một vết cắt nhỏ thôi mà cũng khiến mày khóc được. Có lẽ vì mày đã muốn được khóc từ ban đầu rồi nhưng đó là một cơ hội tốt để mày có lí do để những giọt nước mắt yếu đuối kia chảy ra từ tâm hồn mỏng manh của mày. Thật mông lung như một trò đùa. Mày chỉ là một đứa trẻ bình thường thôi mà, một đứa con trai bình thường mà mày lại có những suy nghĩ phức tạp như thế này để làm gì? Bọn đồng trang lứa thật sự rất đơn giản nhưng mà mày lại trải qua hết lần đau tim tan vỡ rồi tự cố hành hạ cơ thể bản thân này là như thế nào? Mày thường xuyên mất ngủ vào những chuyện không đáng để ý, mày im lặng những suy nghĩ đó nhưng vẫn không thể dừng được. Mày chẳng mong đợi vào gì cả. Bạn thì chẳng có, tương lai cũng không, thời gian cũng chẳng còn để mày cố gắng làm gì? Mình thật sự không cần lí do để buồn. Năng lực của mình có đến đây thôi, đâu phải cái gì mình cũng làm được. Mày mơ quá lớn mày có biết không? Trái đất sẽ quay như bình thường nếu như mày không có ở đây. Vì vậy việc mày tôn tại ở đây là gì? Mày bất tài, đến cả việc mở mồm ra cũng không xong. Mày biết mày có vấn đề nhưng mà lại cố dấu tránh là như thế nào? Rồi mày bắt đầu tìm đến nơi để nói lên cảm xúc, bất cứ thứ gì để mày có thể thấy ổn hơn dù chỉ trong một phút chốc. Thế nên mày mới tìm đến những sợi tóc xoăn kia của mày đúng không? Những sợi tóc xoăn kia trên đầu mày đặc biệt như mày vậy. Chúng rất dễ nhận ra và bản thân mày rất ghét chúng đúng không? Chúng chẳng bao giờ nghe mày cả. Nhưng mà cảm giác khi m nhổ chúng đi thật sự khá là thoải mái. Việc loại bỏ những sợi tóc đó cảm giác chẳng khác gì mày tự loại bỏ chính bản thân mình vậy. Chúng là tóc sâu, tóc hỏng. Giống như bản thân mày vậy. Mày tiếp tục nhổ những sợi tóc đó cho tới khi vùng da trên đỉnh đầu mày như một khu rừng mới bị khai thác xong vậy, trụi hẳn. Vậy mà mày vẫn không dừng ở đó được hả? Chỉ cần dừng tay lại, không đưa tay lên đầu thôi cũng không làm được. Mày cứ nhổ và cuối cùng chuyển sang cậy da đầu thưa thớt của mày cho đến khi nó chảy máu. Máu chảy từ đầu mày xuống trán mà mày vẫn tiếp tục như một đứa điên vậy? Nhưng mà thật sự việc nhổ và cậy đó đã đem đến cho bản thân mày yên bình dù mày đang tự hủy hoại bản thân. Da đầu mày trở nên đau nức, vuốt tóc thì mày có thể cảm thấy một vùng bị thiếu tóc. Mày đã tự ti rồi mà bây giờ mày đã làm cho bản thân mình xấu xí hơn. Còn gì tệ hơn nữa không? 