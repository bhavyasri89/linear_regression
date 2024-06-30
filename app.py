import streamlit as st
import pickle
import numpy as np


with open('Model.pkl','rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



def predict(bed,bath,size):
    

    input_data=np.array([[bed,bath,size]])

    input_df=scaler.transform(input_data)

    return model.predict(input_df)[0]

if __name__ == '__main__':
    st.header('House Price Prediction')

    col1,col2 =st.columns([2,1])

    bed =  col1.slider('No.Of Bedrooms', max_value=10,min_value=0,value=2)
    bath= col1.slider('No.Of Bathrooms',max_value=10,min_value=0,value=2)
    size= col1.number_input('Size in Sqft',max_value=10000,min_value=500,value=1000,step=250)
    

    result=predict(bed,bath,size)
    col2.image('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.dreamstime.com%2Fphotos-images%2Fhouse-prices.html&psig=AOvVaw26TtZtqbewdU60ldIkEjl1&ust=1719818884565000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCOiSsMrmgocDFQAAAAAdAAAAABAE',use_column_width=True)

    col2.write(f'The Predictedmvalue is :{result} Lakhs')