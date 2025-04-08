import streamlit as st
import pickle


model=pickle.load(open('mlr.pkl','rb'))

def mlr_prediction(cylinder,displacement,horsepower,weight,accelaration,model_year):

    c=float(cylinder)
    d=float(displacement)
    h=float(horsepower)
    w=float(weight)
    a=float(accelaration)
    m=float(model_year)
    

    prediction=model.predict([[c,d,h,w,a,m]])
    return abs(prediction[0])



def main():

    # giving the webpage a title
    st.title('Car fuel Efficiency')

    # the following code creates text boxes in which the user can enter
    # the data reqd to make the predictions

    c=st.number_input("Cylinder range")
    d=st.number_input("Displacement range")
    h=st.number_input("Horsepower range")
    w=st.number_input("Weight range")
    a=st.number_input("accelaration range")
    m=st.number_input("model year range")
    result=''

    if st.button('Predict'):
        result=mlr_prediction(c,d,h,w,a,m)
    if result is not None:
        st.success(result)
    else:
        st.error('prediction not valid')


if __name__=='__main__':
    main()