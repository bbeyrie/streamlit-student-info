import streamlit as st

def main():
    # Title
    st.title('User Information App')

    # Input for name
    name = st.text_input('Enter your name')

    # Slider for age
    min_age = st.slider('Minimum Age', min_value=0, max_value=100, value=0)
    max_age = st.slider('Maximum Age', min_value=0, max_value=100, value=100)

    # Button to display user information
    if st.button('Display Information'):
        st.write(f'Name: {name}')
        st.write(f'Age: Between {min_age} and {max_age}')

if __name__ == "__main__":
    main()
    