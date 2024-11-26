#======================== IMPORT PACKAGES ===========================
import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Thyroid Identification using XAI"}</h1>', unsafe_allow_html=True)

###



col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Cytomel (Pro)"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name: liothyronine"}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)

    
with col2:
    
    import streamlit as st
    
    x = st.slider("6.4", 0.0, 10.0, (0.0, 6.4), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"252 reviews"}</h1>', unsafe_allow_html=True)


st.write("------------------------------------------------------------------------")

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Thyquidity"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name: levothyroxine"}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)

with col2:
    
    import streamlit as st
    
    x = st.slider("5.4", 0.0, 10.0, (0.0, 5.4), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"210 reviews"}</h1>', unsafe_allow_html=True)

    
st.write("------------------------------------------------------------------------")


col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Westhroid"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name: thyroid desiccated "}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)

with col2:
    
    import streamlit as st
    
    x = st.slider("7.4", 0.0, 10.0, (0.0, 7.4), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"102 reviews"}</h1>', unsafe_allow_html=True)

    
st.write("------------------------------------------------------------------------")


col1,col2,col3 = st.columns(3)



with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Niva thyroid"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name:  thyroid desiccated "}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)


with col2:
    
    import streamlit as st
    
    x = st.slider("3.2", 0.0, 10.0, (0.0, 3.2), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"97 reviews"}</h1>', unsafe_allow_html=True)




st.write("------------------------------------------------------------------------")

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Nature-Throid"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name:  thyroid desiccated "}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)


with col2:
    
    import streamlit as st
    
    x = st.slider("6.7", 0.0, 10.0, (0.0, 6.7), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"38 reviews"}</h1>', unsafe_allow_html=True)



st.write("------------------------------------------------------------------------")

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Levoxyl(pro)"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name:  levothyroxine "}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)




with col2:
    
    import streamlit as st
    
    x = st.slider("4.8", 0.0, 10.0, (0.0, 4.8), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"23 reviews"}</h1>', unsafe_allow_html=True)




st.write("------------------------------------------------------------------------")


col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"Tirosint(pro)"}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#000000;font-size:15px;">{"Generic name:  levothyroxine"}</h1>', unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"desiccated"}</h1>', unsafe_allow_html=True)



with col2:
    
    import streamlit as st
    
    x = st.slider("5.9", 0.0, 10.0, (0.0, 5.9), 0.5)
 

with col3:    
    st.markdown(f'<h1 style="color:#0000FF;font-size:15px;">{"139 reviews"}</h1>', unsafe_allow_html=True)




