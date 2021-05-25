import streamlit as st
from multiapp import MultiApp
import demo

app = MultiApp()
st.set_page_config(page_title='AttnGAN', initial_sidebar_state = 'auto')

st.sidebar.title("Navigation")
# Add all application here

import subprocess
with open('get_font.sh', 'rb') as file:
    script = file.read()
rc = subprocess.call(script)
app.add_app("Demo", demo.demo_gan)
app.add_app("AttnGAN Explanation", demo.attngan_explained)


# The main app
app.run()